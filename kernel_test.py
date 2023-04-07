import torch
import int4matmul
import numpy as np
import time

################################################################################

# sparsity structure; this is # of total weights (non-zero weights is half that)
SPARSE_BLOCK_SIZE = 32
GROUP_QUANTIZATION_SIZE = 128

torch.manual_seed(123)
np.random.seed(123)
FLOAT_T = torch.float16

################################################################################

# debugging options; you only really need to touch these if you're doing
# kernel development

# only run first test case for 5 iterations (6 total)
PROFILING_MODE = False
# ignore errors
I_LIVE_LIFE_ON_THE_EDGE = False
# drop into a debugger session if error is too high
BREAK_ON_ERROR = False
# what it says on the tin
PRINT_KERNEL_OUTPUTS = False

# different parts of data to randomize
RANDOMIZE_WEIGHTS = True
RANDOMIZE_ZEROPOINTS = True
RANDOMIZE_SCALES = True
RANDOMIZE_STATES = True
RANDOMIZE_SPARSE_MASK = True

################################################################################

# TODO: enable testing of dynamic exponent

def _pack_sparse_matrices(hidden_dim, mtx_size, weights, sparse_mask):
    f_Wq = torch.empty((mtx_size//16, hidden_dim), dtype=torch.int, device="cuda:0")
    f_sparse_q = torch.empty((mtx_size//32, hidden_dim), dtype=torch.int, device="cuda:0")
    int4matmul.weight_matrix_packing(
        f_Wq,
        f_sparse_q,
        weights.to(torch.uint8).contiguous(),
        sparse_mask.to(torch.uint8)
    )
    return f_Wq, f_sparse_q

def generate_dequantized_mtx(weights, zeros, scales, n_groups):
    if len(zeros.shape) == 1:
        W = ((weights - zeros[:, None]) * scales[:, None]).T
        hiW = ((weights.to(torch.float32) - zeros[:, None].to(torch.float32)) * scales[:, None].to(torch.float32)).T
    else:
        W = weights.clone().to(torch.float16)
        hiW = weights.clone().to(torch.float32)

        for i in range(n_groups):
            slc = slice(i*GROUP_QUANTIZATION_SIZE, (i+1)*GROUP_QUANTIZATION_SIZE)
            z = zeros[i, :][:, None]
            s = scales[i, :][:, None]

            W[:, slc] = (W[:, slc] - z)*s
            hiW[:, slc] = (hiW[:, slc] - z.to(torch.float32))*s.to(torch.float32)

        W = W.T
        hiW = hiW.T

    return W, hiW


def generate_weights_matrix(hidden_dim, mtx_size, sparse=False, dtype=torch.float16):
    """
    Generates a random weights matrix of (mtx_size x hidden_dim) size.

    Returns three items:
        * a fp16 matrix (values are dequantized weights)
        * a fp32 matrix (values are dequantized weights, but deqquantization
                         is done at higher precision)
        * a 4-tuple with elements:
            1. an int32 packed matrix
            2. a matrix/vector of zeros
            3. a matrix/vector of scales
            4. optionally: an int32 packed matrix containing the sparse mask
    """

    b_Wq = torch.randint(0, 16, (hidden_dim, mtx_size), dtype=torch.int, device="cuda:0")

    if GROUP_QUANTIZATION_SIZE in (-1, None):
        n_groups = 1
        shape = (hidden_dim,)
    else:
        n_groups = int(np.ceil(mtx_size / GROUP_QUANTIZATION_SIZE))
        shape = (n_groups, hidden_dim,)

    zeros = torch.randint(0, 16, shape, dtype=torch.int, device="cuda:0").to(torch.float16)
    scales = ((torch.rand(shape, dtype=dtype, device="cuda:0") * 1.0) + 0.01)

    if not RANDOMIZE_WEIGHTS:
        b_Wq = (b_Wq)*0 + 1
    if not RANDOMIZE_ZEROPOINTS:
        zeros = zeros*0 + 0.0
    if not RANDOMIZE_SCALES:
        scales = scales*0.0 + 1.0

    if sparse:
        sparse_mask = torch.ones((hidden_dim, mtx_size), dtype=dtype, device="cuda:0")

        # TODO: test cases where dim isn't multiple of sparse block
        assert mtx_size % SPARSE_BLOCK_SIZE == 0

        itms = np.arange(SPARSE_BLOCK_SIZE).astype(int)

        # pregenerate "random" sparse masks; we don't need actual random masks
        # to test the kernel anyway
        RANDOM_OPTIONS = 1024
        if RANDOMIZE_SPARSE_MASK:
            N_CHOICES = np.array([
                np.random.choice(itms, size=(SPARSE_BLOCK_SIZE//2,), replace=False)
                for _ in range(RANDOM_OPTIONS)
            ])
        else:
            N_CHOICES = np.array([
                # np.arange(0, SPARSE_BLOCK_SIZE, 2)
                np.arange(SPARSE_BLOCK_SIZE//2, SPARSE_BLOCK_SIZE, 1)
                for _ in range(RANDOM_OPTIONS)
            ])

        rstr = np.random.randint(0, RANDOM_OPTIONS, size=(hidden_dim, mtx_size//SPARSE_BLOCK_SIZE))
        rrange = np.arange(0, mtx_size, SPARSE_BLOCK_SIZE, dtype=int)
        idxes = torch.tensor((N_CHOICES[rstr][:, :, :] + rrange[None, :, None]).reshape((hidden_dim, -1)), device=sparse_mask.device, dtype=torch.int64)
        sparse_mask.scatter_(1, idxes, 0)
    else:
        sparse_q = None

    W, hiW = generate_dequantized_mtx(b_Wq, zeros, scales, n_groups)

    if sparse:
        Wq, sparse_q = _pack_sparse_matrices(hidden_dim, mtx_size, b_Wq, sparse_mask)

        sparse_mask = sparse_mask.T.to(torch.int32)
        W *= sparse_mask.to(dtype)
        hiW *= sparse_mask.to(torch.float32)
    else:
        b_Wq = b_Wq.T

        Wq = torch.empty((mtx_size//8, hidden_dim), dtype=torch.int, device="cuda:0")

        for i in range(0, b_Wq.shape[0], 8):
            write_idx = i // 8

            v = b_Wq[i, :].clone()
            v |= b_Wq[i+1, :] << 4
            v |= b_Wq[i+2, :] << 8
            v |= b_Wq[i+3, :] << 12
            v |= b_Wq[i+4, :] << 16
            v |= b_Wq[i+5, :] << 20
            v |= b_Wq[i+6, :] << 24
            v |= b_Wq[i+7, :] << 28
            Wq[write_idx, :] = v

    return W, hiW, (Wq, zeros, scales, sparse_q)

def generate_hidden_states(seq_len, mtx_size, batch_size=1, dtype=torch.float16):
    states = ((torch.rand((batch_size, seq_len, mtx_size), dtype=dtype, device="cuda:0") * 2.0) - 1)
    if not RANDOMIZE_STATES:
        states = (states*0.0) + 1.0
    return states

################################################################################

def fp16_fp16_matmul(W, x):
    return x @ W

def fp32_fp32_matmul(W, x):
    return x.to(torch.float32) @ W

def fp16_int4_matmul(Wq, x, scales, zeros, sparse_q):
    outs = torch.zeros((x.shape[0], x.shape[1], Wq.shape[1]), dtype=FLOAT_T, device="cuda:0")
    int4matmul.quant_int4_linear_mult(
        outs, Wq, x, scales, zeros, GROUP_QUANTIZATION_SIZE, sparse_q
    )
    return outs

def abs_error(truth, candidate):
    return ((candidate - truth) / truth).abs().nanmean()

################################################################################

def run_test_case(batch_size, mtx_size, hidden_dim, seq_len, sparse, dtype, n_repeats=100):
    error = {}
    runtimes = {}
    avg_runtime = {}

    W, hiW, (Wq, zeros, scales, sparse_q) = generate_weights_matrix(
        hidden_dim, mtx_size, sparse, dtype=dtype)
    x = generate_hidden_states(seq_len, mtx_size, batch_size, dtype=dtype)

    mm_ff = fp16_fp16_matmul(W, x)
    mm_ff_32 = fp32_fp32_matmul(hiW, x)
    mm_if = fp16_int4_matmul(Wq, x, scales, zeros, sparse_q)

    error["fp16,fp16"] = abs_error(mm_ff_32, mm_ff)
    error["int4,fp16"] = abs_error(mm_ff_32, mm_if)

    if (PRINT_KERNEL_OUTPUTS):
        print(mm_ff)
        print(mm_if)
        print(error)

    if BREAK_ON_ERROR and (error["int4,fp16"] > 0.1):
        import pdb; pdb.set_trace()

    N_TRIALS = 5
    if PROFILING_MODE:
        N_TRIALS = 1
        n_repeats = 5

    for name, func in [
        ("fp16,fp16", lambda: fp16_fp16_matmul(W, x)),
        ("int4,fp16", lambda: fp16_int4_matmul(Wq, x, scales, zeros, sparse_q)),
    ]:
        times = []
        for _ in range(N_TRIALS):
            start = time.time()
            for _ in range(n_repeats):
                func()
            torch.cuda.synchronize("cuda:0")
            end = time.time()
            times.append(end-start)

        runtimes[name] = times
        avg_runtime[name] = sum(times) / (N_TRIALS * n_repeats)

    return (error, avg_runtime, runtimes)

################################################################################

if __name__ == "__main__":
    TRIALS = [
        # batch size, mtx size, hidden dim, seq len, sparse
        (1, 128, 128, 2048, False),
        (1, 128, 128, 2048, True),
        (1, 1024, 1024, 2048, False),
        (1, 1024, 1024, 2048, True),
        (1, 20480, 5120, 2048, False),
        (1, 20480, 5120, 2048, True),
        (1, 5120, 20480, 2048, False),
        (1, 5120, 20480, 2048, True),
        (1, 1024, 1024, 1, False),
        (1, 5120, 5120, 1, False),
        (1, 16384, 16384, 1, False),
        (1, 20480, 5120, 1, False),
        (1, 5120, 20480, 1, False),
        (1, 5120, 5120, 1, True),
        (1, 5120, 20480, 1, True),
        (1, 16384, 16384, 1, True),
        (1, 20480, 5120, 1, True),
    ]

    runtime_table = []

    if PROFILING_MODE:
        TRIALS = TRIALS[:1]

    for trial_params in TRIALS:
        batch_size, mtx_size, hidden_dim, seq_len, is_sparse = trial_params

        error, avg_runtime, runtimes = run_test_case(batch_size, mtx_size, hidden_dim, seq_len, is_sparse, dtype=FLOAT_T)

        max_error = max(abs(e) for e in error.values())

        # we define a serious error as:
        # 1. either BOTH cases have finite error (inf pops up if fp32 precision
        #    gives a 0 in the output), AND error exceeds 10%
        # 2. OR: int4-fp16 does not have finite error but fp16 does
        # (if fp16 has non-finite error, we don't care)
        is_probably_serious = (
            torch.isfinite(error["fp16,fp16"])
            and torch.isfinite(error["int4,fp16"])
            and max_error > 0.1
        ) or (torch.isfinite(error["fp16,fp16"]) and not torch.isfinite(error["int4,fp16"]))

        if is_probably_serious:
            if not I_LIVE_LIFE_ON_THE_EDGE:
                raise ValueError("Test case %r had test error %r; stopping" % (trial_params, error))
            else:
                print("Test case %r had test error %r; but let's keep going anyway and see what happens" % (trial_params, max_error))

        for k,v in avg_runtime.items():
            l, r = k.split(",")

            runtime_table.append(
                [
                    batch_size,
                    mtx_size,
                    hidden_dim,
                    seq_len,
                    ["", "16:32"][is_sparse],
                    l,
                    r,
                    "%.2f" % (v * 1000.0 * 1000.0)
                ]
            )

    # print out a markdown table for our data
    lines = [[] for x in range(len(runtime_table) + 2)]

    c = zip(*runtime_table)
    col_names = [
        "B", "S", "N", "M", "", "", "", "Time (µs)",
    ]
    for col, col_name in zip(c, col_names):
        col = [str(x) for x in col]

        max_len = max([len(x) for x in col] + [len(col_name)]) + 3

        lines[0].append(col_name.center(max_len))
        lines[1].append(("-"*(max_len-2)).center(max_len))

        for i, v in enumerate(col):
            # TODO: % loss relative-to-fp16 calculation
            lines[i+2].append(v.rjust(max_len-1) + " ")

    print("\n".join("|%s|" % "|".join(line) for line in lines))