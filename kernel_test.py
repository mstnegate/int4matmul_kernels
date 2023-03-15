import torch
import int4matmul
import numpy as np
import time

torch.manual_seed(123)
np.random.seed(123)

FLOAT_T = torch.float16
SPARSE_BLOCK_SIZE = 32

################################################################################

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
    zeros = ((torch.rand((hidden_dim,), dtype=dtype, device="cuda:0") * 2.0) - 1)
    scales = ((torch.rand((hidden_dim,), dtype=dtype, device="cuda:0") * 2.0) - 1)

    if sparse:
        sparse_mask = torch.ones((hidden_dim, mtx_size), dtype=dtype, device="cuda:0")

        # TODO: test cases where dim isn't multiple of sparse block
        assert mtx_size % SPARSE_BLOCK_SIZE == 0

        itms = np.arange(SPARSE_BLOCK_SIZE).astype(int)

        # pregenerate "random" sparse masks; we don't need actual random masks
        # to test the kernel anyway
        RANDOM_OPTIONS = 1024
        N_CHOICES = np.array([
            np.random.choice(itms, size=(SPARSE_BLOCK_SIZE//2,), replace=False)
            for _ in range(RANDOM_OPTIONS)
        ])

        rstr = np.random.randint(0, RANDOM_OPTIONS, size=(hidden_dim, mtx_size//SPARSE_BLOCK_SIZE))
        rrange = np.arange(0, mtx_size, SPARSE_BLOCK_SIZE, dtype=int)
        idxes = torch.tensor((N_CHOICES[rstr][:, :, :] + rrange[None, :, None]).reshape((hidden_dim, -1)), device=sparse_mask.device, dtype=torch.int64)
        sparse_mask.scatter_(1, idxes, 0)
    else:
        sparse_q = None

    W = ((b_Wq - zeros[:, None]) * scales[:, None]).T
    hiW = ((b_Wq.to(torch.float32) - zeros[:, None].to(torch.float32)) * scales[:, None].to(torch.float32)).T

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
    return ((torch.rand((batch_size, seq_len, mtx_size), dtype=dtype, device="cuda:0") * 2.0) - 1)

################################################################################

def fp16_fp16_matmul(W, x):
    return x @ W

def fp32_fp32_matmul(W, x):
    return x.to(torch.float32) @ W

def fp16_int4_matmul(Wq, x, scales, zeros, sparse_q):
    outs = torch.zeros((x.shape[0], x.shape[1], Wq.shape[1]), dtype=FLOAT_T, device="cuda:0")
    int4matmul.quant_int4_linear_mult(
        outs, Wq, x, scales, zeros, sparse_q
    )
    return outs

def abs_error(truth, candidate):
    return ((candidate - truth) / truth).abs().nanmean()

################################################################################

def run_test_case(mtx_size, hidden_dim, seq_len, sparse, dtype, n_repeats=100):
    error = {}
    runtimes = {}
    avg_runtime = {}

    W, hiW, (Wq, zeros, scales, sparse_q) = generate_weights_matrix(
        hidden_dim, mtx_size, sparse, dtype=dtype)
    x = generate_hidden_states(seq_len, mtx_size, 2, dtype=dtype)

    mm_ff = fp16_fp16_matmul(W, x)
    mm_ff_32 = fp32_fp32_matmul(hiW, x)
    mm_if = fp16_int4_matmul(Wq, x, scales, zeros, sparse_q)

    error["fp16,fp16"] = abs_error(mm_ff_32, mm_ff)
    error["int4,fp16"] = abs_error(mm_ff_32, mm_if)

    N_TRIALS = 5

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
        # mtx size, hidden dim, seq len, sparse
        (128, 128, 2048, False),
        (128, 128, 2048, True),
        (1024, 1024, 2048, False),
        (1024, 1024, 2048, True),
        (16384, 4096, 2048, False),
        (16384, 4096, 2048, True),
        (1024, 1024, 1, False),
        (1024, 1024, 1, True),
        (16384, 16384, 1, False),
        (16384, 16384, 1, True),
    ]

    runtime_table = []

    for trial_params in TRIALS:
        mtx_size, hidden_dim, seq_len, is_sparse = trial_params

        error, avg_runtime, runtimes = run_test_case(mtx_size, hidden_dim, seq_len, is_sparse, dtype=FLOAT_T)

        max_error = max(abs(e) for e in error.values())
        if max_error > 0.1:
            raise ValueError("Test case %r had test error %r; stopping" % (trial_params, max_error))

        for k,v in avg_runtime.items():
            l, r = k.split(",")

            runtime_table.append(
                [
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
        "S", "N", "M", "", "", "", "Time (µs)",
    ]
    for col, col_name in zip(c, col_names):
        col = [str(x) for x in col]

        max_len = max([len(x) for x in col] + [len(col_name)]) + 3

        lines[0].append(col_name.center(max_len))
        lines[1].append(("-"*(max_len-2)).center(max_len))

        for i, v in enumerate(col):
            # TODO: multiplier calculation
            lines[i+2].append(v.rjust(max_len-1) + " ")

    print("\n".join("|".join(line) for line in lines))