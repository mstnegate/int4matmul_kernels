import torch
import int4matmul
import numpy as np
import time

torch.manual_seed(123)

# SIZE = 32
# HIDE_DIM = 32
# SEQ_LEN = 32

# SIZE = 2048
# HIDE_DIM = 8192
# SEQ_LEN = 2048

SIZE = 4096
HIDE_DIM = 16384
SEQ_LEN = 2048

FLOAT_T = torch.float16
SPARSE_BLOCK_SIZE = 32
SPARSIFY = False

################################################################################

zeros = ((torch.rand((HIDE_DIM,), dtype=FLOAT_T, device="cuda:0") * 2.0) - 1)
scales = ((torch.rand((HIDE_DIM,), dtype=FLOAT_T, device="cuda:0") * 2.0) - 1)
x = ((torch.rand((1, SEQ_LEN, SIZE), dtype=FLOAT_T, device="cuda:0") * 2.0) - 1)
b_Wq = torch.randint(0, 16, (HIDE_DIM, SIZE), dtype=torch.int, device="cuda:0")

if SPARSIFY:
    sparse = torch.ones((HIDE_DIM, SIZE), dtype=FLOAT_T, device="cuda:0")

    # TODO: test cases where dim isn't multiple of sparse block
    assert SIZE % SPARSE_BLOCK_SIZE == 0

    itms = np.arange(SPARSE_BLOCK_SIZE).astype(int)

    # pregenerate "random" sparse masks; we don't need actual random masks
    # to test the kernel anyway
    RANDOM_OPTIONS = 1024
    N_CHOICES = [
        np.random.choice(itms, size=(SPARSE_BLOCK_SIZE//2,), replace=False)
        for _ in range(RANDOM_OPTIONS)
    ]

    for i in range(HIDE_DIM):
        rstr = np.random.randint(0, RANDOM_OPTIONS, size=SIZE//SPARSE_BLOCK_SIZE)
        for j in range(0, SIZE, SPARSE_BLOCK_SIZE):
            choices = N_CHOICES[rstr[j // SPARSE_BLOCK_SIZE]]
            sparse[i, j+choices] = 0.0

W = ((b_Wq * scales[:, None]) + zeros[:, None]).T
hiW = ((b_Wq.to(torch.float32) * scales[:, None].to(torch.float32)) + zeros[:, None].to(torch.float32)).T

def fast_pack():
    f_Wq = torch.empty((SIZE//16, HIDE_DIM), dtype=torch.int, device="cuda:0")
    f_sparse_q = torch.empty((SIZE//32, HIDE_DIM), dtype=torch.int, device="cuda:0")
    int4matmul.weight_matrix_packing(
        f_Wq,
        f_sparse_q,
        b_Wq.to(torch.uint8).contiguous(),
        sparse.to(torch.uint8)
    )
    return f_Wq, f_sparse_q

if SPARSIFY:
    Wq, sparse_q = fast_pack()

    sparse = sparse.T.to(torch.int32)
    W *= torch.tensor(sparse).to(FLOAT_T)
    hiW *= torch.tensor(sparse).to(torch.float32)
else:
    b_Wq = b_Wq.T

    Wq = torch.empty((SIZE//8, HIDE_DIM), dtype=torch.int, device="cuda:0")

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

################################################################################

def fp16_fp16_matmul(W, x):
    return x @ W

def fp32_fp32_matmul(W, x):
    return x.to(torch.float32) @ hiW

def fp16_int4_matmul(Wq, x, scales, zeros):
    outs = torch.zeros((x.shape[0], SEQ_LEN, HIDE_DIM), dtype=FLOAT_T, device="cuda:0")
    int4matmul.quant_int4_linear_mult(
        outs, Wq, x, scales, zeros,
        None if not SPARSIFY else sparse_q
    )
    return outs

def abs_error(truth, candidate):
    return ((candidate - truth) / truth).abs().nanmean()

mm_ff = fp16_fp16_matmul(W, x)
mm_ff_32 = fp32_fp32_matmul(W, x)
mm_if = fp16_int4_matmul(Wq, x, scales, zeros)

print("fp16-fp16 pct err: %.3f%%" % (abs_error(mm_ff_32, mm_ff) * 100))
print("fp16-int4 pct err: %.3f%%" % (abs_error(mm_ff_32, mm_if) * 100))

################################################################################

N_TRIALS = 5

for name, func in [
    ("fp16-fp16 mult", lambda: fp16_fp16_matmul(W, x)),
    ("fp16-int4 mult", lambda: fp16_int4_matmul(Wq, x, scales, zeros)),
]:
    print(name)
    times = []
    for _ in range(N_TRIALS):
        start = time.time()
        for _ in range(100):
            func()
        torch.cuda.synchronize("cuda:0")
        end = time.time()
        print(end-start)
        times.append(end-start)
    print(times)
    print(sum(times) / len(times))

print("Done!")