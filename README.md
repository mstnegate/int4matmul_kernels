This repository contains CUDA kernels for [quantized] int4-fp16 matrix multiplication. They were written for [reduced-kobold](https://github.com/mstnegate/reduced-kobold/), but should be usable for anything which packs data in the same way.

The kernels are lightly optimized; see microbenchmarks below. TL;DR: slowdown is around 1.1-1.3x for dense matmul and ~2x for sparse matmul. Actual inference numbers may vary because torch is weird. 

Currently the kernels support quantized int4-fp16 multiplication with optional 16:32 structured sparsity. Group quantization is not supported currently. All dimensions (except sequence length) must be a multiple of 64 currently. This was true for every model I've tested, but feel free to open an issue if you have some legitimate use case where this isn't true.

Note that due to use of tensor cores, the code requires compute capability 7.X or greater. Theoretically any card from Volta onwards *should* work, but I've only tested with a RTX 3080.

kernel_test.py is included for basic speed/accuracy benchmarking.

## Setup

You will need a build environment capable of compiling CUDA extensions for torch; see [this excellent guide](https://pytorch.org/tutorials/advanced/cpp_extension.html) for related reading.

To build, just run `python setup.py install` in whatever env you need these in; integration happens in other libs.

## Benchmarks

Below are [unscientific] microbenchmarks; "real-world" inferencing benchmarks coming soon (whenever I get the matvec kernels finished.)

These benchmarks were run/generated via `kernel_test.py` on a RTX 3080 10G.

(tests were done on a (B x M x N) states tensor and (N x M) weights matrix.)

### Matrix-matrix

| B  |   S    |   N    |   M   |        |       |       | Time (µs)  |
| -- | ------ | ------ | ----- | ------ | ----- | ----- | ---------- |
|  1 |    128 |    128 |  2048 |        |  fp16 |  fp16 |      27.36 |
|  1 |    128 |    128 |  2048 |        |  int4 |  fp16 |      17.59 |
|  1 |    128 |    128 |  2048 |  16:32 |  fp16 |  fp16 |      29.31 |
|  1 |    128 |    128 |  2048 |  16:32 |  int4 |  fp16 |      17.59 |
|  1 |   1024 |   1024 |  2048 |        |  fp16 |  fp16 |     109.44 |
|  1 |   1024 |   1024 |  2048 |        |  int4 |  fp16 |     136.81 |
|  1 |   1024 |   1024 |  2048 |  16:32 |  fp16 |  fp16 |      89.90 |
|  1 |   1024 |   1024 |  2048 |  16:32 |  int4 |  fp16 |     171.98 |
|  1 |  20480 |   5120 |  2048 |        |  fp16 |  fp16 |    7400.88 |
|  1 |  20480 |   5120 |  2048 |        |  int4 |  fp16 |    9003.39 |
|  1 |  20480 |   5120 |  2048 |  16:32 |  fp16 |  fp16 |    7328.57 |
|  1 |  20480 |   5120 |  2048 |  16:32 |  int4 |  fp16 |   15601.06 |
|  1 |   5120 |  20480 |  2048 |        |  fp16 |  fp16 |    7154.65 |
|  1 |   5120 |  20480 |  2048 |        |  int4 |  fp16 |    9038.57 |
|  1 |   5120 |  20480 |  2048 |  16:32 |  fp16 |  fp16 |    7082.33 |
|  1 |   5120 |  20480 |  2048 |  16:32 |  int4 |  fp16 |   15741.77 |


### Matrix-vector

Special case where M=1. Note that there currently isn't any matrix-vector kernel specialization (coming soon!) so this scenario is extremely computationally wasteful.

| B  |   S    |   N    |   M   |        |       |       | Time (µs)  |
| -- | ------ | ------ | ----- | ------ | ----- | ----- | ---------- |
|  1 |   1024 |   1024 |     1 |        |  fp16 |  fp16 |      15.64 |
|  1 |   1024 |   1024 |     1 |        |  int4 |  fp16 |      31.27 |
|  1 |   5120 |   5120 |     1 |        |  fp16 |  fp16 |      97.71 |
|  1 |   5120 |   5120 |     1 |        |  int4 |  fp16 |     162.21 |
|  1 |  16384 |  16384 |     1 |        |  fp16 |  fp16 |     799.30 |
|  1 |  16384 |  16384 |     1 |        |  int4 |  fp16 |     759.28 |
|  1 |  20480 |   5120 |     1 |        |  fp16 |  fp16 |     308.77 |
|  1 |  20480 |   5120 |     1 |        |  int4 |  fp16 |     598.01 |
|  1 |   5120 |  20480 |     1 |        |  fp16 |  fp16 |     343.95 |
|  1 |   5120 |  20480 |     1 |        |  int4 |  fp16 |     295.10 |
|  1 |   5120 |   5120 |     1 |  16:32 |  fp16 |  fp16 |     101.62 |
|  1 |   5120 |   5120 |     1 |  16:32 |  int4 |  fp16 |     234.51 |
|  1 |   5120 |  20480 |     1 |  16:32 |  fp16 |  fp16 |     365.45 |
|  1 |   5120 |  20480 |     1 |  16:32 |  int4 |  fp16 |     504.20 |
|  1 |  16384 |  16384 |     1 |  16:32 |  fp16 |  fp16 |     879.43 |
|  1 |  16384 |  16384 |     1 |  16:32 |  int4 |  fp16 |    1330.87 |
|  1 |  20480 |   5120 |     1 |  16:32 |  fp16 |  fp16 |     347.86 |
|  1 |  20480 |   5120 |     1 |  16:32 |  int4 |  fp16 |     889.20 |
