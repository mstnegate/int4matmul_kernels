This repository contains CUDA kernels for [quantized] int4-fp16 matrix multiplication. They were written for [reduced-kobold](https://github.com/mstnegate/reduced-kobold/), but should be usable for anything which packs data in the same way.

The kernels are lightly optimized; see microbenchmarks below. TL;DR: matmul is ~1.2x slower for dense, ~2x slower for sparse; matvec is ~3x faster for dense, ~2x for sparse (at least for me.)

Currently the kernels support quantized int4-fp16 multiplication with optional 16:32 structured sparsity. Group quantization is not supported currently. All dimensions (except sequence length) must be a multiple of 64 currently. This was true for every model I've tested, but feel free to open an issue if you have some legitimate use case where this isn't true.

Note that due to use of tensor cores, the code requires compute capability 7.X or greater. Theoretically any card from Volta onwards *should* work, but I've only tested with a RTX 3080.

kernel_test.py is included for basic speed/accuracy benchmarking.

## Setup

You will need a build environment capable of compiling CUDA extensions for torch; see [this excellent guide](https://pytorch.org/tutorials/advanced/cpp_extension.html) for related reading.

To build, just run `python setup.py install` in whatever env you need these in; integration happens in other libs.

## Benchmarks

Below are [unscientific] microbenchmarks; proper "real-world" inferencing benchmarks coming at some point.

These benchmarks were run/generated via `kernel_test.py` on a RTX 3080 10G.

(tests were done on a (B x S x N) states tensor and (N x M) weights matrix.)

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

Special case where M=1 (common for inferencing if kv caching is implemented.)

| B  |   S    |   N    | M  |        |       |       | Time (µs)  |
| -- | ------ | ------ | -- | ------ | ----- | ----- | ---------- |
|  1 |   1024 |   1024 |  1 |        |  fp16 |  fp16 |      15.64 |
|  1 |   1024 |   1024 |  1 |        |  int4 |  fp16 |      41.23 |
|  1 |   5120 |   5120 |  1 |        |  fp16 |  fp16 |      96.95 |
|  1 |   5120 |   5120 |  1 |        |  int4 |  fp16 |      42.09 |
|  1 |  16384 |  16384 |  1 |        |  fp16 |  fp16 |     802.68 |
|  1 |  16384 |  16384 |  1 |        |  int4 |  fp16 |     257.95 |
|  1 |  20480 |   5120 |  1 |        |  fp16 |  fp16 |     303.23 |
|  1 |  20480 |   5120 |  1 |        |  int4 |  fp16 |     105.53 |
|  1 |   5120 |  20480 |  1 |        |  fp16 |  fp16 |     327.31 |
|  1 |   5120 |  20480 |  1 |        |  int4 |  fp16 |     105.86 |
|  1 |   5120 |   5120 |  1 |  16:32 |  fp16 |  fp16 |      98.74 |
|  1 |   5120 |   5120 |  1 |  16:32 |  int4 |  fp16 |      50.81 |
|  1 |   5120 |  20480 |  1 |  16:32 |  fp16 |  fp16 |     355.83 |
|  1 |   5120 |  20480 |  1 |  16:32 |  int4 |  fp16 |     142.83 |
|  1 |  16384 |  16384 |  1 |  16:32 |  fp16 |  fp16 |    1004.64 |
|  1 |  16384 |  16384 |  1 |  16:32 |  int4 |  fp16 |     345.91 |
|  1 |  20480 |   5120 |  1 |  16:32 |  fp16 |  fp16 |     332.24 |
|  1 |  20480 |   5120 |  1 |  16:32 |  int4 |  fp16 |     144.62 |
