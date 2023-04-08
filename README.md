This repository contains CUDA kernels for [quantized] int-fp16 matrix multiplication. They were written for [reduced-kobold](https://github.com/mstnegate/reduced-kobold/), but should be usable for anything which packs data in the same way.

The kernels are lightly optimized; see microbenchmarks below. TL;DR: matmul is ~1.3x slower for dense, ~2.2x slower for sparse; matvec is ~3x faster for dense, ~2x for sparse (at least for me.)

Supported configurations:
* [quantized] int4-fp16 multiplication, with optional 16:32 structured sparsity
* [quantized] int3-fp16 and int2-fp16 multiplication; sparsity not supported (currently experimental)

Group quantization is supported for multiples of 64. All [unpacked] dimensions (except sequence length) must be a multiple of 64 currently. This was true for every model I've tested, but feel free to open an issue if you have some legitimate use case where this isn't true.

Note that due to use of tensor cores, the code requires compute capability 7.X or greater. Theoretically any card from Volta onwards *should* work, but I've only tested with a RTX 3080.

kernel_test.py is included for basic speed/accuracy benchmarking.

## Setup

You will need a build environment capable of compiling CUDA extensions for torch; see [this excellent guide](https://pytorch.org/tutorials/advanced/cpp_extension.html) for related reading.

To build, just run `python setup.py install` in whatever env you need these in; integration happens in other libs.

## Benchmarks

Below are [unscientific] microbenchmarks; proper "real-world" inferencing benchmarks coming at some point.

These benchmarks were run/generated via `kernel_test.py` on a RTX 3080 10G.

(tests were done on a (B x M x N) states tensor and (N x S) weights matrix.)

### Matrix-matrix

| B  |   S    |   N    |   M   |        |       |       | Time (µs)  |
| -- | ------ | ------ | ----- | ------ | ----- | ----- | ---------- |
|  1 |    128 |    128 |  2048 |        |  fp16 |  fp16 |      29.31 |
|  1 |    128 |    128 |  2048 |        |  int4 |  fp16 |      17.59 |
|  1 |    128 |    128 |  2048 |  16:32 |  fp16 |  fp16 |      29.32 |
|  1 |    128 |    128 |  2048 |  16:32 |  int4 |  fp16 |      17.59 |
|  1 |   1024 |   1024 |  2048 |        |  fp16 |  fp16 |      90.60 |
|  1 |   1024 |   1024 |  2048 |        |  int4 |  fp16 |     115.26 |
|  1 |   1024 |   1024 |  2048 |  16:32 |  fp16 |  fp16 |      89.14 |
|  1 |   1024 |   1024 |  2048 |  16:32 |  int4 |  fp16 |     191.49 |
|  1 |  20480 |   5120 |  2048 |        |  fp16 |  fp16 |    7720.44 |
|  1 |  20480 |   5120 |  2048 |        |  int4 |  fp16 |    9926.73 |
|  1 |  20480 |   5120 |  2048 |  16:32 |  fp16 |  fp16 |    7735.70 |
|  1 |  20480 |   5120 |  2048 |  16:32 |  int4 |  fp16 |   18078.58 |
|  1 |   5120 |  20480 |  2048 |        |  fp16 |  fp16 |    7808.76 |
|  1 |   5120 |  20480 |  2048 |        |  int4 |  fp16 |   10485.46 |
|  1 |   5120 |  20480 |  2048 |  16:32 |  fp16 |  fp16 |    7450.94 |
|  1 |   5120 |  20480 |  2048 |  16:32 |  int4 |  fp16 |   16128.34 |

### Matrix-vector

Special case where M=1 (common for inferencing if kv caching is implemented.)

| B  |   S    |   N    |   M   |        |       |       | Time (µs)  |
| -- | ------ | ------ | ----- | ------ | ----- | ----- | ---------- |
|  1 |   1024 |   1024 |     1 |        |  fp16 |  fp16 |      15.63 |
|  1 |   1024 |   1024 |     1 |        |  int4 |  fp16 |      41.04 |
|  1 |   5120 |   5120 |     1 |        |  fp16 |  fp16 |      95.76 |
|  1 |   5120 |   5120 |     1 |        |  int4 |  fp16 |      41.04 |
|  1 |  16384 |  16384 |     1 |        |  fp16 |  fp16 |     797.35 |
|  1 |  16384 |  16384 |     1 |        |  int4 |  fp16 |     273.60 |
|  1 |  20480 |   5120 |     1 |        |  fp16 |  fp16 |     302.91 |
|  1 |  20480 |   5120 |     1 |        |  int4 |  fp16 |     115.30 |
|  1 |   5120 |  20480 |     1 |        |  fp16 |  fp16 |     328.32 |
|  1 |   5120 |  20480 |     1 |        |  int4 |  fp16 |     111.39 |
|  1 |   5120 |   5120 |     1 |  16:32 |  fp16 |  fp16 |      99.67 |
|  1 |   5120 |   5120 |     1 |  16:32 |  int4 |  fp16 |      52.77 |
|  1 |   5120 |  20480 |     1 |  16:32 |  fp16 |  fp16 |     363.50 |
|  1 |   5120 |  20480 |     1 |  16:32 |  int4 |  fp16 |     158.30 |
|  1 |  16384 |  16384 |     1 |  16:32 |  fp16 |  fp16 |    1341.27 |
|  1 |  16384 |  16384 |     1 |  16:32 |  int4 |  fp16 |     594.45 |
|  1 |  20480 |   5120 |     1 |  16:32 |  fp16 |  fp16 |     457.61 |
|  1 |  20480 |   5120 |     1 |  16:32 |  int4 |  fp16 |     235.83 |
