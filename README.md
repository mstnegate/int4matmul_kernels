This repository contains some CUDA kernels for [quantized] int4-fp16 matrix multiplication. They were originally written for a GPTQ+SparseGPT implementation, but I never got around to finishing that.

Currently the kernels support quantized int4-fp16 multiplication with optional 16:32 structured sparsity. Group quantization is not supported currently.

Note that due to use of WMMA functions, the code requires compute capability 7.X or greater. Theoretically any card from Volta onwards *should* work, but I've only tested with a RTX 3080.

The kernels aren't very fast, but they're much faster than naive matrix multiplication. I observed around 3-4x slowdown vs. torch's direct fp16-fp16 dense matmul. There's presumably still major speed gains to be found, but it was fast enough to be usable during testing (for me, anyway.)

kernel_test.py is included for basic speed and accuracy testing. It also shows how to use the kernel functions.

## Benchmarks

Below are some unscientific generation speed benchmarks on "real-world" inferencing. All models were run with a RTX 3080 10G, in KoboldAI. INT4 and INT4+16:32 models were quantized/sparsified via [reduced-kobold](https://github.com/mstnegate/reduced-kobold/)

All tests are the median of 7 runs, using identical sampler settings and generating 80 tokens with max context (2048, or I guess more technically 1968 tokens.) Execution time was measured from the front-end.

| Bits | Sparsity |  125M |  2.7B |   13B  |
|------| -------- | ----- | ----- | ------ |
|  16  |   100%   | 2.30s | 3.97s |   OOM  |
|   4  |   100%   | 2.48s | 4.84s |   OOM  |
|   4  |  16:32   | 2.41s | 5.14s | 17.05s |

Below is the same data, but converted to tokens per second for convenience:

| Bits | Sparsity |   125M  |   2.7B  |   13B   |
|------| -------- | ------- | ------- | ------- |
|  16  |   100%   | 34.8t/s | 20.2t/s |   OOM   |
|   4  |   100%   | 32.3t/s | 16.5t/s |   OOM   |
|   4  |  16:32   | 33.2t/s | 15.6t/s | 4.7 t/s |
