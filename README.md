This repository contains some CUDA kernels for [quantized] int4-fp16 matrix multiplication. They were originally written for a GPTQ+SparseGPT implementation, but I never got around to finishing that.

Currently the kernels support quantized int4-fp16 multiplication with optional 16:32 structured sparsity. Group quantization is not supported currently.

Note that due to use of WMMA functions, the code requires compute capability 7.X or greater. Theoretically any card from Volta onwards *should* work, but I've only tested with a RTX 3080.

The kernels aren't very fast, but they're much faster than naive matrix multiplication. I observed around 3-4x slowdown vs. torch's direct fp16-fp16 dense matmul. There's presumably still major speed gains to be found, but it was fast enough to be usable during testing (for me, anyway.)

kernel_test.py is included for basic speed and accuracy testing. It also shows how to use the kernel functions. Note that batched matmul is currently broken for int4-fp16.
