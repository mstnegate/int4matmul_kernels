#include <torch/all.h>
#include <torch/python.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

////////////////////////////////////////////////////////////////////////////////

#define WARP_SIZE 32
#define FULL_MASK 0xFFFFFFFF

////////////////////////////////////////////////////////////////////////////////

// TODO: random sparse mask function for testing (later)

__global__ void PackMatrix(
    uint32_t* __restrict__ wt_outs,
    uint32_t* __restrict__ msk_outs,

    const uint8_t* __restrict__ wt_in,
    const uint8_t* __restrict__ msk_in,

    const size_t height,
    const size_t stride
) {
    const size_t rx = threadIdx.x + blockDim.x * blockIdx.x;
    const size_t ry = threadIdx.y + blockDim.y * blockIdx.y;
    const size_t thrd_idx = (threadIdx.x + threadIdx.y * blockDim.x) % 32;

    if ((ry < height) && (rx < stride)) {
        const int is_active = msk_in[rx + ry*stride] > 0;

        // construct packed sparsity mask via ballot and write it
        // FULL_MASK is safe since mtx height is mult of 32 (warp size)
        const uint32_t bmsk = __ballot_sync(FULL_MASK, is_active);
        msk_outs[ry + (rx/32)*height] = bmsk;

        assert(__popc(bmsk) == 16);

        // next combine items; figure out position first before packing
        const uint32_t pos = __popc(bmsk & ((1 << thrd_idx) - 1));

        // 16 weights, 4 bits each; everything fits fine into a uint64_t
        uint64_t v = wt_in[rx + ry*stride];
        // preprocess and prepare
        v &= 0xF;
        v <<= 4 * pos;
        v *= is_active;

        for (int offset = 16; offset > 0; offset /= 2) {
            v += __shfl_down_sync(FULL_MASK, v, offset);
        }

        // unpack 64 bits; all writes done by one thread since we can't
        // guarantee weight positions if we wanted to pack to 2x32
        if (thrd_idx == 0) {
            wt_outs[ry + rx/16*height] = (uint32_t)(v & 0xFFFFFFFF);
            wt_outs[ry + (rx/16+1)*height] = (uint32_t)((v >> 32) & 0xFFFFFFFF);
        }
    }
}


void sparse_int4_pack(
    torch::Tensor wt_outs,
    torch::Tensor msk_outs,
    torch::Tensor wt_in,
    torch::Tensor msk_in
) {
    // usual health and safety checks
    assert(wt_in.size(1) == (wt_outs.size(0)*8*2));
    assert(msk_in.size(1) == (msk_outs.size(0)*32));
    assert(wt_in.size(0) == wt_outs.size(1));
    assert(msk_in.size(0) == msk_outs.size(1));

    assert(wt_outs.dtype() == at::kInt);
    assert(msk_outs.dtype() == at::kInt);
    assert(wt_in.dtype() == at::kByte);
    assert(msk_in.dtype() == at::kByte);

    // idea: take these:
    //  wt_in: NxM matrix of 4-bit int weights, stored in uint8s
    //  msk_in: NxM matrix of 1-bit sparseness bools, stored in uint8s
    // then pack them into:
    //  wt_outs: (M // 16)xN matrix of 8 4-bit int weights packed into uint32
    //           [technically int32s since torch doesn't support uint32]
    //  msk_outs: (M // 32)xN matrix of 32 1-bit sparse bools, packed into
    //            uint32 [same caveat as before]
    // we intentionally take in un-transposed matrices since:
    //   1. it's rawer and less processed
    //   2. better memory access pattern

    // coordinates based on un-transposed data
    const size_t height = wt_in.size(0);
    const size_t stride = wt_in.size(1);

    //assert(height % 32 == 0);
    assert(stride % 32 == 0);

    const auto THREAD_X = WARP_SIZE;
    const auto THREAD_Y = 1024 / WARP_SIZE;
    dim3 threads(THREAD_X, THREAD_Y);
    dim3 blocks(
        (stride + THREAD_X - 1) / THREAD_X,
        (height + THREAD_Y - 1) / THREAD_Y
    );

    PackMatrix<<<blocks, threads>>>(
        reinterpret_cast<uint32_t*>(wt_outs.data<int32_t>()),
        reinterpret_cast<uint32_t*>(msk_outs.data<int32_t>()),
        wt_in.data<uint8_t>(),
        msk_in.data<uint8_t>(),
        height,
        stride
    );
}