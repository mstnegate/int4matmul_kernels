#include <iostream>
#include <cstdint>
#include "matmul.cuh"

////////////////////////////////////////////////////////////////////////////////
// actual kernel stuff

template <bool is_sparse>
__device__ void MVA_Int4_Half( BOILERPLATE_ACCUM_ARGS(__half, float), SPARSE_MASK_ARGS );

template <typename scalar_t>
__global__ void MVV_Int4_Dense( RECEIVE_BOILERPLATE_ACCUM_ARGS(scalar_t, float) );

template <typename scalar_t>
__global__ void MVV_Int4_Sparse( RECEIVE_BOILERPLATE_ACCUM_ARGS(scalar_t, float), RECEIVE_SPARSE_MASK_ARGS );

////////////////////////////////////////////////////////////////////////////////

// invalid specializations
template <typename scalar_t>
__global__ void MVV_Int4_Dense(
    RECEIVE_BOILERPLATE_ACCUM_ARGS(scalar_t, float)
) { assert(0); }

template <typename scalar_t>
__global__ void MVV_Int4_Sparse(
    RECEIVE_BOILERPLATE_ACCUM_ARGS(scalar_t, float),
    RECEIVE_SPARSE_MASK_ARGS
) { assert(0); }

////////////////////////////////////////////////////////////////////////////////

template <bool is_sparse>
__device__ float smem_dot(
    const __half mult, uint32_t* __restrict__ packed_wts,
    const size_t i_offset, const size_t j_offset,
    const __half scale, const __half zero
);

template <>
__device__ float smem_dot<false>(
    const __half mult, uint32_t* __restrict__ packed_wts,
    const size_t i_offset, const size_t j_offset,
    const __half scale, const __half zero
) {
    float m = 0.0f;

    uint32_t* read_hd = packed_wts + j_offset + i_offset*4*BLOCK_SIZE;


    #pragma unroll
    for(int w = 0; w < 32; w += 8) {
        uint32_t v = *read_hd;
        #pragma unroll
        for(int i = 0; i < 8; i += 4) {
            // didn't see much gain by trying to pack 4x16 into a shuffle
            __half a = __shfl_sync(FULL_MASK, mult, w+i+0);
            __half b = __shfl_sync(FULL_MASK, mult, w+i+1);
            __half c = __shfl_sync(FULL_MASK, mult, w+i+2);
            __half d = __shfl_sync(FULL_MASK, mult, w+i+3);

            #define ACC_OP(X) m += __half2float(X)*__half2float(dequantize(v, scale, zero)); v >>= 4;
            ACC_OP(a)
            ACC_OP(b)
            ACC_OP(c)
            ACC_OP(d)
            #undef ACC_OP
        }
        read_hd += BLOCK_SIZE;
    }
    return m;
}

template <>
__device__ float smem_dot<true>(
    const __half mult, uint32_t* __restrict__ packed_wts,
    const size_t i_offset, const size_t j_offset,
    const __half scale, const __half zero
) {
    const size_t MASK_ROWS = (BLOCK_SIZE / 8 / 4);

    float m = 0.0f;

    uint32_t* read_hd = packed_wts + j_offset + i_offset*2*BLOCK_SIZE;
    uint32_t msk = *(packed_wts + BLOCK_SIZE*((BLOCK_SIZE/8 - MASK_ROWS) + i_offset) + j_offset);

    uint64_t v = 0U;
    v += *read_hd;
    v += ((uint64_t)*(read_hd + BLOCK_SIZE)) << 32;

    #pragma unroll
    for(int i = 0; i < 32; i += 4) {
        __half a = __shfl_sync(FULL_MASK, mult, i+0);
        __half b = __shfl_sync(FULL_MASK, mult, i+1);
        __half c = __shfl_sync(FULL_MASK, mult, i+2);
        __half d = __shfl_sync(FULL_MASK, mult, i+3);

        // heavy branching's fine since it's all in register
        #define ACC_OP(X) if (msk & 1) { m += __half2float(X)*__half2float(dequantize((uint32_t)(v & 0xF), scale, zero)); v >>= 4; } msk >>= 1;
        ACC_OP(a)
        ACC_OP(b)
        ACC_OP(c)
        ACC_OP(d)
        #undef ACC_OP

        __syncwarp();
    }

    return m;
}

////////////////////////////////////////////////////////////////////////////////

#define BLOCK_WIDTH_SPAN 4

// generic half-precision matrix-vector mult
template <bool is_sparse>
__device__ void MVA_Int4_Half(
    BOILERPLATE_ACCUM_ARGS(__half, float),
    SPARSE_MASK_ARGS
) {
    multiplier += blockIdx.z*seq_len*in_size;
    outs += blockIdx.z*seq_len*out_size;

    // conceptually simple; load in blocks of 64x64 and proceed; only thing to
    // note is that we sweep across instead of down to minimize loads of
    // multiplier, and that we fused the unpacking and math ops
    __shared__ uint32_t packed_wts[(BLOCK_SIZE / 8)*BLOCK_SIZE];

    const size_t warpIdx = threadIdx.y;
    // TODO: properly index these
    const size_t warpSubIdx = warpIdx % 2;
    const size_t warpSuperIdx = warpIdx / 2;
    const size_t warpOffset = warpSubIdx * WARP_SIZE + threadIdx.x;

    const __half ZERO = __float2half(0.0f);

    // first GMEM load: loading the multiplier (a single vector);
    // there's redundancy involved since we have a 64x64 block and 128 threads
    __half m;

    const size_t downPos = blockIdx.y*BLOCK_SIZE;
    const size_t offset = (downPos + warpSuperIdx*WARP_SIZE + threadIdx.x);
    if (offset < in_size) {
        m = multiplier[offset];
    } else {
        m = ZERO;
    }
    __syncthreads();

    size_t mtx_j = blockIdx.x*BLOCK_SIZE*BLOCK_WIDTH_SPAN;
    #pragma unroll
    for(size_t j = 0; j < BLOCK_WIDTH_SPAN; ++j) {
        if (mtx_j >= out_size) { break; }

        __syncthreads();

        __half scale = scales[mtx_j + warpOffset];
        __half zero = zeros[mtx_j + warpOffset];
        #ifdef FMA_TRANSFORM
            zero = __hneg(__hmul(zero, scale))
        #endif

        // GMEM loading chunk
        gmem_load_weights<is_sparse, BLOCK_SIZE>::load(
            matrix, packed_wts, sparse_mask,
            warpSuperIdx, warpOffset,
            downPos/8, mtx_j,
            out_size);

        __syncthreads();

        // fused unpack and add
        float acc = smem_dot<is_sparse>(m, packed_wts, warpSuperIdx, warpOffset, scale, zero);

        __syncthreads();

        // accumulate our results
        atomicAdd(&outs[mtx_j + warpOffset], acc);

        mtx_j += BLOCK_SIZE;
    }
}


////////////////////////////////////////////////////////////////////////////////

template <>
__global__ void MVV_Int4_Dense<float>(
    RECEIVE_BOILERPLATE_ARGS(float)
) {
    assert(0);
}
template <>
__global__ void MVV_Int4_Sparse<float>(
    RECEIVE_BOILERPLATE_ARGS(float),
    RECEIVE_SPARSE_MASK_ARGS
) {
    assert(0);
}

template <>
__global__ void MVV_Int4_Dense<c10::Half>(
    RECEIVE_BOILERPLATE_ACCUM_ARGS(c10::Half, float)
) {
    MVA_Int4_Half<false>(
        // torch stuff; torch doesn't store halfs internally as CUDA halfs,
        // but they *are* bit-compatible so reinterpret_cast solves the issue
        outs,
        reinterpret_cast<const uint32_t*>(matrix),
        reinterpret_cast<const __half*>(multiplier),
        reinterpret_cast<const __half*>(scales),
        reinterpret_cast<const __half*>(zeros),
        in_size, seq_len, mtx_in_size, out_size, nullptr
    );
}
template <>
__global__ void MVV_Int4_Sparse<c10::Half>(
    RECEIVE_BOILERPLATE_ACCUM_ARGS(c10::Half, float),
    RECEIVE_SPARSE_MASK_ARGS
) {
    MVA_Int4_Half<true>(
        outs,
        reinterpret_cast<const uint32_t*>(matrix),
        reinterpret_cast<const __half*>(multiplier),
        reinterpret_cast<const __half*>(scales),
        reinterpret_cast<const __half*>(zeros),
        in_size, seq_len, mtx_in_size, out_size,
        reinterpret_cast<const uint32_t*>(sparse_mask)
    );
}


void matvec_int4(
    torch::Tensor outs,
    torch::Tensor matrix,
    torch::Tensor x,
    torch::Tensor scales,
    torch::Tensor zeros,
    c10::optional<torch::Tensor> sparse_mask
) {
    // TODO: proper dimensioning later
    const bool is_sparse = sparse_mask.has_value() && sparse_mask.value().defined();

    // special case of matrix multiplication: with seq-len 1 tensors; see the
    // matmul kernel for explanations on dimensions and such

    const auto batch_size = x.size(0);
    const auto seq_len = x.size(1);
    const auto in_size = x.size(2);

    const auto mtx_in_size = matrix.size(0);
    const auto out_size = matrix.size(1);

    assert(seq_len == 1);

    if (is_sparse) {
        assert((mtx_in_size*16) == in_size);
    } else {
        assert((mtx_in_size*8) == in_size);
    }
    assert(outs.size(0) == batch_size);
    assert(outs.size(1) == seq_len);
    assert(outs.size(2) == out_size);
    assert(zeros.size(0) == out_size);
    assert(scales.size(0) == out_size);

    const auto THREAD_X = WARP_SIZE;
    const auto THREAD_Y = WMMA_CHUNK_COUNT;

    dim3 threads(THREAD_X, THREAD_Y);
    dim3 blocks(
        (out_size + (BLOCK_SIZE*BLOCK_WIDTH_SPAN) - 1) / (BLOCK_SIZE*BLOCK_WIDTH_SPAN),
        (in_size + BLOCK_SIZE - 1) / BLOCK_SIZE,
        batch_size
    );

    // allocate the float accumulator; this technically creates a tiny bit of
    // memory overhead; should fix that at some point
    auto temp_outs = outs.to(torch::kFloat32);

    if (is_sparse) {
        auto actual_sparse = sparse_mask.value();
        assert(actual_sparse.size(0)*2 == mtx_in_size);
        assert(actual_sparse.size(1) == out_size);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.type(), "mmv_int4", ([&] {
            MVV_Int4_Sparse<<<blocks, threads>>>(
                temp_outs.data<float>(),
                matrix.data<int32_t>(),
                x.data<scalar_t>(),
                scales.data<scalar_t>(),
                zeros.data<scalar_t>(),
                // multiply mtx_in_size by 2 to pretend it's still in terms of
                // weights; easier indexing
                in_size, seq_len, mtx_in_size*2, out_size, actual_sparse.data<int32_t>()
            );
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.type(), "mmv_int4", ([&] {
            MVV_Int4_Dense<<<blocks, threads>>>(
                temp_outs.data<float>(),
                matrix.data<int32_t>(),
                x.data<scalar_t>(),
                scales.data<scalar_t>(),
                zeros.data<scalar_t>(),
                in_size, seq_len, mtx_in_size, out_size
            );
        }));
    }
    temp_outs = temp_outs.to(torch::kHalf);
    outs.index_put_({"..."}, temp_outs);

}
