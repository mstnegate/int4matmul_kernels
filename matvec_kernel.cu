#include <iostream>
#include <cstdint>
#include "matmul.cuh"

////////////////////////////////////////////////////////////////////////////////

#define SMEM_DOT_ARGS \
    const __half mult, uint32_t* __restrict__ packed_wts, \
    const size_t i_offset, const size_t j_offset, \
    const __half scale, const __half zero

template <bool is_sparse, uint32_t BITS, Quantization qm>
struct smem_dot_inner {
    __device__ inline static float calc(SMEM_DOT_ARGS);
};

template <uint32_t BITS, Quantization qm>
struct smem_dot_inner<false, BITS, qm> {
    __device__ inline static float calc(SMEM_DOT_ARGS) {
        static_assert(is_power_of_two(BITS), "Generic weight unpacking only supports powers of 2.");

        float m = 0.0f;

        constexpr size_t OUTER_STEP = 32 / BITS;

        // N-bit packing => N 32-bit ints needed to hold 32 N-bit weights
        uint32_t* read_hd = packed_wts + j_offset + i_offset*BITS*BLOCK_SIZE;

        #pragma unroll
        for(int w = 0; w < 32; w += OUTER_STEP) {
            uint32_t v = *read_hd;
            #pragma unroll
            for(int i = 0; i < OUTER_STEP; i += 4) {
                // didn't see much gain by packing 4x16 into a shuffle
                __half a = __shfl_sync(FULL_MASK, mult, w+i+0);
                __half b = __shfl_sync(FULL_MASK, mult, w+i+1);
                __half c = __shfl_sync(FULL_MASK, mult, w+i+2);
                __half d = __shfl_sync(FULL_MASK, mult, w+i+3);

                #define ACC_OP(X) m += __half2float(X)*__half2float(dequantize<qm, BITS>::call(v, scale, zero)); v >>= BITS;
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
};

template <Quantization qm>
struct smem_dot_inner<false, 3, qm> {
    __device__ inline static float calc(SMEM_DOT_ARGS) {
        if constexpr(BLOCK_SIZE != 64U) {
            assert(0); // hardcoded assumption
        }

        float m = 0.0f;

        uint32_t* read_hd = packed_wts + j_offset + i_offset*3*BLOCK_SIZE;

        #define ACC_OP(X) m += __half2float(X)*__half2float(dequantize<qm, 3>::call(v, scale, zero)); v >>= 3;
        #define FETCH_SEQ(j) \
            __half a = __shfl_sync(FULL_MASK, mult, j+0); \
            __half b = __shfl_sync(FULL_MASK, mult, j+1); \
            __half c = __shfl_sync(FULL_MASK, mult, j+2); \
            __half d = __shfl_sync(FULL_MASK, mult, j+3);

        // manually unrolled kernels to deal with the edge values; not pretty
        {
            uint32_t v = *read_hd; read_hd += BLOCK_SIZE;
            uint32_t v2 = *read_hd; read_hd += BLOCK_SIZE;
            #pragma unroll
            for(int i = 0; i < 8; i += 4) {
                FETCH_SEQ(i);
                ACC_OP(a)
                ACC_OP(b)
                ACC_OP(c)
                ACC_OP(d)
            }
            {
                FETCH_SEQ(8);
                ACC_OP(a)
                ACC_OP(b)
                // weight 11: 2 bits from packed_wt #1, 1 bit from packed_wt #2
                v |= ((v2 & 0x1) << 2);
                ACC_OP(c)
                // we're now in packed 2
                v = v2 >> 1;
                v2 = *read_hd;
                ACC_OP(d)
            }
            #pragma unroll
            for(int i = 12; i < 20; i += 4) {
                FETCH_SEQ(i);
                ACC_OP(a)
                ACC_OP(b)
                ACC_OP(c)
                ACC_OP(d)
            }
            {
                FETCH_SEQ(20);
                ACC_OP(a)
                // weight 21: 1 bit from packed_wt #2, 2 bits from packed_wt #3
                v |= ((v2 & 0x3) << 1);
                ACC_OP(b)
                v = v2 >> 2;
                ACC_OP(c)
                ACC_OP(d)
            }
            #pragma unroll
            for(int i = 24; i < 32; i += 4) {
                FETCH_SEQ(i);
                ACC_OP(a)
                ACC_OP(b)
                ACC_OP(c)
                ACC_OP(d)
            }
        }
        #undef FETCH_SEQ
        #undef ACC_OP

        return m;
    }
};

template <uint32_t BITS, Quantization qm>
struct smem_dot_inner<true, BITS, qm> {
    // general case not supported
    __device__ inline static float calc(SMEM_DOT_ARGS) { assert(0); }
};

template <Quantization qm>
struct smem_dot_inner<true, 4, qm> {
    __device__ inline static float calc(SMEM_DOT_ARGS) {
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
            #define ACC_OP(X) if (msk & 1) { m += __half2float(X)*__half2float(dequantize<qm, 4>::call((uint32_t)(v & 0xF), scale, zero)); v >>= 4; } msk >>= 1;
            ACC_OP(a)
            ACC_OP(b)
            ACC_OP(c)
            ACC_OP(d)
            #undef ACC_OP

            __syncwarp();
        }

        return m;
    }
};


template <bool is_sparse, uint32_t BITS, Quantization qm>
struct smem_dot {
    __device__ inline static float calc(SMEM_DOT_ARGS) {
        return smem_dot_inner<is_sparse, BITS, qm>::calc(mult, packed_wts, i_offset, j_offset, scale, zero);
    }
};
template <bool is_sparse, uint32_t BITS>
struct smem_dot<is_sparse, BITS, Quantization::DYNAMIC_EXPONENT_SYM> {
    __device__ inline static float calc(SMEM_DOT_ARGS) {
        // WARNING: this assumes that the entire warp branches the same way!
        if (__hge(scale, __float2half(0.0f))) {
            return smem_dot_inner<is_sparse, BITS, Quantization::LINEAR>::calc(
                mult, packed_wts, i_offset, j_offset, scale, zero);
        } else {
            return smem_dot_inner<is_sparse, BITS, Quantization::EXPONENT_SYM>::calc(
                mult, packed_wts, i_offset, j_offset, __hneg(scale), zero);
        }
    }
};
#undef SMEM_DOT_ARGS

////////////////////////////////////////////////////////////////////////////////

#define BLOCK_WIDTH_SPAN 4

// generic half-precision matrix-vector mult
template <bool is_sparse, uint32_t BITS>
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
    size_t gq_offset = (downPos/group_size)*out_size;
    #pragma unroll
    for(size_t j = 0; j < BLOCK_WIDTH_SPAN; ++j) {
        if (mtx_j >= out_size) { break; }

        __syncthreads();

        __half scale = scales[mtx_j + warpOffset + gq_offset];
        __half zero = zeros[mtx_j + warpOffset + gq_offset];
        #ifdef FMA_TRANSFORM
            zero = __hneg(__hmul(zero, scale))
        #endif

        // GMEM loading chunk
        gmem_load_weights<is_sparse, BITS, BLOCK_SIZE>::load(
            matrix, packed_wts, sparse_mask,
            warpSuperIdx, warpOffset,
            downPos*BITS/32, mtx_j,
            out_size);

        __syncthreads();

        // fused unpack and add
        float acc = smem_dot<is_sparse, BITS, Quantization::DYNAMIC_EXPONENT_SYM>::calc(
            m, packed_wts, warpSuperIdx, warpOffset, scale, zero);

        __syncthreads();

        // accumulate our results
        atomicAdd(&outs[mtx_j + warpOffset], acc);

        mtx_j += BLOCK_SIZE;
    }
}


////////////////////////////////////////////////////////////////////////////////

// see matmul_kernel.cu for an explanation of why we're doing this
#define ARG_LST \
    outs, \
    reinterpret_cast<const uint32_t*>(matrix), \
    reinterpret_cast<const __half*>(multiplier), \
    reinterpret_cast<const __half*>(scales), \
    reinterpret_cast<const __half*>(zeros), \
    group_size, in_size, seq_len, mtx_in_size, out_size

#define MACRO_OP(IS_SPARSE, BITS) \
    template <typename scalar_t> \
    __global__ void matvec_intermediate_ ## IS_SPARSE ## _ ## BITS( \
        RECEIVE_BOILERPLATE_ACCUM_ARGS(scalar_t, float), RECEIVE_SPARSE_MASK_ARGS \
    ) { assert(0); } \
    template <> \
    __global__ void matvec_intermediate_ ## IS_SPARSE ## _ ## BITS<c10::Half>( \
        RECEIVE_BOILERPLATE_ACCUM_ARGS(c10::Half, float), RECEIVE_SPARSE_MASK_ARGS \
    ) { \
        if (IS_SPARSE) { \
            MVA_Int4_Half<true, BITS>(ARG_LST, reinterpret_cast<const uint32_t*>(sparse_mask)); \
        } else { \
            MVA_Int4_Half<false, BITS>(ARG_LST, nullptr); \
        } \
    }

#include "kernel_specs.cuh"

#undef ARG_LST
#undef MACRO_OP

template <uint32_t bits>
void matvec_packed_int(TENSOR_MULT_ARGS) {
    // TODO: proper dimensioning later
    const bool is_sparse = sparse_mask.has_value() && sparse_mask.value().defined();

    if (group_size < 0) {
        group_size = 0x0FFFFFFF;
    } else {
        assert(group_size % 64 == 0);
    }

    // special case of matrix multiplication: with seq-len 1 tensors; see the
    // matmul kernel for explanations on dimensions and such

    const auto batch_size = multiplier.size(0);
    const auto seq_len = multiplier.size(1);
    const auto in_size = multiplier.size(2);

    const auto mtx_in_size = matrix.size(0);
    const auto out_size = matrix.size(1);

    assert(seq_len == 1);

    if (is_sparse) {
        assert((mtx_in_size * 32 * 2 / bits) == in_size);
    } else {
        assert((mtx_in_size * 32 / bits) == in_size);
    }

    assert(outs.size(0) == batch_size);
    assert(outs.size(1) == seq_len);
    assert(outs.size(2) == out_size);
    assert(zeros.size(1) == out_size);
    assert(scales.size(1) == out_size);
    assert((group_size * zeros.size(0)) >= in_size);
    assert((group_size * scales.size(0)) >= in_size);

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

    const size_t pass_in_size = mtx_in_size * (is_sparse ? 2 : 1);
    const int32_t* sparse_msk = is_sparse ? sparse_mask.value().data<int32_t>() : nullptr;

    if (is_sparse) {
        auto actual_sparse = sparse_mask.value();
        assert(actual_sparse.size(0)*2 == mtx_in_size);
        assert(actual_sparse.size(1) == out_size);
    }

    // dispatch conditionally; see comments for intermediate defns for why this is
    #define MACRO_OP(IS_SPARSE, BITS) \
        if ((is_sparse == IS_SPARSE) && (bits == BITS)) { \
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(multiplier.type(), "mvec_int", ([&] { \
            matvec_intermediate_ ## IS_SPARSE ## _ ## BITS<<<blocks, threads>>>( \
                temp_outs.data<float>(), \
                matrix.data<int32_t>(), \
                multiplier.data<scalar_t>(), \
                scales.data<scalar_t>(), \
                zeros.data<scalar_t>(), \
                group_size, in_size, seq_len, pass_in_size, out_size, sparse_msk \
            ); \
        })); \
        }

    #include "kernel_specs.cuh"

    #undef MACRO_OP

    temp_outs = temp_outs.to(torch::kHalf);
    outs.index_put_({"..."}, temp_outs);
}

void matvec_int4(TENSOR_MULT_ARGS) { matvec_packed_int<4>(TENSOR_MULT_PASS); }
void matvec_int3(TENSOR_MULT_ARGS) { matvec_packed_int<3>(TENSOR_MULT_PASS); }
void matvec_int2(TENSOR_MULT_ARGS) { matvec_packed_int<2>(TENSOR_MULT_PASS); }