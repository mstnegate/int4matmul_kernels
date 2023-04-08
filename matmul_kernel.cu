#include <iostream>
#include <cstdint>

#include <mma.h>
using namespace nvcuda;

#include "matmul.cuh"

////////////////////////////////////////////////////////////////////////////////

union BUF_TYPE {
    __half h[BLOCK_SIZE * BUF_MTX_WIDTH];
    float f[BLOCK_SIZE * BUF_MTX_F32_WIDTH];
};

typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> WMMA_A_FRAG;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> WMMA_B_FRAG;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> WMMA_B_FRAG_COL;
typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> WMMA_ACC_FRAG;

////////////////////////////////////////////////////////////////////////////////
// factored out loads

// conventions: offset is location within the block, pos is the location within the matrices

__device__ void gmem_load_multiplier(
    const __half* __restrict__ mult_in,
    __half* __restrict__ buf,

    const size_t i_offset,
    const size_t j_offset,

    const size_t i_pos,
    const size_t j_pos,

    const size_t in_size,
    const size_t seq_len
) {
    const auto off_end = mult_in + in_size*seq_len;
    const __half ZERO = __float2half(0.0f);

    // TODO: adjust these schemes for other tile sizes

#ifdef RESPECT_STRICT_ALIASING
    const size_t downset = i_pos + i_offset;

    const size_t back_increment = 4*BUF_MTX_WIDTH - WARP_SIZE;
    const size_t in_back_increment = 4*in_size - WARP_SIZE;

    const __half* m_in = mult_in + in_size*downset + j_offset + j_pos;
    __half* m_out = buf + BUF_MTX_WIDTH*i_offset + j_offset;
    #pragma unroll
    for(auto i = 0; i < (BLOCK_SIZE / 4); ++i) {
        if (m_in < off_end) {
            *m_out = *m_in;
            m_in += WARP_SIZE;
            m_out += WARP_SIZE;
        } else {
            *m_out = ZERO;
        }

        if (m_in < off_end) {
            *m_out = *m_in;
            m_in += in_back_increment;
            m_out += back_increment;
        } else {
            *m_out = ZERO;
        }
    }
#else
    // TODO: make positioning (offsetting) more consistent
    const size_t downset = i_pos + i_offset*WMMA_CHUNK_SIZE;

    const __half* m_in = mult_in + in_size*downset + j_offset*2 + j_pos;
    __half* m_out = buf + BUF_MTX_WIDTH*i_offset*WMMA_CHUNK_SIZE + j_offset*2;

    const int C_ZERO = *(int*)&ZERO;

    #pragma unroll
    for(auto i = 0; i < (BLOCK_SIZE / 4); ++i) {
        if (m_in < off_end) {
            *(int*)m_out = *(int*)m_in;
        } else {
            *(int*)m_out = C_ZERO;
        }

        m_out += BUF_MTX_WIDTH;
        m_in += in_size;
    }
#endif
}


template <bool is_transposed>
__device__ size_t smem_mm_calc_coord(
    const size_t i, const size_t j, const size_t k, const size_t warp_idx
);

template <>
__device__ size_t smem_mm_calc_coord<false>(
    const size_t i, const size_t j, const size_t k, const size_t warp_idx
) {
    return (
        (j*WMMA_CHUNK_SIZE + k*WMMA_CHUNK_SIZE*BUF_MTX_WIDTH) // sweep-based
        + (((warp_idx%SB_DIM_B)*SB_DIM_N)*WMMA_CHUNK_SIZE) // warp-based
    );
}
template <>
__device__ size_t smem_mm_calc_coord<true>(
    const size_t i, const size_t j, const size_t k, const size_t warp_idx
) {
    return (
        (j*WMMA_CHUNK_SIZE*BUF_MTX_WIDTH + k*WMMA_CHUNK_SIZE) // sweep-based
        + (((warp_idx%SB_DIM_B)*SB_DIM_N)*WMMA_CHUNK_SIZE*BUF_MTX_WIDTH) // warp-based
    );
}

template <typename a_type, typename b_type, bool is_transposed>
__device__ void gen_smem_block_matmul(
    WMMA_ACC_FRAG acc[][SB_DIM_N],
    const __half* __restrict__ mult,
    const __half* __restrict__ wts,
    const size_t warp_index
) {
    a_type a;
    b_type b[SB_DIM_N];

    #pragma unroll
    for(auto k = 0; k < SB_DIM_K; ++k) {
        // go down the multiplier dimension
        #pragma unroll
        for(auto i = 0; i < SB_DIM_M; ++i) {
            // and now: load in; this one is actually genuinely weird
            wmma::load_matrix_sync(
                a,
                (
                    mult
                    // sweep-based position
                    + (k*WMMA_CHUNK_SIZE + i*WMMA_CHUNK_SIZE*BUF_MTX_WIDTH)
                    // warp-based position
                    + (((warp_index/SB_DIM_B)*SB_DIM_M)*WMMA_CHUNK_SIZE*BUF_MTX_WIDTH)
                ),
                BUF_MTX_WIDTH
            );

            // go across the weights dimension
            #pragma unroll
            for(auto j = 0; j < SB_DIM_N; ++j) {
                if (i == 0) {
                    // first load; gets recycled of course
                    wmma::load_matrix_sync(
                        b[j],
                        (
                            wts + smem_mm_calc_coord<is_transposed>(i, j, k, warp_index)
                        ),
                        BUF_MTX_WIDTH
                    );
                }

                wmma::mma_sync(acc[i][j], a, b[j], acc[i][j]);
            }
        }

    }
}

template <bool is_sparse>
__device__ void smem_block_matmul(
    WMMA_ACC_FRAG acc[][SB_DIM_N],
    const __half* __restrict__ mult,
    const __half* __restrict__ wts,
    const size_t warp_index
);
template <>
__device__ void smem_block_matmul<false>(
    WMMA_ACC_FRAG acc[][SB_DIM_N],
    const __half* __restrict__ mult,
    const __half* __restrict__ wts,
    const size_t warp_index
) {
    gen_smem_block_matmul<WMMA_A_FRAG, WMMA_B_FRAG, false>(acc, mult, wts, warp_index);
}
template <>
__device__ void smem_block_matmul<true>(
    WMMA_ACC_FRAG acc[][SB_DIM_N],
    const __half* __restrict__ mult,
    const __half* __restrict__ wts,
    const size_t warp_index
) {
    gen_smem_block_matmul<WMMA_A_FRAG, WMMA_B_FRAG_COL, true>(acc, mult, wts, warp_index);
}


__device__ void smem_write_accs(
    WMMA_ACC_FRAG acc[][SB_DIM_N],
    float* __restrict__ write_buf,
    const size_t warp_index
) {
    #pragma unroll
    for(auto i = 0; i < SB_DIM_M; ++i) {
        // go across the weights dimension
        #pragma unroll
        for(auto j = 0; j < SB_DIM_N; ++j) {
            // L1 issues *here*, apparently...?
            wmma::store_matrix_sync(
                write_buf
                + (j + (warp_index%SB_DIM_B)*SB_DIM_N)*WMMA_CHUNK_SIZE
                + (i + (warp_index/SB_DIM_B)*SB_DIM_M)*BUF_MTX_F32_WIDTH*WMMA_CHUNK_SIZE,
                acc[i][j],
                BUF_MTX_F32_WIDTH,
                wmma::mem_row_major
            );
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

template <bool is_sparse, uint32_t BITS>
__device__ void MMA_Int4_Half(
    BOILERPLATE_ARGS(__half),
    SPARSE_MASK_ARGS
) {
    __shared__ BUF_TYPE base_buf;
    auto buf = base_buf.h;

    // TODO: investigate putting unpacked weights to bottom of packed buf and
    //       unpacking top-to-bottom; should save enough memory to boost warp
    //       occupancy on a 3080; stacking buffers vertically might make this
    //       easier (but cause bank conflicts (?))
    __shared__ uint32_t wt_buf[(BLOCK_SIZE * BITS / 32) * BLOCK_SIZE];
    __half* wt_unpacked_buf = buf + BLOCK_SIZE;

    const __half ZERO = __float2half(0.0f);

    const size_t warpIdx = threadIdx.y;
    // TODO: properly index these
    const size_t warpSubIdx = warpIdx % 2;
    const size_t warpSuperIdx = warpIdx / 2;
    const size_t warpOffset = warpSubIdx * WARP_SIZE + threadIdx.x;

    // adjust for batch dim
    multiplier += blockIdx.z*seq_len*in_size;
    outs += blockIdx.z*seq_len*out_size;

    WMMA_ACC_FRAG acc[SB_DIM_M][SB_DIM_N];
    #pragma unroll
    for (auto i = 0; i < SB_DIM_M; ++i) {
        #pragma unroll
        for (auto j = 0; j < SB_DIM_N; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    // TODO: reduce redundant group quant fetches
    constexpr size_t step = BLOCK_SIZE * BITS / 32;
    for(size_t mtx_i = 0; mtx_i < mtx_in_size; mtx_i += step) {
        __half scale = ZERO;
        __half zero = ZERO;
        {
            const auto j_idx = (blockIdx.x*BLOCK_SIZE + warpOffset) + ((mtx_i*32/BITS)/group_size)*out_size;
            scale = scales[j_idx];
            zero = zeros[j_idx];
            #ifdef FMA_TRANSFORM
                zero = __hneg(__hmul(zero, scale))
            #endif
        }

        // GMEM loading chunk

        // grab the weights first since there's no bounds-checking to desync things
        gmem_load_weights<is_sparse, BITS, BLOCK_SIZE>::load(
            matrix, wt_buf, sparse_mask,
            warpSuperIdx, warpOffset,
            mtx_i, blockIdx.x*BLOCK_SIZE,
            out_size);

        gmem_load_multiplier(multiplier, buf,
                             warpIdx, threadIdx.x,
                             blockIdx.y*BLOCK_SIZE, mtx_i*32/BITS,
                             in_size, seq_len);
        __syncthreads();

        // everything is now in GMEM; all our operations now touch only SMEM

        // step 1: unpack packed weights from SMEM -> SMEM
        smem_unpack_weights<is_sparse, BLOCK_SIZE, BUF_MTX_WIDTH, BITS, Quantization::DYNAMIC_EXPONENT_SYM>::load(
            wt_buf, wt_unpacked_buf,
            warpSuperIdx, warpOffset,
            scale, zero);

        __syncthreads();

        // step 2: actual matrix mult
        smem_block_matmul<is_sparse>(acc, buf, wt_unpacked_buf, warpIdx);
        __syncthreads();
    }

    // we're done with the main loop; dump our data back to GMEM
    float* out_buf = base_buf.f;

    // dump accumulators into our write buffer
    smem_write_accs(acc, out_buf, warpIdx);

    __syncthreads(); // be very certain

    // and then dump from SMEM => GMEM
    #ifdef RESPECT_STRICT_ALIASING
        #pragma unroll
        for(auto i = 0; i < (BLOCK_SIZE / 2); ++i) {
            const size_t write_j = blockIdx.x*BLOCK_SIZE + warpOffset;
            const size_t write_i = blockIdx.y*BLOCK_SIZE + i*2 + warpSuperIdx;
            if ((write_j < out_size) && (write_i < seq_len)) {
                outs[write_j + write_i*out_size] = \
                    __float2half(out_buf[warpOffset + (warpSuperIdx + i*2)*BUF_MTX_F32_WIDTH]);
            }
        }
    #else
        half a[2]; // temp buf because i don't trust half2s
        #pragma unroll
        for(auto i = 0; i < (BLOCK_SIZE / 4); ++i) {
            const size_t write_j = blockIdx.x*BLOCK_SIZE + threadIdx.x*2;
            const size_t write_i = blockIdx.y*BLOCK_SIZE + warpIdx*WMMA_CHUNK_SIZE + i;
            const size_t v = warpIdx*WMMA_CHUNK_SIZE + i;
            if ((write_j < out_size) && (write_i < seq_len)) {
                a[0] = __float2half(out_buf[threadIdx.x*2 + v*BUF_MTX_F32_WIDTH]);
                a[1] = __float2half(out_buf[threadIdx.x*2+1 + v*BUF_MTX_F32_WIDTH]);

                *(int*)(outs + write_j + write_i*out_size) = *(int*)&a;
            }
        }
    #endif
}

////////////////////////////////////////////////////////////////////////////////

// torch dispatch requires definitions for fp32 and fp64; look at older commits
// (0d20a43 or older) if you neeed fp32 for some reason; fp64 not supported

// as of writing, CUDA doesn't allow defining __global__ functions as member
// methods (including static inlines) which means partial template spec isn't
// available (AFAICT anyway); this obliges us to do the horrific act of
// manually specifying every template full specialization then have compiletime
// code switching downstream
#define ARG_LST \
    reinterpret_cast<__half*>(outs), \
    reinterpret_cast<const uint32_t*>(matrix), \
    reinterpret_cast<const __half*>(multiplier), \
    reinterpret_cast<const __half*>(scales), \
    reinterpret_cast<const __half*>(zeros), \
    group_size, in_size, seq_len, mtx_in_size, out_size

#define MACRO_OP(IS_SPARSE, BITS) \
    template <typename scalar_t> \
    __global__ void matmul_intermediate_ ## IS_SPARSE ## _ ## BITS( \
        RECEIVE_BOILERPLATE_ARGS(scalar_t), RECEIVE_SPARSE_MASK_ARGS \
    ) { assert(0); } \
    template <> \
    __global__ void matmul_intermediate_ ## IS_SPARSE ## _ ## BITS<c10::Half>( \
        RECEIVE_BOILERPLATE_ARGS(c10::Half), RECEIVE_SPARSE_MASK_ARGS \
    ) { \
        if (IS_SPARSE) { \
            MMA_Int4_Half<true, BITS>(ARG_LST, reinterpret_cast<const uint32_t*>(sparse_mask)); \
        } else { \
            MMA_Int4_Half<false, BITS>(ARG_LST, nullptr); \
        } \
    }

#include "kernel_specs.cuh"

#undef ARG_LST
#undef MACRO_OP


template <uint32_t bits>
void matmul_packed_int(TENSOR_MULT_ARGS) {
    const bool is_sparse = sparse_mask.has_value() && sparse_mask.value().defined();

    if (group_size < 0) {
        // don't pick something too big in case of weird uint shenanigans
        group_size = 0x0FFFFFFF;
    } else {
        assert(group_size % 64 == 0);
    }

    // perform matrix multiplication:
    //    x * W^T = O
    // x : [batch_size, seq_len, in_size]
    // W : [out_size, in_size]
    // O : [batch_size, seq_len, out_size]
    //
    // with zeros and scales such that
    //    W = scales * Wq + zeros
    // scales : [n_groups, out_size,]
    // zeros  : [n_groups, out_size,]

    const auto batch_size = multiplier.size(0);
    const auto seq_len = multiplier.size(1);
    const auto in_size = multiplier.size(2);

    const auto mtx_in_size = matrix.size(0);
    const auto out_size = matrix.size(1);

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
    assert(out_size % BLOCK_SIZE == 0);

    const auto THREAD_X = WARP_SIZE;
    const auto THREAD_Y = WMMA_CHUNK_COUNT;

    dim3 threads(THREAD_X, THREAD_Y);
    dim3 blocks(
        (out_size + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE,
        batch_size
    );

    // TODO: toss exceptions for unsupported sparse/bit combinations

    // let sparse pretend it's the same size as other things; only 4-bit
    // supported for it currently anyway
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
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(multiplier.type(), "mmv_int", ([&] { \
            matmul_intermediate_ ## IS_SPARSE ## _ ## BITS<<<blocks, threads>>>( \
                outs.data<scalar_t>(), \
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
}

void matmul_int4(TENSOR_MULT_ARGS) { matmul_packed_int<4>(TENSOR_MULT_PASS); }
void matmul_int3(TENSOR_MULT_ARGS) { matmul_packed_int<3>(TENSOR_MULT_PASS); }
void matmul_int2(TENSOR_MULT_ARGS) { matmul_packed_int<2>(TENSOR_MULT_PASS); }