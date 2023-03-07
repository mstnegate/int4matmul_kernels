#include <torch/all.h>
#include <torch/python.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_fp16.h>

#include <iostream>
#include <cstdint>

#include <mma.h>
using namespace nvcuda;

////////////////////////////////////////////////////////////////////////////////

#define NON_WMMA_BLOCK_SIZE 1024
#define WARP_SIZE 32

////////////////////////////////////////////////////////////////////////////////

template <typename scalar_t>
__device__ inline scalar_t _fma(const scalar_t a, const scalar_t b, const scalar_t c) {
    return a*b + c;
}
template<>
__device__ inline float _fma<float>(const float a, const float b, const float c) {
    return __fmaf_rn(a, b, c);
}
template<>
__device__ inline __half _fma<__half>(const __half a, const __half b, const __half c) {
    return __hfma(a, b, c);
}

// TODO: i wonder if half arithmetic "just works"
template <typename scalar_t>
__device__ inline scalar_t _mult(const scalar_t a, const scalar_t b) {
    return a*b;
}
template<>
__device__ inline __half _mult<__half>(const __half a, const __half b) {
    return __hmul(a, b);
}

template <typename scalar_t>
__device__ inline scalar_t _cast_in(const unsigned int a) {
    return (scalar_t)a;
}
template <>
__device__ inline __half _cast_in<__half>(const unsigned int a) {
    return __uint2half_rn(a);
}

template <typename scalar_t>
__device__ inline float _cast_out(scalar_t a) {
    return (float)a;
}
template <>
__device__ inline float _cast_out(unsigned int a) {
    return (float)a;
}
template <>
__device__ inline float _cast_out(__half a) {
    return __half2float(a);
}

template <typename scalar_t>
__device__ inline scalar_t _cast_from_accum(float a) {
    return (scalar_t)a;
}
template <>
__device__ inline float _cast_from_accum<float>(float a) {
    return a;
}
template <>
__device__ inline __half _cast_from_accum<__half>(float a) {
    return __float2half(a);
}


////////////////////////////////////////////////////////////////////////////////
// actual kernel stuff

const uint32_t MSK = 0xF;

// everything takes these
#define BASE_BOILERPLATE_ARGS(T, U) \
    T* __restrict__ outs, \
    const U* __restrict__ matrix, \
    const T* __restrict__ multiplier, \
    const T* __restrict__ scales, \
    const T* __restrict__ zeros, \
    const size_t in_size, \
    const size_t seq_len, \
    const size_t mtx_in_size, \
    const size_t out_size
#define BASE_SPARSE_MASK_ARGS(U) \
    const U* __restrict__ sparse_mask

#define BOILERPLATE_ARGS(T) BASE_BOILERPLATE_ARGS(T, uint32_t)
#define SPARSE_MASK_ARGS BASE_SPARSE_MASK_ARGS(uint32_t)

#define RECEIVE_BOILERPLATE_ARGS(T) BASE_BOILERPLATE_ARGS(T, int32_t)
#define RECEIVE_SPARSE_MASK_ARGS BASE_SPARSE_MASK_ARGS(int32_t)



// CUDA boilerplate, don't question it
template <bool is_sparse>
__device__ void MMA_Int4_Float( BOILERPLATE_ARGS(float), SPARSE_MASK_ARGS );

template <bool is_sparse>
__device__ void MMA_Int4_Half( BOILERPLATE_ARGS(__half), SPARSE_MASK_ARGS );

template <typename scalar_t>
__global__ void MMV_Int4_Dense( RECEIVE_BOILERPLATE_ARGS(scalar_t) );

template <typename scalar_t>
__global__ void MMV_Int4_Sparse( RECEIVE_BOILERPLATE_ARGS(scalar_t), RECEIVE_SPARSE_MASK_ARGS );

////////////////////////////////////////////////////////////////////////////////

// invalid specializations

template <bool is_sparse>
__device__ void MMA_Int4_Float<is_sparse>(
    BOILERPLATE_ARGS(float),
    SPARSE_MASK_ARGS
) { assert(0); }

template <typename scalar_t>
__global__ void MMV_Int4_Dense(
    RECEIVE_BOILERPLATE_ARGS(scalar_t)
) { assert(0); }

template <typename scalar_t>
__global__ void MMV_Int4_Sparse<float>(
    RECEIVE_BOILERPLATE_ARGS(scalar_t),
    RECEIVE_SPARSE_MASK_ARGS
) { assert(0); }

////////////////////////////////////////////////////////////////////////////////

template <>
__device__ void MMA_Int4_Float<false>(
    BOILERPLATE_ARGS(float),
    SPARSE_MASK_ARGS
) {
    // naive matmul impl; just included for reference
    __shared__ float multiplier_chunk[NON_WMMA_BLOCK_SIZE];

    const size_t mtx_j_idx = blockIdx.x*blockDim.x + threadIdx.x;
    const size_t base_offset = blockIdx.y*blockDim.y*in_size + blockIdx.z*seq_len*in_size;
    float accum = 0.0;

    float scale = 0;
    float zero = 0;

    if (mtx_j_idx < out_size) {
        scale = _cast_out(scales[mtx_j_idx]);
        zero =_cast_out(zeros[mtx_j_idx]);
    }

    // first: blocking along in_size (the row of x we're multiplying)
    for(size_t r = 0; r < in_size; r += NON_WMMA_BLOCK_SIZE){
        const size_t i_idx = r + threadIdx.x + base_offset;
        if ((r + threadIdx.x) < in_size) {
            multiplier_chunk[threadIdx.x] = _cast_out(multiplier[i_idx]);
        } else {
            multiplier_chunk[threadIdx.x] = 0;
        }

        __syncthreads();

        size_t mzo = r/8;
        size_t upto = NON_WMMA_BLOCK_SIZE;
        if ((in_size - r) < upto) {
            upto = in_size - r;
        }
        if (mtx_j_idx < out_size) {
            for(size_t mtik = 0; mtik < upto; mtik += 8, ++mzo) {
                uint32_t a = matrix[mtx_j_idx + mzo*out_size];

                #pragma unroll
                for (size_t ii = 0; ii < 8; ++ii) {
                    accum += multiplier_chunk[mtik + ii] * (scale * _cast_out(a & MSK) + zero);
                    a >>= 4;
                }
            }
        }
        __syncthreads();
    }

    const size_t write_i_idx = blockIdx.y*blockDim.y;
    const size_t write_j_idx = mtx_j_idx;
    // i iterates 1 at a time so no need for safety checks
    if ((write_j_idx < out_size) && (write_i_idx < seq_len)) {
        outs[
            write_j_idx
            + write_i_idx*out_size
            + blockIdx.z*out_size*seq_len
        ] = _cast_from_accum<float>(accum);
    }
}

template <bool is_sparse>
__device__ void load_in(
    const bool is_empty,
    __half* __restrict__ dest,
    const uint32_t* __restrict__ src,
    const uint32_t* __restrict__ sparse_mask,
    const size_t read_stride,
    const size_t read_j,
    const size_t read_i,
    const size_t src_i,
    const size_t src_j,
    const __half scale,
    const __half zero
);

template <>
__device__ void load_in<false>(
    const bool is_empty,
    __half* __restrict__ dest,
    const uint32_t* __restrict__ src,
    const uint32_t* __restrict__ sparse_mask,
    const size_t read_stride,
    const size_t read_j,
    const size_t read_i,
    const size_t src_i,
    const size_t src_j,
    const __half scale,
    const __half zero
) {
    const __half HALF_ZERO = __float2half(0.0f);

    dest += (read_i * 8 * 32);

    if (!is_empty) {
        uint32_t v = src[src_j + read_stride * (src_i + read_i)];
        #pragma unroll
        for(size_t ii = 0; ii < 8; ++ii) {
            dest[ii*32] = _fma(scale, __float2half((float)(v & MSK)), zero);
            v >>= 4;
        }
    } else {
        #pragma unroll
        for(size_t ii = 0; ii < 8; ++ii) {
            dest[ii*32] = HALF_ZERO;
        }
    }
}

template <>
__device__ void load_in<true>(
    const bool is_empty,
    __half* __restrict__ dest,
    const uint32_t* __restrict__ src,
    const uint32_t* __restrict__ sparse_mask,
    const size_t read_stride,
    const size_t read_j,
    const size_t read_i,
    const size_t src_i,
    const size_t src_j,
    const __half scale,
    const __half zero
) {
    // similar idea to dense model except we have half the actual weights to
    // load, so every weight is shared by 2 threads
    const __half HALF_ZERO = __float2half(0.0f);

    __shared__ uint8_t coord_buf[32 * 16];

    if (!is_empty) {
        // naive approach: grab mask, iterate, calculate coords
        {
            uint32_t sp_msk = sparse_mask[src_j + read_stride * (src_i/4)];
            // TODO: make sure read_i's stride is 32
            size_t wdx = read_i * 8;
            size_t coord = __popc(sp_msk & ((1 << (wdx))- 1));

            sp_msk >>= wdx;
            #pragma unroll
            for (size_t ii = 0; ii < 8; ++ii){
                if ((sp_msk & 1) == 1) {
                    coord_buf[read_j + coord*32] = wdx;
                    ++coord;
                } else {
                    dest[wdx*32] = HALF_ZERO;
                }

                ++wdx;
                sp_msk >>= 1;
            }
        }

        __syncthreads();

        uint32_t v = src[src_j + read_stride * ((src_i + read_i)/2)];
        v >>= (read_i % 2) * 16;
        v &= 0xFFFF;

        size_t seek = read_i*4*32 + read_j;

        #pragma unroll
        for(size_t ii = 0; ii < 4; ++ii) {
            dest[coord_buf[seek + (ii*32)] * 32] =
                _fma(scale, __float2half((float)(v & MSK)), zero);
            v >>= 4;
        }
    } else {
        dest += (read_i * 8 * 32);

        #pragma unroll
        for(size_t ii = 0; ii < 8; ++ii) {
            dest[ii*32] = HALF_ZERO;
        }
    }
}

// generic half-precision matrix mult
template <bool is_sparse>
__device__ void MMA_Int4_Half(
    BOILERPLATE_ARGS(__half),
    SPARSE_MASK_ARGS
) {
    // TODO: batch dim

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    // TODO: flip load and storage order
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    const size_t warpI = threadIdx.x / WARP_SIZE;
    const size_t warpJ = threadIdx.y;

    const int read_j = (threadIdx.x + threadIdx.y * blockDim.x) % 32;
    const int read_i = (threadIdx.x + threadIdx.y * blockDim.x) / 32;

    const int j_idx = blockIdx.x * 32 + read_j;

    const __half HALF_ZERO = __float2half(0.0f);
    __half scale = HALF_ZERO;
    __half zero = HALF_ZERO;
    if (j_idx < out_size) {
        scale = scales[j_idx];
        zero = zeros[j_idx];
    }

    // TODO: might be able to get better warp occupancy by splitting wt_buf
    //       in half and loading as needed (we only need 2 16x16 blocks at a time)
    __shared__ __half wt_buf[32 * 32];
    __shared__ __half mult_buf[32 * 32];

    for(size_t mtx_i = 0; mtx_i < mtx_in_size; mtx_i += 4) {
        const int write_i = read_i * 8;
        load_in<is_sparse>(
            (mtx_i + read_i) >= mtx_in_size,

            wt_buf + read_j,
            matrix,
            sparse_mask,

            out_size,

            read_j,
            read_i,
            mtx_i,
            j_idx,

            scale,
            zero
        );

        #pragma unroll
        for(size_t ii = 0; ii < 8; ++ii) {
            const size_t rd_i = (blockIdx.y*32 + write_i + ii);
            const size_t rd_j = mtx_i*8 + read_j;

            if ((rd_j < in_size) && (rd_i < seq_len)) {
                mult_buf[(write_i + ii)*32 + read_j] = multiplier[
                     rd_j + rd_i*in_size + blockIdx.z*in_size*seq_len
                ];
            } else {
                mult_buf[(write_i + ii)*32 + read_j] = HALF_ZERO;
            }
        }

        __syncthreads();

        // load up weights block into matrix buf
        wmma::load_matrix_sync(b_frag, wt_buf + warpI*16 + warpI*32*16, 32U);
        // ...and the multiplier block
        wmma::load_matrix_sync(a_frag, (mult_buf + warpI*16 + warpJ*16*32), 32U);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncthreads();
        wmma::load_matrix_sync(b_frag, wt_buf + warpI*16 + (1-warpI)*32*16, 32U);
        wmma::load_matrix_sync(a_frag, (mult_buf + (1-warpI)*16 + warpJ*16*32), 32U);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncthreads();
    }

    __shared__ float output_buf[32 * 32];
    // TODO: check if possible to save shared mem here (needs saving registers too though)
    wmma::store_matrix_sync(output_buf + warpJ*16*32 + warpI*16, acc_frag, 32U, wmma::mem_row_major);
    __syncthreads();

    #pragma unroll
    for(int ii = 0; ii < 8; ++ii){
        const int write_j = read_j + (blockIdx.x * 32);
        const int write_i = (ii*4+read_i) + (blockIdx.y * 32);

        if ((write_i < seq_len) && (write_j < out_size)) {
            outs[
                write_j + write_i*out_size + blockIdx.z*out_size*seq_len
            ] = __float2half(output_buf[read_j + (ii*4+read_i) * 32]);
        }
    }
}


////////////////////////////////////////////////////////////////////////////////

template <>
__global__ void MMV_Int4_Dense<float>(
    RECEIVE_BOILERPLATE_ARGS(float)
) {
    MMA_Int4_Float<false>(
        outs, reinterpret_cast<const uint32_t*>(matrix), multiplier, scales, zeros,
        in_size, seq_len, mtx_in_size, out_size, nullptr
    );
}
template <>
__global__ void MMV_Int4_Sparse<float>(
    RECEIVE_BOILERPLATE_ARGS(float),
    RECEIVE_SPARSE_MASK_ARGS
) {
    MMA_Int4_Float<true>(
        outs, reinterpret_cast<const uint32_t*>(matrix), multiplier, scales, zeros,
        in_size, seq_len, mtx_in_size, out_size, reinterpret_cast<const uint32_t*>(sparse_mask)
    );
}

template <>
__global__ void MMV_Int4_Dense<c10::Half>(
    RECEIVE_BOILERPLATE_ARGS(c10::Half)
) {
    MMA_Int4_Half<false>(
        // torch stuff; torch doesn't store halfs internally as CUDA halfs,
        // but they *are* bit-compatible so reinterpret_cast solves the issue
        reinterpret_cast<__half*>(outs),
        reinterpret_cast<const uint32_t*>(matrix),
        reinterpret_cast<const __half*>(multiplier),
        reinterpret_cast<const __half*>(scales),
        reinterpret_cast<const __half*>(zeros),
        in_size, seq_len, mtx_in_size, out_size, nullptr
    );
}
template <>
__global__ void MMV_Int4_Sparse<c10::Half>(
    RECEIVE_BOILERPLATE_ARGS(c10::Half),
    RECEIVE_SPARSE_MASK_ARGS
) {
    MMA_Int4_Half<true>(
        reinterpret_cast<__half*>(outs),
        reinterpret_cast<const uint32_t*>(matrix),
        reinterpret_cast<const __half*>(multiplier),
        reinterpret_cast<const __half*>(scales),
        reinterpret_cast<const __half*>(zeros),
        in_size, seq_len, mtx_in_size, out_size,
        reinterpret_cast<const uint32_t*>(sparse_mask)
    );
}


void matmul_int4(
    torch::Tensor outs,
    torch::Tensor matrix,
    torch::Tensor x,
    torch::Tensor scales,
    torch::Tensor zeros,
    c10::optional<torch::Tensor> sparse_mask
) {
    const bool is_sparse = sparse_mask.has_value() && sparse_mask.value().defined();

    // perform matrix multiplication:
    //    x * W^T = O
    // x : [batch_size, seq_len, in_size]
    // W : [out_size, in_size]
    // O : [batch_size, seq_len, out_size]
    //
    // with zeros and scales such that
    //    W = scales * Wq + zeros
    // scales : [out_size,]
    // zeros  : [out_size,]

    const auto batch_size = x.size(0);
    const auto seq_len = x.size(1);
    const auto in_size = x.size(2);

    const auto mtx_in_size = matrix.size(0);
    const auto out_size = matrix.size(1);

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

    const auto THREAD_X = NON_WMMA_BLOCK_SIZE;
    const auto THREAD_Y = 1;

    dim3 threads(THREAD_X, THREAD_Y);
    dim3 blocks(
        (out_size + THREAD_X - 1) / THREAD_X,
        (seq_len + THREAD_Y - 1) / THREAD_Y,
        batch_size
    );

    if (x.dtype() == at::kHalf) {
        threads.x = WARP_SIZE * 2;
        threads.y = 2;

        blocks.x = (out_size + 32 - 1) / 32;
        blocks.y = (seq_len + 32 - 1) / 32;
    }

    if (is_sparse) {
        auto actual_sparse = sparse_mask.value();
        assert(actual_sparse.size(0)*2 == mtx_in_size);
        assert(actual_sparse.size(1) == out_size);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.type(), "mmv_int4", ([&] {
            MMV_Int4_Sparse<<<blocks, threads>>>(
                outs.data<scalar_t>(),
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
            MMV_Int4_Dense<<<blocks, threads>>>(
                outs.data<scalar_t>(),
                matrix.data<int32_t>(),
                x.data<scalar_t>(),
                scales.data<scalar_t>(),
                zeros.data<scalar_t>(),
                in_size, seq_len, mtx_in_size, out_size
            );
        }));
    }

}
