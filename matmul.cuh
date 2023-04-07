////////////////////////////////////////////////////////////////////////////////
// optional defines/flags:

// transforms (x - zp)*s => (s*x + (-zp*s)) during unpacking step to allow
//  for a FMA instead of a sub-mul; might help if ALU pipe's bottlenecked
//  somehow (it wasn't for me, but who knows what'll happen)
// #define FMA_TRANSFORM

// replaces GMEM => SMEM load routines with [slower] code that respects
//  strict aliasing rules; the current method technically uses UB but it
//  seems to be in common-enough use that nvcc probably won't break it
// #define RESPECT_STRICT_ALIASING

#include <torch/all.h>
#include <torch/python.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_fp16.h>

////////////////////////////////////////////////////////////////////////////////
// useful global defines; don't touch these

#define WARP_SIZE 32
#define FULL_MASK 0xFFFFFFFF
#define MSK 0xF

// everything takes these
#define BASE_BOILERPLATE_ARGS(T, OT, U) \
    OT* __restrict__ outs, \
    const U* __restrict__ matrix, \
    const T* __restrict__ multiplier, \
    const T* __restrict__ scales, \
    const T* __restrict__ zeros, \
    const size_t group_size, \
    const size_t in_size, \
    const size_t seq_len, \
    const size_t mtx_in_size, \
    const size_t out_size
#define BASE_SPARSE_MASK_ARGS(U) \
    const U* __restrict__ sparse_mask

#define BOILERPLATE_ARGS(T) BASE_BOILERPLATE_ARGS(T, T, uint32_t)
#define BOILERPLATE_ACCUM_ARGS(T, O) BASE_BOILERPLATE_ARGS(T, O, uint32_t)
#define SPARSE_MASK_ARGS BASE_SPARSE_MASK_ARGS(uint32_t)

#define RECEIVE_BOILERPLATE_ARGS(T) BASE_BOILERPLATE_ARGS(T, T, int32_t)
#define RECEIVE_BOILERPLATE_ACCUM_ARGS(T, O) BASE_BOILERPLATE_ARGS(T, O, int32_t)
#define RECEIVE_SPARSE_MASK_ARGS BASE_SPARSE_MASK_ARGS(int32_t)

////////////////////////////////////////////////////////////////////////////////

#define WMMA_CHUNK_SIZE 16
// TODO: properly lift some of these to template parameters
#define WMMA_CHUNK_COUNT 4
#define BLOCK_SIZE (WMMA_CHUNK_SIZE * WMMA_CHUNK_COUNT)

// spacing between rows in the buffer; sometimes necessary in case the kernel
// steps on its own toes while loading in/out; keeping it here for now
#define BUF_MTX_PADDING 8
#define BUF_MTX_WIDTH (BLOCK_SIZE*2 + BUF_MTX_PADDING)
#define BUF_MTX_F32_WIDTH (BLOCK_SIZE + BUF_MTX_PADDING/2)
#define WARP_ITER_COUNT ((BLOCK_SIZE * BLOCK_SIZE) / (WARP_SIZE * WMMA_CHUNK_COUNT))

// subblock dim handling in terms of WMMA blocks; convention here is:
//  subblock is MxN matrix, arranged in AxB block matrix (M*A=B*N=BLOCK_SIZE)
#define SB_DIM_A (WMMA_CHUNK_COUNT / 2)
#define SB_DIM_B (2)
#define SB_DIM_M (2)
#define SB_DIM_N (WMMA_CHUNK_COUNT/2)
// K dim for matrix mults
#define SB_DIM_K (WMMA_CHUNK_COUNT)

////////////////////////////////////////////////////////////////////////////////
// decls for common functionality

enum class Quantization {
    LINEAR,
    EXPONENT_SYM,
    DYNAMIC_EXPONENT_SYM,
};

#define DEQUANTIZZE_ARGS \
    uint32_t v, const __half scale, const __half zero
template <Quantization qm>
__device__ inline __half dequantize(DEQUANTIZZE_ARGS);

template<>
__device__ inline __half dequantize<Quantization::LINEAR>(DEQUANTIZZE_ARGS) {
    #ifndef FMA_TRANSFORM
        return __hmul(__hsub(__uint2half_rn(v & MSK), zero), scale);
    #else
        return __hfma(__uint2half_rn(v & MSK), scale, zero);
    #endif
};

#define EXP_MSK 0x7
template<>
__device__ inline __half dequantize<Quantization::EXPONENT_SYM>(DEQUANTIZZE_ARGS) {
    int32_t iv = (v & 0x8 ? -1 : 1) * (int)(1 << (v & EXP_MSK));
    return __hfma(__int2half_rn(iv), scale, zero);
};

template<>
__device__ inline __half dequantize<Quantization::DYNAMIC_EXPONENT_SYM>(DEQUANTIZZE_ARGS) {
    // for efficiency this needs to be handled further up in the call stack
    assert(0);
};
#undef DEQUANTIZZE_ARGS

////////////////////////////////////////////////////////////////////////////////
// off-loaded loads

// TODO: true 3-bit specializations

// function partial specialization isn't really a thing, so...

#define LOAD_WEIGHTS_ARGS \
    const uint32_t* __restrict__ wt_mtx, \
    uint32_t* __restrict__ buf, \
    const uint32_t* __restrict__ sparse_mask, \
    const size_t i_offset, \
    const size_t j_offset, \
    const size_t i_pos, \
    const size_t j_pos, \
    const size_t out_size

template<bool is_sparse, size_t STRIDE>
struct gmem_load_weights {
    __device__ inline static void load(LOAD_WEIGHTS_ARGS);
};
template<size_t STRIDE>
struct gmem_load_weights<false, STRIDE> {
    __device__ inline static void load(LOAD_WEIGHTS_ARGS) {
        // sparse_mask is unused, of course
        const size_t down_size = (BLOCK_SIZE/8/2);
        const uint32_t* wt_in = wt_mtx + j_pos + j_offset + ((i_pos + i_offset*down_size) * out_size);
        uint32_t* wt_out = buf + STRIDE*i_offset*down_size + j_offset;
        #pragma unroll
        for(auto i = 0; i < down_size; ++i) {
            *wt_out = *wt_in;
            wt_in += out_size;
            wt_out += STRIDE;
        }
    }
};
template<size_t STRIDE>
struct gmem_load_weights<true, STRIDE> {
    __device__ inline static void load(LOAD_WEIGHTS_ARGS) {
        // housekeeping note: we normally have N rows of data; due to sparsity
        // we now have N//2, then N//4 masking entries; the weights are stored
        // at the top of the buffer and the masks at the bottom
        //
        // only other thing to be wary of is that coordinates are based on unpacked
        // matrix coordinates; we have to divide by 2 for weights and 4 for mask

        const size_t WEIGHT_ROWS = (BLOCK_SIZE / 8 / 2);
        const size_t MASK_ROWS = (BLOCK_SIZE / 8 / 4);

        const uint32_t* wt_in = wt_mtx + j_pos + j_offset + ((i_pos/2 + i_offset) * out_size);
        uint32_t* wt_out = buf + STRIDE*i_offset + j_offset;
        #pragma unroll
        for(auto i = 0; i < (WEIGHT_ROWS / 2); ++i) {
            *wt_out = *wt_in;
            wt_in += 2*out_size;
            wt_out += 2*STRIDE;
        }

        // now the mask
        const uint32_t* msk_in = (
            sparse_mask
            + (j_pos + j_offset)
            + ((i_pos/4 + i_offset) * out_size)
        );
        wt_out = buf + STRIDE*((BLOCK_SIZE/8 - MASK_ROWS) + i_offset) + j_offset;
        #pragma unroll
        for(auto i = 0; i < (MASK_ROWS / 2); ++i) {
            *wt_out = *msk_in;
            msk_in += 2*out_size;
            wt_out += 2*STRIDE;
        }
    }
};

#undef LOAD_WEIGHTS_ARGS


#define UNPACK_WEIGHTS_ARGS \
    const uint32_t* __restrict__ packed_wts, \
    __half* __restrict__ unpacked_wts, \
    const size_t i_offset, \
    const size_t j_offset, \
    const __half scale, \
    const __half zero

template<bool is_sparse, size_t STRIDE_IN, size_t STRIDE_OUT, Quantization qm>
struct smem_unpack_weights_inner {
    __device__ inline static void load(UNPACK_WEIGHTS_ARGS);
};

template<size_t STRIDE_IN, size_t STRIDE_OUT, Quantization qm>
struct smem_unpack_weights_inner<false, STRIDE_IN, STRIDE_OUT, qm> {
    __device__ inline static void load(UNPACK_WEIGHTS_ARGS) {
        const size_t down_size = (BLOCK_SIZE/8/2);

        const uint32_t* wt_in = packed_wts + j_offset + STRIDE_IN*i_offset*down_size;
        __half* wt_out = unpacked_wts + j_offset + STRIDE_OUT*i_offset*8*down_size;

        #pragma unroll
        for(auto i = 0; i < down_size; ++i) {
            uint32_t v = *wt_in;
            #pragma unroll
            for(auto j = 0; j < 8; ++j) {
                *wt_out = dequantize<qm>(v, scale, zero);
                v >>= 4;
                wt_out += STRIDE_OUT;
            }
            wt_in += STRIDE_IN;
        }
    }
};

#ifndef RESPECT_STRICT_ALIASING
struct WIDE_S_SB {
    uint32_t lsb;
    uint32_t msb;
};
struct WIDE_S_AUX {
    __half zero;
    __half scale;
    uint32_t msk;
};
union WIDE_S {
    uint64_t p;
    WIDE_S_SB sb;
    WIDE_S_AUX aux;
};
#endif

template<size_t STRIDE_IN, size_t STRIDE_OUT, Quantization qm>
struct smem_unpack_weights_inner<true, STRIDE_IN, STRIDE_OUT, qm> {
    __device__ inline static void load(UNPACK_WEIGHTS_ARGS) {
        const __half ZERO = __float2half(0.0f);
        const size_t MASK_ROWS = (BLOCK_SIZE / 8 / 4);

        uint32_t ref_msk = *(
            (packed_wts + (BLOCK_SIZE/8 - MASK_ROWS)*STRIDE_IN)
            + j_offset + STRIDE_IN*i_offset
        );
        #ifdef RESPECT_STRICT_ALIASING
            uint32_t lsb = *(packed_wts + j_offset + STRIDE_IN*i_offset*2);
            uint32_t msb = *(packed_wts + j_offset + STRIDE_IN*(i_offset*2 + 1));
        #else
            WIDE_S sb;
            WIDE_S aux;

            sb.sb.lsb = *(packed_wts + j_offset + STRIDE_IN*i_offset*2);
            sb.sb.msb = *(packed_wts + j_offset + STRIDE_IN*(i_offset*2 + 1));
            aux.aux.zero = zero;
            aux.aux.scale = scale;
            aux.aux.msk = ref_msk;

            // TODO: can offload the mask shuffle into smem?
        #endif

        // n.b. that we're storing the transpose so this will look kinda silly
        __half* wt_out = unpacked_wts
                         + (j_offset - threadIdx.x)*STRIDE_OUT
                         + (i_offset*32 + threadIdx.x);

        const uint32_t T_MSK = 1 << threadIdx.x;
        const uint32_t T_MSK_A = T_MSK - 1;

        #pragma unroll
        for (auto wi = 0; wi < WARP_SIZE; ++wi) {
            #ifdef RESPECT_STRICT_ALIASING
                // grab appropriate vars; we have to go about it in this weird way
                // because cuda doesn't like warp shuffles with inconsistent args
                const uint32_t umsb = __shfl_sync(FULL_MASK, msb, wi);
                uint32_t ulsb = __shfl_sync(FULL_MASK, lsb, wi);
                const __half wscale = __shfl_sync(FULL_MASK, scale, wi);
                const __half wzero = __shfl_sync(FULL_MASK, zero, wi);
                uint32_t msk = __shfl_sync(FULL_MASK, ref_msk, wi);

                ulsb = threadIdx.x < 8 ? ulsb : umsb;
                ulsb >>= (4 * (threadIdx.x % 8));
                __half h = dequantize<qm>(ulsb, wscale, wzero);
            #else
                uint64_t p = __shfl_sync(FULL_MASK, sb.p, wi);
                const WIDE_S usb = *(WIDE_S*)&p;
                p = __shfl_sync(FULL_MASK, aux.p, wi);
                const WIDE_S uaux = *(WIDE_S*)&p;

                __half h = dequantize<qm>(
                    (threadIdx.x < 8 ? usb.sb.lsb : usb.sb.msb) >> (4 * (threadIdx.x % 8)),
                    uaux.aux.scale, uaux.aux.zero);

                uint32_t msk = uaux.aux.msk;
            #endif

            auto seek_d = __popc(msk & T_MSK_A);

            __half w = __shfl_sync(FULL_MASK, h, seek_d);
            w = (msk & T_MSK) ? w : ZERO;

            *wt_out = w;
            wt_out += STRIDE_OUT;
        }
        __syncthreads();
    }
};


template<bool is_sparse, size_t STRIDE_IN, size_t STRIDE_OUT, Quantization qm>
struct smem_unpack_weights {
    __device__ inline static void load(UNPACK_WEIGHTS_ARGS) {
        smem_unpack_weights_inner<is_sparse, STRIDE_IN, STRIDE_OUT, qm>::load(packed_wts, unpacked_wts, i_offset, j_offset, scale, zero);
    }
};
template<bool is_sparse, size_t STRIDE_IN, size_t STRIDE_OUT>
struct smem_unpack_weights<is_sparse, STRIDE_IN, STRIDE_OUT, Quantization::DYNAMIC_EXPONENT_SYM> {
    __device__ inline static void load(UNPACK_WEIGHTS_ARGS) {
        // TODO: add a compile-flagged ballot check to make sure warps are synced

        // WARNING: this assumes that the entire warp branches the same way!
        if (__hge(scale, __float2half(0.0f))) {
            smem_unpack_weights_inner<is_sparse, STRIDE_IN, STRIDE_OUT, Quantization::LINEAR>::load(
                packed_wts, unpacked_wts, i_offset, j_offset, scale, zero);
        } else {
            smem_unpack_weights_inner<is_sparse, STRIDE_IN, STRIDE_OUT, Quantization::EXPONENT_SYM>::load(
                packed_wts, unpacked_wts, i_offset, j_offset, __hneg(scale), zero);
        }
    }
};


#undef UNPACK_WEIGHTS_ARGS