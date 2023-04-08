#include <torch/all.h>
#include <torch/python.h>
#include <ATen/ATen.h>

#include <stdexcept>

#include "matmul.cuh"

#define REGISTER_WIDTH(N) \
    void matmul_int ## N(TENSOR_MULT_ARGS); \
    void matvec_int ## N(TENSOR_MULT_ARGS); \
    void spec_mm_int ## N(TENSOR_MULT_ARGS) { \
        if (multiplier.size(1) == 1) { \
            matvec_int ## N(TENSOR_MULT_PASS); \
        } else { \
            matmul_int ## N(TENSOR_MULT_PASS); \
        } \
    }

REGISTER_WIDTH(4)
REGISTER_WIDTH(3)
REGISTER_WIDTH(2)

#undef REGISTER_WIDTH

void quant_matmul(uint32_t bits, TENSOR_MULT_ARGS) {
    if (multiplier.size(1) == 1) {
        if (bits == 4) {
            matvec_int4(TENSOR_MULT_PASS);
        } else if (bits == 3) {
            matvec_int3(TENSOR_MULT_PASS);
        } else if (bits == 2) {
            matvec_int2(TENSOR_MULT_PASS);
        } else {
            throw std::invalid_argument("Unsupported bit width for matvec");
        }
    } else {
        if (bits == 4) {
            matmul_int4(TENSOR_MULT_PASS);
        } else if (bits == 3) {
            matmul_int3(TENSOR_MULT_PASS);
        } else if (bits == 2) {
            matmul_int2(TENSOR_MULT_PASS);
        } else {
            throw std::invalid_argument("Unsupported bit width for matmul");
        }
    }
}

void sparse_int4_pack(
    torch::Tensor wt_outs,
    torch::Tensor msk_outs,
    torch::Tensor wt_in,
    torch::Tensor msk_in
);
void weight_matrix_packing(
    torch::Tensor wt_outs,
    torch::Tensor msk_outs,
    torch::Tensor wt_in,
    torch::Tensor msk_in
) {
    sparse_int4_pack(wt_outs, msk_outs, wt_in, msk_in);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("weight_matrix_packing", &weight_matrix_packing, "int4_sparse_pack");

    m.def("quant_matmul", &quant_matmul, "intn_fp16_mult");

    // compat functions
    m.def("quant_int4_linear_mult", &spec_mm_int4, "int4_fp16_mult");
    m.def("quant_int4_linear_mult_mtx", &matmul_int4, "int4_fp16_mult_mtx");
    m.def("quant_int4_linear_mult_vec", &matvec_int4, "int4_fp16_mult_vec");

    m.def("quant_int3_linear_mult", &spec_mm_int3, "int3_fp16_mult");
    m.def("quant_int3_linear_mult_mtx", &matmul_int3, "int3_fp16_mult_mtx");
    m.def("quant_int3_linear_mult_vec", &matvec_int3, "int3_fp16_mult_vec");

    m.def("quant_int2_linear_mult", &spec_mm_int2, "int2_fp16_mult");
    m.def("quant_int2_linear_mult_mtx", &matmul_int2, "int2_fp16_mult_mtx");
    m.def("quant_int2_linear_mult_vec", &matvec_int2, "int2_fp16_mult_vec");
}
