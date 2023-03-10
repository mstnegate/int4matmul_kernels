#include <torch/all.h>
#include <torch/python.h>
#include <ATen/ATen.h>

void matmul_int4(
    torch::Tensor outs,
    torch::Tensor matrix,
    torch::Tensor multiplier,
    torch::Tensor scales,
    torch::Tensor zeros,
    c10::optional<torch::Tensor> sparse_mask
);
void sparse_int4_pack(
    torch::Tensor wt_outs,
    torch::Tensor msk_outs,
    torch::Tensor wt_in,
    torch::Tensor msk_in
);

void quant_int4_linear_mult(
    torch::Tensor outs,
    torch::Tensor matrix,
    torch::Tensor multiplier,
    torch::Tensor scales,
    torch::Tensor zeros,
    c10::optional<torch::Tensor> sparse_mask
) {
    matmul_int4(outs, matrix, multiplier, scales, zeros, sparse_mask);
}

void weight_matrix_packing(
    torch::Tensor wt_outs,
    torch::Tensor msk_outs,
    torch::Tensor wt_in,
    torch::Tensor msk_in
) {
    sparse_int4_pack(wt_outs, msk_outs, wt_in, msk_in);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant_int4_linear_mult", &quant_int4_linear_mult, "int4_fp16_mult");
    m.def("weight_matrix_packing", &weight_matrix_packing, "int4_sparse_pack");
}