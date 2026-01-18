#include <torch/extension.h>

torch::Tensor fused_wave_cuda(torch::Tensor x, double alpha);

torch::Tensor fused_wave(torch::Tensor x, double alpha) {
    TORCH_CHECK(x.is_cuda(), "fused_wave only supports CUDA tensors. Use CPU fallback for CPU tensors.");
    return fused_wave_cuda(x, alpha);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_wave", &fused_wave, "Fused wave op (CUDA)");
}
