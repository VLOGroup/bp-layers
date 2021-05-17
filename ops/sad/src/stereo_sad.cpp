#include <torch/extension.h>
#include "stereo_sad_kernel.cuh"

// C++ interface
// AT_ASSERTM in pytorch 1.0
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

at::Tensor stereo_sad_forward(at::Tensor f0, at::Tensor f1, int min_disp, int max_disp, float step)
{
  CHECK_INPUT(f0)
  CHECK_INPUT(f1)
  return cuda::stereo_sad_forward(f0, f1, min_disp, max_disp, step);
}       

std::vector<at::Tensor> stereo_sad_backward(at::Tensor f0, at::Tensor f1, int min_disp, int max_disp, 
                               at::Tensor in_grad)
{
  CHECK_INPUT(f0)
  CHECK_INPUT(f1)
  CHECK_INPUT(in_grad)
  return cuda::stereo_sad_backward(f0, f1, min_disp, max_disp, in_grad);
}       

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward", &stereo_sad_forward, "SAD Matching forward (CUDA)");
  m.def("backward", &stereo_sad_backward, "SAD Matching backward (CUDA)");
}