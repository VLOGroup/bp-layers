#pragma once
#include <ATen/ATen.h>

namespace cuda
{
  at::Tensor stereo_sad_forward(at::Tensor f0, at::Tensor f1, int min_disp, int max_disp, float step);
  std::vector<at::Tensor> stereo_sad_backward(at::Tensor f0, at::Tensor f1, int min_disp, 
                                              int max_disp, at::Tensor in_grad);
}