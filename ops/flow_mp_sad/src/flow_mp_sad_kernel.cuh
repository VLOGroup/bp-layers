#pragma once
#include <ATen/ATen.h>

namespace cuda
{
  std::vector<at::Tensor> flow_mp_sad_forward(at::Tensor f0, at::Tensor f1, int sws, int offset_u, int offset_v, 
                                              int blockIdx_u, int blockIdx_v);
  std::vector<at::Tensor> flow_mp_sad_backward(at::Tensor f0, at::Tensor f1, int sws, 
                                               at::Tensor in_grad_u, at::Tensor in_grad_v,
                                               at::Tensor u_star, at::Tensor v_star);
}