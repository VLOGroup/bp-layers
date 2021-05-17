#pragma once
#include <ATen/ATen.h>
#include <vector>

namespace cuda
{
std::vector<at::Tensor> lbp_forward_min_sum(at::Tensor cost, 
                       at::Tensor jump, 
                       at::Tensor edge,
                       at::Tensor messages, unsigned short delta);

std::vector<at::Tensor> lbp_backward_min_sum(at::Tensor cost, 
                                            at::Tensor jump, 
                                            at::Tensor edge,
                                            at::Tensor in_grad,
                                            at::Tensor messages,
                                            at::Tensor messages_argmin,
                                            at::Tensor message_scale);

}