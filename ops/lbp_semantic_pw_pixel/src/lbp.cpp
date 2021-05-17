#include <torch/extension.h>
#include <iostream>

#include "lbp_min_sum_kernel.cuh"

// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);            


// ================================================================================================
// MIN-SUM LBP
// ================================================================================================
std::vector<at::Tensor> lbp_forward_min_sum(at::Tensor cost, 
                       at::Tensor jump, 
                       at::Tensor edge,
                       at::Tensor messages, unsigned short delta)
{
  CHECK_INPUT(cost)
  CHECK_INPUT(jump)
  CHECK_INPUT(edge)
  CHECK_INPUT(messages)

  return cuda::lbp_forward_min_sum(cost, jump, edge, messages, delta);
}                                    

std::vector<at::Tensor> lbp_backward_min_sum(at::Tensor cost, 
                                                at::Tensor edge,
                                                at::Tensor in_grad,
                                                at::Tensor messages,
                                                at::Tensor messages_argmin,
                                                at::Tensor message_scale)
{
  CHECK_INPUT(cost)
  //CHECK_INPUT(jump)
  CHECK_INPUT(edge)
  CHECK_INPUT(in_grad) 
  CHECK_INPUT(messages)
  CHECK_INPUT(messages_argmin)
  CHECK_INPUT(message_scale)

  return cuda::lbp_backward_min_sum(cost, edge, in_grad, messages, messages_argmin, message_scale);
}

// ================================================================================================
// Pytorch Interfaces
// ================================================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward_minsum", &lbp_forward_min_sum, "LBP forward (CUDA)");
  m.def("backward_minsum", &lbp_backward_min_sum, "LBP backward (CUDA)");
}

