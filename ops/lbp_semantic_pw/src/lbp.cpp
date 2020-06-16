// This file is part of bp-layers.
//
// Copyright (C) 2020 Patrick Knöbelreiter <knoebelreiter at icg dot tugraz dot at>
// Christian Sormann <christian dot sormann at icg dot tugraz dot at>
// Institute for Computer Graphics and Vision, Graz University of Technology
// https://www.tugraz.at/institute/icg/teams/team-pock/
//
// bp-layers is free software: you can redistribute it and/or modify it under the
// terms of the GNU Affero General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// bp-layers is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.


#include <torch/extension.h>
#include <iostream>

#include "lbp_min_sum_kernel.cuh"

// C++ interface
// AT_ASSERTM in pytorch 1.0
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
                                                at::Tensor jump, 
                                                at::Tensor edge,
                                                at::Tensor in_grad,
                                                at::Tensor messages,
                                                at::Tensor messages_argmin,
                                                at::Tensor message_scale)
{
  CHECK_INPUT(cost)
  CHECK_INPUT(jump)
  CHECK_INPUT(edge)
  CHECK_INPUT(in_grad) 
  CHECK_INPUT(messages)
  CHECK_INPUT(messages_argmin)
  CHECK_INPUT(message_scale)

  return cuda::lbp_backward_min_sum(cost, jump, edge, in_grad, messages, messages_argmin, message_scale);
}

// ================================================================================================
// Pytorch Interfaces
// ================================================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward_minsum", &lbp_forward_min_sum, "LBP forward (CUDA)");
  m.def("backward_minsum", &lbp_backward_min_sum, "LBP backward (CUDA)");
}

