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
                                            at::Tensor edge,
                                            at::Tensor in_grad,
                                            at::Tensor messages,
                                            at::Tensor messages_argmin,
                                            at::Tensor message_scale);

}