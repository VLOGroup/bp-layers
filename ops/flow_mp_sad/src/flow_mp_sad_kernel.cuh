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

namespace cuda
{
  std::vector<at::Tensor> flow_mp_sad_forward(at::Tensor f0, at::Tensor f1, int sws, int offset_u, int offset_v, 
                                              int blockIdx_u, int blockIdx_v);
  std::vector<at::Tensor> flow_mp_sad_backward(at::Tensor f0, at::Tensor f1, int sws, 
                                               at::Tensor in_grad_u, at::Tensor in_grad_v,
                                               at::Tensor u_star, at::Tensor v_star);
}