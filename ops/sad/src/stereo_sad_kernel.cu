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

#include <cuda.h>
#include <cuda_runtime.h>
#include "stereo_sad_kernel.cuh"
#include "tensor.h"
#include "error_util.h"

// get y for position x
__device__ float dLinearInterpolation1D(float x, float x0, float x1, float y0, float y1)
{
	return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
}

// ============================================================================
// CUDA KERNELS
// ============================================================================
__global__ void stereo_sad_cuda_forward_kernel(
    KernelData f0,
    KernelData f1,
    int min_disp,
    int max_disp,
    float step,
    KernelData output
    )
{
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int d_idx = blockIdx.z * blockDim.z + threadIdx.z;

  // check inside image
  int n = 0;
  if(x >= f0.size3 || y >= f0.size2 || d_idx >= output.size3)
    return;

  // 
   int d = (d_idx * step) + min_disp;

  // skip outside pixels  
  if(x - d < 0 || x - d >= f0.size3)
    return;
 
  float sad = 0.0f;
  for(int c = 0; c < f0.size1; ++c)
  {
    float f1_c = 0.0;
    if(step == 1.0)
    {
      f1_c = f1(n, c, y, x - d);
    }
    else
    {
      int floor_x = (int) floorf(x - d);
      int ceil_x = floor_x + 1;
      float x_pos = x - d;
      f1_c = dLinearInterpolation1D(x_pos, floor_x, ceil_x, f1(n, c, y, floor_x), f1(n, c, y, ceil_x));
    }
    
    sad += fabs(f0(n, c, y, x) - f1_c);
  }

  // write result back to global memory
  output(n, y, x, d_idx) = sad;
}

__global__ void stereo_sad_cuda_backward_kernel(
    KernelData f0,
    KernelData f1,
    int min_disp,
    int max_disp,
    KernelData in_grad,
    KernelData df0,
    KernelData df1
    )
{
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int c = blockIdx.z * blockDim.z + threadIdx.z;

  float eps = 1e-15;

  // check inside image
  int n = 0;
  if(x >= f0.size3 || y >= f0.size2 || c >= f0.size1)
    return;

  float grad_f0 = 0.0f;
  float grad_f1 = 0.0f;
  for(int d = min_disp; d <= max_disp; ++d)
  {
    int idx = d - min_disp;
    // skip outside pixels  
    if(x - d >= 0 && x - d < f0.size3)
    {
      float diff = f0(n, c, y, x) - f1(n, c, y, x - d);
      if(fabsf(diff) > eps) // gradient is zero if diff is zero!
        grad_f0 += (diff / fabsf(diff)) * in_grad(n, y, x, idx);
    }

    if(x + d >= 0 && x + d < f0.size3)
    { 
      float diff1 = f0(n, c, y, x + d) - f1(n, c, y, x);
      if(fabsf(diff1) > eps)
        grad_f1 -= (diff1 / fabsf(diff1)) * in_grad(n, y, x + d, idx);
    }
  }

  df0(n, c, y, x) = grad_f0;
  df1(n, c, y, x) = grad_f1;
}


// ============================================================================
// CPP KERNEL CALLS
// ============================================================================
namespace cuda
{
  at::Tensor stereo_sad_forward(at::Tensor f0, at::Tensor f1, int min_disp, int max_disp, float step)
  {
    int N = f0.size(0);
    int C = f0.size(1);
    int H = f0.size(2);
    int W = f0.size(3);
    int D = (max_disp - min_disp + 1) / step;


    auto cost_vol = at::ones({N, H, W, D}, f0.options()) * 40;

    // parallelise over H x W x D
    const dim3 blockSize(8, 8, 4);
    const dim3 numBlocks(std::ceil(W / static_cast<float>(blockSize.x)),
                         std::ceil(H / static_cast<float>(blockSize.y)),
                         std::ceil(D / static_cast<float>(blockSize.z)));

      stereo_sad_cuda_forward_kernel<<<numBlocks, blockSize>>>(f0, f1, min_disp, max_disp, step, cost_vol);
      cudaSafeCall(cudaGetLastError());
    return cost_vol;
  }   

  std::vector<at::Tensor> stereo_sad_backward(at::Tensor f0, at::Tensor f1, 
                                              int min_disp, int max_disp,
                                              at::Tensor in_grad)
  {
    int N = f0.size(0);
    int C = f0.size(1);
    int H = f0.size(2);
    int W = f0.size(3);
    int D = max_disp - min_disp + 1;

    auto df0 = at::zeros_like(f0);
    auto df1 = at::zeros_like(f1);

    // parallelise over H x W x D
    const dim3 blockSize(8, 8, 4);
    const dim3 numBlocks(std::ceil(W / static_cast<float>(blockSize.x)),
                         std::ceil(H / static_cast<float>(blockSize.y)),
                         std::ceil(D / static_cast<float>(blockSize.z)));

    stereo_sad_cuda_backward_kernel<<<numBlocks, blockSize>>>(f0, f1, min_disp, max_disp, in_grad, 
                                                              df0, df1);
    cudaSafeCall(cudaGetLastError());

    std::vector<at::Tensor> gradients;
    gradients.push_back(df0);
    gradients.push_back(df1);

    return gradients;
  }            
}