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
#include "flow_mp_sad_kernel.cuh"
#include "tensor.h"
#include "error_util.h"

// ============================================================================
// CUDA KERNELS
// ============================================================================
__global__ void flow_mp_sad_cuda_forward_kernel(
    KernelData f0,
    KernelData f1,
    int sws,
    KernelData cv_u,
    KernelData cv_v,
    KernelData u_star,
    KernelData v_star,
    int offset_u,
    int offset_v,
    int blockIdx_u, // necessary for argmin computation
    int blockIdx_v  // same here
    )
{
  // parallelize over u, loop over v 
  //const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  // const int u_idx = blockIdx.z * blockDim.z + threadIdx.z;

  const int u_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int x = blockIdx.z * blockDim.z + threadIdx.z;

    // shared memory for search-window matching costs
  extern __shared__ float sdata[];

  // global defines
  unsigned short K = cv_u.size3;
  short sws_half = sws / 2;
  short u = u_idx - sws_half;

  //short sm_offset = blockDim.x * K * K * threadIdx.y + K * K * threadIdx.x;
  short sm_offset = blockDim.z * K * K * threadIdx.y + K * K * threadIdx.z;

  // check inside image for reference pixel
  int n = 0;
  if(x >= f0.size3 || y >= f0.size2 || u_idx >= K)
    return;

  // initialize all sws with constant value (initialize all v displacements for given u_idx)
  for(short v_idx = 0; v_idx < K; ++v_idx)
  {  
    sdata[sm_offset + K * v_idx + u_idx] = 40.0;
  }
  __syncthreads();
  
  // skip outside pixels  
  // if(x + u < 0 || x + u >= f0.size3)
  //   return;

  // I cannot return outside pixels directly, because I need all the treads for the min-computation
  // later!!
  if(x + offset_u + u >= 0 && x + offset_u + u < f0.size3) // check match inside
  {
    for(short v = -sws_half; v <= sws_half; ++v)
    {
      short v_idx = v + sws_half;

      // skip outside pixels (match-pixel)
      if(y + offset_v + v < 0 || y + offset_v + v >= f0.size2)
          continue;

      float sad = 0.0f;
      for(int c = 0; c < f0.size1; ++c)
      {
          sad += fabs(f0(n, c, y, x) - f1(n, c, y + offset_v + v, x + offset_u + u));
      }

      // save result to shared mem
      sdata[sm_offset + K * v_idx + u_idx] = sad;
      //cv_all(n, y, x, v_idx, u_idx) = sad;
    }
  }
  __syncthreads(); // all u-threads must be ready here!

  // compute min-projection in shared memory
  // Note: u_idx is parallelized within the kernel!
  float min_v = 9999999.0;
  short argmin_v = 0;
  for(unsigned short v_idx = 0; v_idx < K; ++v_idx)
  {
    if(sdata[sm_offset + K * v_idx + u_idx] < min_v)
    {
      min_v = sdata[sm_offset + K * v_idx + u_idx];
      argmin_v = v_idx;
    }  
  }
  
  // update min only if the current block has a better min
  // if(min_v < cv_u(n, y, x, u_idx)) // for inplace variant which I do not have yet
  //{
    cv_u(n, y, x, u_idx) = min_v;
    u_star(n, y, x, u_idx) = argmin_v + blockIdx_v * sws; // sws = K - 1 => default overlap
  //}

  // compute min-projection in shared memory
  // here I swap rules and use the u_idx as v_idx for easier parallelization
  float min_u = 9999999.0;
  short v_idx = u_idx;
  short argmin_u = 0;
  for(unsigned short u_idx = 0; u_idx < K; ++u_idx)
  {  
    if(sdata[sm_offset + K * v_idx + u_idx] < min_u)
    {
      min_u = sdata[sm_offset + K * v_idx + u_idx];
      argmin_u = u_idx;
    }
  }
  
  // update min only if the current block has a better min
  //if(min_u < cv_v(n, y, x, v_idx)) // for inplace variant which I do not have yet
  //{
    cv_v(n, y, x, v_idx) = min_u;
    v_star(n, y, x, v_idx) = argmin_u + blockIdx_u * sws; // sws = K - 1 => default overlap
  //}
}

__global__ void flow_mp_sad_cuda_backward_kernel(
    KernelData f0,
    KernelData f1,
    int sws,
    KernelData in_grad_u,
    KernelData in_grad_v,
    KernelData u_star, 
    KernelData v_star,
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

  int sws_half = sws / 2;

  float grad_f0 = 0.0f;
  float grad_f1 = 0.0f;
  for(short u = -sws_half; u <= sws_half; ++u)
  {
    short u_idx = u + sws_half;
    short v_idx = u_star(n, y, x, u_idx);
    short v = v_idx - sws_half;

    // skip outside pixels  
    if(x + u >= 0 && x + u < f0.size3 && y + v >= 0 && y + v < f0.size2)
    {
      float diff = f0(n, c, y, x) - f1(n, c, y + v, x + u);
      if(fabsf(diff) > eps) // gradient is zero if diff is zero!
     {    
       float update = diff / fabsf(diff) * in_grad_u(n, y, x, u_idx);
       // local update for df0
       grad_f0 += update;

       // global update for df1 (multiple vars can point to one address!)
       atomicAdd(&df1(n, c, y + v, x + u), -update);
     }
    }

  }

  for(short v = -sws_half; v <= sws_half; ++v)
  {
    short v_idx = v + sws_half;
    short u_idx = v_star(n, y, x, v_idx);
    short u = u_idx - sws_half;

    // copied from above, only change is that here in_grad_v is used
    if(x + u >= 0 && x + u < f0.size3 && y + v >= 0 && y + v < f0.size2)
    {
      float diff = f0(n, c, y, x) - f1(n, c, y + v, x + u);
      if(fabsf(diff) > eps) // gradient is zero if diff is zero!
      {    
        float update = diff / fabsf(diff) * in_grad_v(n, y, x, v_idx);
        // local update for df0
        grad_f0 += update;

        // global update for df1 (multiple vars can point to one address!)
        atomicAdd(&df1(n, c, y + v, x + u), -update);
      }
    }
  }

  df0(n, c, y, x) = grad_f0;

}


// ============================================================================
// CPP KERNEL CALLS
// ============================================================================
namespace cuda
{
  std::vector<at::Tensor> flow_mp_sad_forward(at::Tensor f0, at::Tensor f1, int sws, int offset_u, int offset_v, 
                                              int blockIdx_u, int blockIdx_v)
  {
    int N = f0.size(0);
    int C = f0.size(1);
    int H = f0.size(2);
    int W = f0.size(3);
    int K = sws + 1;

    auto cv_u = at::ones({N, H, W, K}, f0.options()) * 40;
    auto cv_v = at::ones({N, H, W, K}, f0.options()) * 40;

    auto u_star = at::zeros({N, H, W, K}, f0.options());
    auto v_star = at::zeros({N, H, W, K}, f0.options());

    //auto cv_all = at::ones({N, H, W, K, K}, f0.options()) * 40;

    if(K > 128)
      std::cout << "Error: Maximal search window size is " << K << " which is larger than max allowed 128!!" << std::endl;

    // parallelise over H x W x K
    // all K need to be in one block in order to have access to the same shared memory!
    // K needs to be the first, because last idx must be < 64.
    const dim3 blockSize(K, 1, 1);
    const dim3 numBlocks(std::ceil(K / static_cast<float>(blockSize.x)),
                         std::ceil(H / static_cast<float>(blockSize.y)),
                         std::ceil(W / static_cast<float>(blockSize.z)));

    const int threadsPerBlock = blockSize.x * blockSize.y * blockSize.z;

    // std::cout << "N=" << N << " C=" << C << " H=" << H << " W=" << W << " K=" << K << std::endl;
    // std::cout << "threadsPerBlock=" << threadsPerBlock << std::endl;
    // std::cout << "numBlocks.x=" << numBlocks.x << " .y=" << numBlocks.y << " .z=" << numBlocks.z << std::endl;
    // std::cout << "mem-use=" << threadsPerBlock*K*sizeof(float) << "bytes" << std::endl;

      //CudaTimer cut;
      //cut.start();
      flow_mp_sad_cuda_forward_kernel<<<numBlocks, blockSize, threadsPerBlock*K*sizeof(float)>>>(f0, f1, sws, cv_u, cv_v, u_star, v_star, offset_u, offset_v, blockIdx_u, blockIdx_v);
      cudaSafeCall(cudaGetLastError());
      // cudaDeviceSynchronize();
      //std::cout << "SAD forward time " << cut.elapsed() << std::endl;
    std::vector<at::Tensor> res;
    //cost_vols.push_back(cv_all);
    res.push_back(cv_u);
    res.push_back(cv_v);
    res.push_back(u_star);
    res.push_back(v_star);
    return res;
  }   

  std::vector<at::Tensor> flow_mp_sad_backward(at::Tensor f0, at::Tensor f1, 
                                              int sws, at::Tensor in_grad_u, at::Tensor in_grad_v,
                                              at::Tensor u_star, at::Tensor v_star)
  {
    int N = f0.size(0);
    int C = f0.size(1);
    int H = f0.size(2);
    int W = f0.size(3);
    int K = sws + 1;

    auto df0 = at::zeros_like(f0);
    auto df1 = at::zeros_like(f1);

    // parallelise over H x W x D
    const dim3 blockSize(8, 8, 4);
    const dim3 numBlocks(std::ceil(W / static_cast<float>(blockSize.x)),
                         std::ceil(H / static_cast<float>(blockSize.y)),
                         std::ceil(C / static_cast<float>(blockSize.z)));

    //CudaTimer cut;
    //cut.start();
    flow_mp_sad_cuda_backward_kernel<<<numBlocks, blockSize>>>(f0, f1, sws, in_grad_u, in_grad_v, 
                                                               u_star, v_star, df0, df1);
    cudaSafeCall(cudaGetLastError());
    // cudaDeviceSynchronize();

    //std::cout << "SAD backward time " << cut.elapsed() << std::endl;

    std::vector<at::Tensor> gradients;
    gradients.push_back(df0);
    gradients.push_back(df1);

    return gradients;
  }            
}