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


#include "../../include/error_util.h"
#include "lbp_min_sum_kernel.cuh"
#include "util.cuh"


// ============================================================================
// CUDA KERNELS
// ============================================================================
__global__ void lbp_cuda_forward_kernel_reduction_min_sum(
    KernelData cost,
    KernelData5 jump, 
    KernelData edges,
    KernelData5 messages,
    KernelData5 messages_argmin,
    KernelData message_scale,
    const unsigned short x_in,
    const unsigned short direction, 
    int shared_mem_offset, unsigned short delta)
{
  unsigned short y = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned short c = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned short x = 0;
  if(direction == UP || direction == DOWN)
  {
    x = y;
    y = x_in;
  }
  else
  {
    x = x_in;
    // y = y;
  }
  

  // shared memory h
  extern __shared__ float sdata[];

  // message size is N x 4 x H x W x C
  // cost size is N x H x W x C
  // edges: N x 1 x H x W
  // jumps: 1 x 1 x H x W
  const short N = cost.size0;
  const short H = cost.size1;
  const short W = cost.size2;
  const short C = cost.size3;

  const unsigned int tid = threadIdx.y + blockDim.y * threadIdx.x;

  const float max_float = 1e15;

  // check inside image
  if(c >= C || x >= W || y >= H)
  {
    // write large number that will never win
    sdata[tid] = max_float;
    return;
  }

  unsigned int n = 0;

  float L2 = jump(direction, y, x, 0, jump.size4 - 1) ;

  unsigned short start = max(c - delta + 1, 0); 
  unsigned short stop = min(c + delta - 1, C - 1);

  float edgeWeight = edges(n, direction, y, x);
  
  // write to shared memory 
  // compute message for every label    
  sdata[tid] = cost(n, y, x, c);

  // add costs from all neighbors
  if(direction != RIGHT) { sdata[tid] += messages(n, RIGHT, y, x, c); }
  if(direction != LEFT) { sdata[tid] += messages(n, LEFT, y, x, c); }
  if(direction != UP) { sdata[tid] += messages(n, UP, y, x, c); }
  if(direction != DOWN) { sdata[tid] += messages(n, DOWN, y, x, c); }
  float h = sdata[tid];
  __syncthreads();

  // save h in shared mem
  sdata[tid] = h;
  sdata[tid + shared_mem_offset] = static_cast<float>(c);
  __syncthreads();

  // if delta is larger or equal than this threshold use old version as it is a little faster
  int old_version_threshold = C;

  float msg = 0.0;
  int msg_argmin = 0;

  // if there is no truncation use old version
  if(delta >= old_version_threshold)
  {
    //OLD VERSION /////////////////////
    sdata[tid] = h;
    __syncthreads();

    msg = max_float; //minVal + jump(0, 0, 0, jump.size3 - 1) * edgeWeight;
    msg_argmin = 0;
    for(unsigned short label = 0; label < C; ++label)
    {
      // compute min in local var to avoid global mem writes
      float new_msg = sdata[label + blockDim.y * threadIdx.x] + jump(direction, y, x, label, c) * edgeWeight;
      msg = fminf(msg, new_msg);
  
      if(msg == new_msg)
      {
        msg_argmin = label;
      }
    }
    __syncthreads();
    /////////////////
  }
  else
  {
    //TRUNC SPEED UP VERSION ///////////////////////////////////
    for(unsigned int s=blockDim.y / 2; s > 0; s>>=1)
    {
      if(tid - (threadIdx.x * blockDim.y) < s && tid + s < (threadIdx.x * blockDim.y) + C)
      {
        //min parallel reduction
        float min_val = sdata[tid];
        float min_label = sdata[tid + shared_mem_offset];
        if(sdata[tid + s] <= sdata[tid])
        {
          min_val = sdata[tid + s];
          min_label = sdata[shared_mem_offset + tid + s];
        }
        //min val parallel reduction
        sdata[tid] = min_val;
        //argmin prallel reduction
        sdata[shared_mem_offset + tid] = min_label;
      }
      __syncthreads();
    }

    float min_h = sdata[threadIdx.x * blockDim.y];
    int argmin_h = sdata[shared_mem_offset + threadIdx.x * blockDim.y];
    __syncthreads();

    msg = min_h + jump(direction, y, x, 0, jump.size4 - 1) * edgeWeight;
    msg_argmin = static_cast<int>(argmin_h); 

    sdata[tid] = h;
    __syncthreads();

    for(unsigned short label = start; label < stop + 1; ++label)
    {
      // compute min in local var to avoid global mem writes
      float new_msg = sdata[label + blockDim.y * threadIdx.x] + jump(direction, y, x, label, c) * edgeWeight;
      if(new_msg <= msg)
      {
        msg = new_msg;
        msg_argmin = label;
      }
    }
    __syncthreads();
    /////////////////////////////
  }

  // if(x == 2 && y == 0 && direction == DOWN)
  // {
  //   printf("argmin : %i argmin h  %i min_val %f min_h %f h: %f \n", msg_argmin, argmin_h, msg, min_h, h);
  // }

  // compute normalization with 2nd reduction
  sdata[tid] = (float)exp((double)msg);
  __syncthreads();

  for(unsigned int s=blockDim.y / 2; s > 0; s>>=1)
  {
    if(tid - (threadIdx.x * blockDim.y) < s && tid + s < (threadIdx.x * blockDim.y) + C)
    {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // normalize message
  double sum_exp = max((double)sdata[blockDim.y * threadIdx.x], 1e-45);
  float logSumExp = (float)log(sum_exp); 

  // if(sum_exp < 1e-10)
  // {
  //   printf("sum exp zero: %f , logsumexp: %f msg: %f \n", sum_exp, msg);
  // }

  //float logSumExp = 0.0;
  if(direction == RIGHT) 
  { 
    messages(n, LEFT, y, x+1, c) = msg - logSumExp; 
    messages_argmin(n, LEFT, y, x+1, c) = msg_argmin;
    message_scale(n, LEFT, y, x+1) = sum_exp;
  }
  if(direction == LEFT) 
  { 
    messages(n, RIGHT, y, x-1, c) = msg - logSumExp; 
    messages_argmin(n, RIGHT, y, x-1, c) = msg_argmin;
    message_scale(n, RIGHT, y, x-1) = sum_exp;
  }
  if(direction == UP) 
  { 
    messages(n, DOWN, y-1, x, c) = msg - logSumExp; 
    messages_argmin(n, DOWN, y-1, x, c) = msg_argmin;
    message_scale(n, DOWN, y-1, x) = sum_exp;
  }
  if(direction == DOWN) 
  { 
    messages(n, UP, y+1, x, c) = msg - logSumExp;
    messages_argmin(n, UP, y+1, x, c) = msg_argmin;
    message_scale(n, UP, y+1, x) = sum_exp;
   }
}

__global__ void lbp_cuda_backward_kernel_reduction_min_sum(
  KernelData cost,
  KernelData edges,
  KernelData5 messages,
  KernelData5 messages_argmin,
  KernelData message_scale,
  KernelData5 in_grad,
  KernelData gradient_unary,
  KernelData5 gradient_pairwise,
  KernelData gradient_edge,
  KernelData gradient_accumulation,
  KernelData gradient_accumulation_tmp,
  KernelData5 saved_prev_grad_msg,
  const unsigned short x_in,
  const unsigned short direction,
  bool compute_cross,
  const unsigned int n)
{

    //initialize utility variables
    unsigned short y = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned short c = blockIdx.y * blockDim.y + threadIdx.y;
    
    //unsigned int n = 0;
    unsigned int x;

    if(direction == UP || direction == DOWN)
    {
      x = y;
      y = x_in;
    }
    else
    {
      x = x_in;
    }

    // shared memory h
    extern __shared__ float sdata[];

    // message size is N x 4 x H x W x C
    // cost size is N x H x W x C
    // edges: N x 1 x H x W
    // jumps: 1 x 1 x H x W
    const short N = cost.size0;
    const short H = cost.size1;
    const short W = cost.size2;
    const short C = cost.size3;

    const unsigned int tid = threadIdx.y + blockDim.y * threadIdx.x;

    const float max_float = 1e15;

    // check inside image
    if(c >= C || x >= W || y >= H)
    {
      // write large number that will never win
      sdata[tid] = max_float;
      return;
    }


    //calc backward message

    short prev_row_shift = 0;
    short prev_col_shift = 0;
    if(direction == LEFT)
    {
      prev_row_shift = 0;
      prev_col_shift = 1;
    }
    if(direction == RIGHT)
    {
      prev_row_shift = 0;
      prev_col_shift = -1;
    }
    if(direction == DOWN)
    {
      prev_row_shift = -1;
      prev_col_shift = 0;
    }
    if(direction == UP)
    {
      prev_row_shift = 1;
      prev_col_shift = 0;
    }

    int grad_xy_idx = 0;
    if(direction == UP)
    {
      grad_xy_idx = DOWN;
    }
    if(direction == DOWN)
    {
      grad_xy_idx = UP;
    }
    if(direction == LEFT)
    {
      grad_xy_idx = RIGHT;
    }
    if(direction == RIGHT)
    {
      grad_xy_idx = LEFT;
    }

    float edgeWeight = edges(n, grad_xy_idx, y, x);

    int HOR_IDX = 0;
    int UP_IDX = 1;
    int DOWN_IDX = 2;
    
    ///////////////////////in_grad normalization ////////////////////////////////////////////

    float original_message_val = messages(n, direction, y + prev_row_shift, x + prev_col_shift, c) + log(message_scale(n, direction, y + prev_row_shift, x + prev_col_shift));
    float message_exp_sum = message_scale(n, direction, y + prev_row_shift, x + prev_col_shift); 
    sdata[tid] = in_grad(n, direction, y + prev_row_shift, x + prev_col_shift, c);
    __syncthreads();

    float in_grad_normalized = 0.0;
    // normalization
    for(unsigned short label = 0; label < C; ++label)
    {
        float J_norm_factor = - (1.0 / message_exp_sum) * exp(original_message_val);
       
        if(c == label)
        {
          J_norm_factor = 1.0 - (1.0 / message_exp_sum) * exp(original_message_val);
        }

        //printf("tid %i label %i norm msg val %f \n", tid, label, norm_msg_val);

        //in_grad is in sdata
        in_grad_normalized += sdata[label + blockDim.y * threadIdx.x] * J_norm_factor;
    }
    __syncthreads();

    ///////////////////////acc normalization ////////////////////////////////////////////
    sdata[tid] = getGradientAcc(gradient_accumulation, direction, n, y, x, c, HOR_IDX);
    __syncthreads();

    float acc_normalized = 0.0;
    // normalization
    for(unsigned short label = 0; label < C; ++label)
    {
        float J_norm_factor = - (1.0 / message_exp_sum) * exp(original_message_val);
       
        if(c == label)
        {
          J_norm_factor = 1.0 - (1.0 / message_exp_sum) * exp(original_message_val);
        }

        //in_grad is in sdata
        acc_normalized += sdata[label + blockDim.y * threadIdx.x] * J_norm_factor;
    }
    __syncthreads();

    /////////////////////////////////

    int min_index = (int)messages_argmin(n, direction, y + prev_row_shift, x + prev_col_shift, c);

    float additive_hor = in_grad_normalized + acc_normalized;
      
    float additive_up = 0.0;
    float additive_down = 0.0;
    if(compute_cross)
    {
      additive_up = saved_prev_grad_msg(n, UP, y + prev_row_shift, x + prev_col_shift, c) + getGradientAcc(gradient_accumulation, direction, n, y, x, c, UP_IDX);      
      additive_down = saved_prev_grad_msg(n, DOWN, y + prev_row_shift, x + prev_col_shift, c) + getGradientAcc(gradient_accumulation, direction, n, y, x, c, DOWN_IDX); 
    }

    // so that gradient_acc is not changed before assigning
    __syncthreads();

    //unary gradient
    atomicAdd(&gradient_unary(n, y, x, min_index), additive_hor);
    atomicAdd(&gradient_unary(n, y, x, min_index), additive_up);
    atomicAdd(&gradient_unary(n, y, x, min_index), additive_down);

    //pairwise gradient
    atomicAdd(&gradient_pairwise(grad_xy_idx, y, x, min_index, c), edgeWeight * additive_hor);
    atomicAdd(&gradient_pairwise(grad_xy_idx, y, x, min_index, c), edgeWeight * additive_up);
    atomicAdd(&gradient_pairwise(grad_xy_idx, y, x, min_index, c), edgeWeight * additive_down);

    //edge gradient
    // atomicAdd(&gradient_edge(0, grad_xy_idx, y, x), jump(grad_xy_idx, y, x, min_index, c) * additive_hor);
    // atomicAdd(&gradient_edge(0, grad_xy_idx, y, x), jump(grad_xy_idx, y, x, min_index, c) * additive_up);
    // atomicAdd(&gradient_edge(0, grad_xy_idx, y, x), jump(grad_xy_idx, y, x, min_index, c) * additive_down);

    updateGradientAcc(gradient_accumulation_tmp, additive_hor, direction, n, y, x, min_index, HOR_IDX);
    updateGradientAcc(gradient_accumulation_tmp, additive_up, direction, n, y, x, min_index, UP_IDX);
    updateGradientAcc(gradient_accumulation_tmp, additive_down, direction, n, y, x, min_index, DOWN_IDX);

    __syncthreads();

    setGradientAcc(gradient_accumulation, getGradientAcc(gradient_accumulation_tmp, direction, n, y, x, c, HOR_IDX), direction, n, y, x, c, HOR_IDX);
    setGradientAcc(gradient_accumulation, getGradientAcc(gradient_accumulation_tmp, direction, n, y, x, c, UP_IDX), direction, n, y, x, c, UP_IDX);
    setGradientAcc(gradient_accumulation, getGradientAcc(gradient_accumulation_tmp, direction, n, y, x, c, DOWN_IDX), direction, n, y, x, c, DOWN_IDX);

    __syncthreads();

    saved_prev_grad_msg(n, direction, y, x, c) = getGradientAcc(gradient_accumulation, direction, n, y, x, c, HOR_IDX);
  }

// ============================================================================
// CPP KERNEL CALLS
// ============================================================================
namespace cuda
{
  std::vector<at::Tensor> lbp_reduction_min_sum(at::Tensor cost, at::Tensor jump, at::Tensor edge, at::Tensor messages, unsigned short delta)
  {
    int N = cost.size(0);
    int H = cost.size(1);
    int W = cost.size(2);
    int C = cost.size(3);

    //int max_iter = 2;

    auto options = at::TensorOptions(cost.options());
    // at::Tensor messages = at::zeros({N, 4, H, W, C}, options);

    at::Tensor messages_argmin = at::zeros({N, 4, H, W, C}, options);
    at::Tensor message_scale = at::zeros({N, 4, H, W}, options);

    //cost = cost.permute({0, 2, 3, 1}).contiguous();

    // parallelize over image rows and disparities
    // block-size in disparity dimension must be >= number of disparities
    // then all the synchronization can be done over blocks (fast)
    // otherwise global synchronization is necessary
    int blockDimC = static_cast<int>(std::min(powf(2.0f, std::ceil(log2f(C))), 1024.0f));
    int blockDimHW = static_cast<int>(std::max(static_cast<float>(1024.0f / blockDimC / 1.0f), 1.0f));
    
    // attention: 1024 is maximal number of threads per block!!
    const dim3 blockSize(blockDimHW, blockDimC); 
    const dim3 numBlocksLR(std::ceil(H / static_cast<float>(blockSize.x)),
                           std::ceil(C / static_cast<float>(blockSize.y)));

    const dim3 numBlocksUD(std::ceil(W / static_cast<float>(blockSize.x)),
                           std::ceil(C / static_cast<float>(blockSize.y)));

    if(numBlocksLR.y != 1)
    {
      std::cout << "SOMETHING IS WRONG: Blocksize over disps is not 1=:" << numBlocksLR.y << "C=" << C << std::endl;
    }

    const int threadsPerBlock = blockSize.x * blockSize.y * blockSize.z;

    // to Right
    for(unsigned short x = 0; x < W - 1; ++x)
    {
      // compute min messages
      lbp_cuda_forward_kernel_reduction_min_sum<<<numBlocksLR, blockSize, 2 * threadsPerBlock * sizeof(float)>>>(cost, jump, edge, messages, messages_argmin, message_scale, x, RIGHT, threadsPerBlock, delta);
      cudaSafeCall(cudaGetLastError());
    }
    
    // to LEFT
    for(unsigned short x = W - 1; x > 0; --x)
    {
      // compute min messages
      lbp_cuda_forward_kernel_reduction_min_sum<<<numBlocksLR, blockSize, 2 * threadsPerBlock * sizeof(float)>>>(cost, jump, edge, messages, messages_argmin, message_scale, x, LEFT, threadsPerBlock, delta);
      cudaSafeCall(cudaGetLastError());
    }

    // to DOWN
    for(unsigned short y = 0; y < H - 1; ++y)
    {
      // compute min messages
      lbp_cuda_forward_kernel_reduction_min_sum<<<numBlocksUD, blockSize, 2 * threadsPerBlock * sizeof(float)>>>(cost, jump, edge, messages, messages_argmin, message_scale, y, DOWN, threadsPerBlock, delta);
      cudaSafeCall(cudaGetLastError());
    }

    // to UP
    for(unsigned short y = H - 1; y > 0; --y)
    {
      // compute min messages
      lbp_cuda_forward_kernel_reduction_min_sum<<<numBlocksUD, blockSize, 2 * threadsPerBlock * sizeof(float)>>>(cost, jump, edge, messages, messages_argmin, message_scale, y, UP, threadsPerBlock, delta);
      cudaSafeCall(cudaGetLastError());
    }
    
    //auto beliefs = messages.sum({1}) + cost; 
    std::vector<at::Tensor> output_vec;
    output_vec.push_back(messages);
    output_vec.push_back(messages_argmin);
    output_vec.push_back(message_scale);

    return output_vec;
  }

  std::vector<at::Tensor> lbp_forward_min_sum(at::Tensor cost, 
                         at::Tensor jump, 
                         at::Tensor edge,
                         at::Tensor messages, unsigned short delta)
  {
    return lbp_reduction_min_sum(cost, jump, edge, messages, delta);
  }           

  //=============================================================================
  // BACKWARD
  //=============================================================================
  std::vector<at::Tensor> lbp_backward_min_sum(at::Tensor cost, 
                                                at::Tensor edge,
                                                at::Tensor in_grad,
                                                at::Tensor messages,
                                                at::Tensor messages_argmin,
                                                at::Tensor message_scale)
  {
    int N = cost.size(0);
    int H = cost.size(1);
    int W = cost.size(2);
    int C = cost.size(3);

    auto options = at::TensorOptions(cost.options());

    at::Tensor gradient_unary = at::zeros({N, H, W, C}, options);

    at::Tensor gradient_pairwise = at::zeros({4, H, W, C, C}, options);

    at::Tensor gradient_edge = at::zeros({N, 4, H, W}, options);

    at::Tensor gradient_messages = at::zeros({N, 4, H, W, C}, options);

    gradient_messages += in_grad;

    at::Tensor saved_prev_grad_msg = at::zeros({N, 4, H, W, C}, options);

    at::Tensor gradient_accumulation;

    // parallelize over image rows and disparities
    // block-size in disparity dimension must be >= number of disparities
    // then all the synchronization can be done over blocks (fast)
    // otherwise global synchronization is necessary
    int blockDimC = static_cast<int>(std::min(powf(2.0f, std::ceil(log2f(C))), 1024.0f));
    int blockDimHW = static_cast<int>(std::max(static_cast<float>(1024.0f / blockDimC / 1.0f), 1.0f));

    // attention: 1024 is maximal number of threads per block!!
    const dim3 blockSize(blockDimHW, blockDimC); 
    const dim3 numBlocksLR(std::ceil(H / static_cast<float>(blockSize.x)),
                          std::ceil(C / static_cast<float>(blockSize.y)));

    const dim3 numBlocksUD(std::ceil(W / static_cast<float>(blockSize.x)),
                          std::ceil(C / static_cast<float>(blockSize.y)));

    //printf("blockDimC %i \n", blockDimC);
    //printf("blockDimHW %i \n", blockDimHW);

    if(numBlocksLR.y != 1)
      std::cout << "SOMETHING IS WRONG: Blocksize over disps is not 1: " << numBlocksLR.y << std::endl;

    const int threadsPerBlock = blockSize.x * blockSize.y * blockSize.z;

    const float max_float = 1e15;

    for(int n = 0; n < N; ++n)
    {
      ////////////////////UNARY GRADIENT////////////////////////////

      //to DOWN
      gradient_accumulation = at::zeros({N, W, 3, C}, options);
      for(short y = 1; y < H; ++y)
      {
           // compute min messages
           at::Tensor gradient_accumulation_tmp = at::zeros({N, W, 3, C}, options);
           lbp_cuda_backward_kernel_reduction_min_sum<<<numBlocksUD, blockSize, threadsPerBlock * sizeof(float)>>>(cost, edge, messages, messages_argmin, message_scale, in_grad, gradient_unary, gradient_pairwise, gradient_edge, gradient_accumulation, gradient_accumulation_tmp, saved_prev_grad_msg, y, DOWN, false, n);
           cudaSafeCall(cudaGetLastError());
      }

      // to UP
      gradient_accumulation = at::zeros({N, W, 3, C}, options);
      for(short y = H - 2; y >= 0; --y)
      {
          // compute min messages
          at::Tensor  gradient_accumulation_tmp = at::zeros({N, W, 3, C}, options);
          lbp_cuda_backward_kernel_reduction_min_sum<<<numBlocksUD, blockSize, threadsPerBlock * sizeof(float)>>>(cost, edge, messages, messages_argmin, message_scale, in_grad, gradient_unary, gradient_pairwise, gradient_edge, gradient_accumulation, gradient_accumulation_tmp, saved_prev_grad_msg, y, UP, false, n);
          cudaSafeCall(cudaGetLastError());
      }

      // to LEFT
      gradient_accumulation = at::zeros({N, H, 3, C}, options);
      for(short x = W-2; x >= 0; --x)
      {
        // compute min messages
        at::Tensor gradient_accumulation_tmp = at::zeros({N, W, 3, C}, options);
        lbp_cuda_backward_kernel_reduction_min_sum<<<numBlocksLR, blockSize, threadsPerBlock * sizeof(float)>>>(cost, edge, messages, messages_argmin, message_scale, in_grad, gradient_unary, gradient_pairwise, gradient_edge, gradient_accumulation, gradient_accumulation_tmp, saved_prev_grad_msg, x, LEFT, true, n);
        cudaSafeCall(cudaGetLastError());
      }

      // to RIGHT
      gradient_accumulation = at::zeros({N, H, 3, C}, options);
      for(short x = 1; x < W; ++x)
      {
        // compute min messages
        at::Tensor gradient_accumulation_tmp = at::zeros({N, W, 3, C}, options);
        lbp_cuda_backward_kernel_reduction_min_sum<<<numBlocksLR, blockSize, threadsPerBlock * sizeof(float)>>>(cost, edge, messages, messages_argmin, message_scale, in_grad, gradient_unary, gradient_pairwise, gradient_edge, gradient_accumulation, gradient_accumulation_tmp, saved_prev_grad_msg, x, RIGHT, true, n);
        cudaSafeCall(cudaGetLastError());
      }
    }

    std::vector<at::Tensor> output_vec;
    
    output_vec.push_back(gradient_unary);
    output_vec.push_back(gradient_pairwise);
    output_vec.push_back(gradient_edge);
    output_vec.push_back(gradient_messages);

    return output_vec;

  }

}