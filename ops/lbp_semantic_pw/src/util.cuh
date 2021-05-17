#pragma once
#include <ATen/ATen.h>
#include <vector>

#include "../../include/tensor.h"

enum DIRECTION {LEFT, RIGHT, UP, DOWN};


extern __device__ __forceinline__ float getDerivativeValue(KernelData prev_values, int direction, float current_value, float max_float) 
{
  float ret_val = 0.0;
  if(prev_values(0, direction, 0, 0) != max_float)
  {
    ret_val = prev_values(0, direction, 0, 0);
  }
  else
  {
    ret_val = current_value;
  }

  return ret_val;
}


extern __device__ __forceinline__ float computeCrossGradient(KernelData prev_values, float max_float) 
{
  
  bool is_cross = false;
  if(prev_values(0,0,0,0) != max_float ||
     prev_values(0,1,0,0) != max_float ||
     prev_values(0,2,0,0) != max_float ||
     prev_values(0,3,0,0) != max_float)
     {
       is_cross = true;
     }

  return is_cross;
}

extern __device__ __forceinline__ float getGradientAcc(KernelData gradient_accumulation, int direction, int n, int y, int x, int c, int grad_acc_idx) 
{

  float ret_val = 0.0;
  if(direction == UP || direction == DOWN)
  {
    ret_val = gradient_accumulation(n, x, grad_acc_idx, c);
  }
  if(direction == LEFT || direction == RIGHT)
  {
    ret_val = gradient_accumulation(n, y, grad_acc_idx, c);
  }

  return ret_val;

}

extern __device__ __forceinline__ void updateGradientAcc(KernelData gradient_accumulation, float value, int direction, int n, int y, int x, int c, int grad_acc_idx) 
{

  if(direction == UP || direction == DOWN)
  {
    //gradient_accumulation(n, x, grad_acc_idx, c) = value;
    atomicAdd(&gradient_accumulation(n, x, grad_acc_idx, c), value);
  }
  if(direction == LEFT || direction == RIGHT)
  {
    //gradient_accumulation(n, y, grad_acc_idx, c) = value;
    atomicAdd(&gradient_accumulation(n, y, grad_acc_idx, c), value);
  }

}

extern __device__ __forceinline__ void setGradientAcc(KernelData gradient_accumulation, float value, int direction, int n, int y, int x, int c, int grad_acc_idx) 
{

  if(direction == UP || direction == DOWN)
  {
    gradient_accumulation(n, x, grad_acc_idx, c) = value;
  }
  if(direction == LEFT || direction == RIGHT)
  {
    gradient_accumulation(n, y, grad_acc_idx, c) = value;
  }

}

extern __device__ __forceinline__ float getEdgeWeight(KernelData edges, int n, int y, int x, int direction) 
{
  
  float w = 1.0;
  if(direction == UP || direction == DOWN)
  {
    w = edges(n, 1, y, x);
  }
  else
  {
    w = edges(n, 0, y, x);
  }

  return w;
}
