#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

/** \brief Struct pointer TensorKernelData that can be used in CUDA kernels.
 *
 *  This struct provides the device data pointer as well as important class
 *  properties.
 */
struct KernelData5
{
  /** Pointer to device buffer. */
  float* data_;

  unsigned int stride0;
  unsigned int stride1;
  unsigned int stride2;
  unsigned int stride3;
  unsigned int stride4;

  // size of dimensions 0 - 4
  unsigned short size0;
  unsigned short size1;
  unsigned short size2;		
  unsigned short size3;
  unsigned short size4;
  

  /** Access the image via the () operator 
   * @param x0 Position in the first dimension.
   * @param x1 Position in the second dimension.
   * @param x2 Position in the third dimension.
   * @param x3 Position in the forth dimension.
   * @param x4 Position in the fifth dimension.
   * @return value at position (x0, x1, x2, x3, x4).
   */
  __device__ float& operator()(short x0, short x1, short x2, short x3, short x4)
  {
    return data_[x0 * stride0 + x1 * stride1 + x2 * stride2 + x3 *stride3 + x4];
  }

  /** Get position / coordinates for a linear index.
   * @param[in] linearIdx Linear index.
   * @param[out] dim0 Position in the first dimension.
   * @param[out] dim1 Position in the second dimension.
   * @param[out] dim2 Position in the third dimension.
   * @param[out] dim3 Position in the forth dimension.
   */
  __device__ void coords(unsigned int linearIdx, short *x0, short *x1, short *x2, short *x3, short *x4)
  {
          // modulo is slow
//            *dim0 = linearIdx / stride0;
//            *dim1 = (linearIdx % stride0) / stride1;
//            *dim2 = ((linearIdx % stride0) % stride1) / stride2;
//            *dim3 = ((linearIdx % stride0) % stride1) % stride2;
          *x0 = linearIdx / stride0;
          *x1 = (linearIdx - *x0 * stride0) / stride1;
          *x2 = (linearIdx - (*x0 * stride0 + *x1 * stride1)) / stride2;
          *x3 = linearIdx - (*x0 * stride0 + *x1 * stride1 + *x2 * stride2);
          *x4 = linearIdx - (*x0 * stride0 + *x1 * stride1 + *x2 * stride2 + *x3 * stride3);
      }

  /** Constructor */
  __host__ KernelData5(const at::Tensor &tensor) :
              data_(tensor.data<float>()), 
              size0(tensor.size(0)), 
              size1(tensor.size(1)),
              size2(tensor.size(2)), 
              size3(tensor.size(3)),
              size4(tensor.size(4)),
              stride0(tensor.stride(0)),
              stride1(tensor.stride(1)),
              stride2(tensor.stride(2)),
              stride3(tensor.stride(3)),
              stride4(tensor.stride(4))
  {
  }
};

/** \brief Struct pointer TensorKernelData that can be used in CUDA kernels.
 *
 *  This struct provides the device data pointer as well as important class
 *  properties.
 */
struct KernelData
{
  /** Pointer to device buffer. */
  float* data_;

  unsigned int stride0;
  unsigned int stride1;
  unsigned int stride2;
  unsigned int stride3;

  // size of dimensions 0 - 3
  unsigned short size0;
  unsigned short size1;
  unsigned short size2;		
  unsigned short size3;
  

  /** Access the image via the () operator 
   * @param x0 Position in the first dimension.
   * @param x1 Position in the second dimension.
   * @param x2 Position in the third dimension.
   * @param x3 Position in the forth dimension.
   * @return value at position (x0, x1, x2, x3).
   */
  __device__ float& operator()(short x0, short x1, short x2, short x3)
  {
    return data_[x0 * stride0 + x1 * stride1 + x2 * stride2 + x3];
  }

  /** Get position / coordinates for a linear index.
   * @param[in] linearIdx Linear index.
   * @param[out] dim0 Position in the first dimension.
   * @param[out] dim1 Position in the second dimension.
   * @param[out] dim2 Position in the third dimension.
   * @param[out] dim3 Position in the forth dimension.
   */
  __device__ void coords(unsigned int linearIdx, short *x0, short *x1, short *x2, short *x3)
  {
          // modulo is slow
//            *dim0 = linearIdx / stride0;
//            *dim1 = (linearIdx % stride0) / stride1;
//            *dim2 = ((linearIdx % stride0) % stride1) / stride2;
//            *dim3 = ((linearIdx % stride0) % stride1) % stride2;
          *x0 = linearIdx / stride0;
          *x1 = (linearIdx - *x0 * stride0) / stride1;
          *x2 = (linearIdx - (*x0 * stride0 + *x1 * stride1)) / stride2;
          *x3 = linearIdx - (*x0 * stride0 + *x1 * stride1 + *x2 * stride2);
      }

  /** Constructor */
  __host__ KernelData(const at::Tensor &tensor) :
              data_(tensor.data<float>()), 
              size0(tensor.size(0)), 
              size1(tensor.size(1)),
              size2(tensor.size(2)), 
              size3(tensor.size(3)),
              stride0(tensor.stride(0)),
              stride1(tensor.stride(1)),
              stride2(tensor.stride(2)),
              stride3(tensor.stride(3))
  {
    // std::cout << "size of size " << tensor.sizes().size() << std::endl;
    // std::cout << "s0 " << tensor.size(0) << std::endl;
    // std::cout << "s1 " << tensor.size(1) << std::endl;
    // std::cout << "s2 " << tensor.size(2) << std::endl;
    // std::cout << "s3 " << tensor.size(3) << std::endl;
    //std::cout << "s0 " << tensor.size(4) << std::endl;
  }
};