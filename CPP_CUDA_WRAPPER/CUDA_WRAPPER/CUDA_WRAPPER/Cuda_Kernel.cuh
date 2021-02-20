#ifndef _CUDA_KERNEL_CUH_
#define _CUDA_KERNEL_CUH_
#pragma once

#include <iostream>
#include <cstdint>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define HostToDevice	cudaMemcpyHostToDevice
#define DeviceToHost	cudaMemcpyDeviceToHost

namespace Cuda_Kernel
{
    template<class _Ty>
    __global__ void vector_addition_kernel(_Ty* A, _Ty* B, _Ty* C, const std::size_t& array_size)
    {
        int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    	
        if (threadID < array_size)
        {
            C[threadID] = A[threadID] + B[threadID];
        }
    }
	
}

#endif /* _CUDA_KERNEL_CUH_ */