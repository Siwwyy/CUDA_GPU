#ifndef _KERNEL_CUH_
#define _KERNEL_CUH_
#pragma once

#include <iostream>
#include <cstdint>


#define HostToDevice	cudaMemcpyHostToDevice
#define DeviceToHost	cudaMemcpyDeviceToHost

namespace Cuda_Kernel
{
	
    void kernel_double(double* A, double* B, double* C, const std::size_t& array_size);
	
}

#endif /* _KERNEL_CUH_ */