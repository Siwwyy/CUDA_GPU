#ifndef _KERNEL_CUH_
#define _KERNEL_CUH_
#pragma once

#include <iostream>
#include <cstdint>

#include "Cuda_Kernel.cuh"

#define HostToDevice	cudaMemcpyHostToDevice
#define DeviceToHost	cudaMemcpyDeviceToHost

enum class device_type
{
	global = 0,
	host,
	device,
	device_type_amount
};

namespace Cuda_Kernel
{
	
    void kernel_double(double* A, double* B, double* C, const std::size_t& array_size);

	template<typename Runnable, device_type device_type_>
	void test_function(const Runnable& runnable)
	{
		if constexpr(device_type_ == device_type::global)
		{
			//global
			test_global<Runnable>(runnable);
			
		}
		else if constexpr (device_type_ == device_type::host)
		{
			//host
		}
		else
		{
			//device
		}
	}
}

#endif /* _KERNEL_CUH_ */