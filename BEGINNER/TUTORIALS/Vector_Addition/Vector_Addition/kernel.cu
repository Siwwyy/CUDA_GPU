
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <iostream>
#include <random>
#include <algorithm>


#define HostToDevice	cudaMemcpyHostToDevice
#define DeviceToHost	cudaMemcpyDeviceToHost

constexpr std::size_t Vector_Size = 1 << 4;

//CPU FUNCTIONS

//GPU FUNCTIONS
__global__ void Vector_Addition(const float* const vec_a, const float* const vec_b, float* const vec_c);
__global__ void Vector_Addition(const int* const vec_a, const int* const vec_b, int* const vec_c);


int main(int argc, char* argv[])
{
	printf("%s Starting...\n", argv[0]);

	//printf_s("Vector size %llu\n", Vector_Size);
	std::cout << "Vector size " << Vector_Size << '\n';

	using array_type = int;
	auto* h_a = new array_type[Vector_Size];
	auto* h_b = new array_type[Vector_Size];
	auto* h_c = new array_type[Vector_Size];


	std::random_device random_device;
	std::mt19937 generator(random_device());
	//std::uniform_real_distribution<std::remove_pointer_t<decltype(h_a)>> distribution(1.0, static_cast<float>(Vector_Size));
	std::uniform_real_distribution<float> distribution(1.0, static_cast<float>(Vector_Size));


	std::generate_n(h_a, Vector_Size, [&]() mutable {return distribution(generator); });
	std::generate_n(h_b, Vector_Size, [&]() mutable {return distribution(generator); });
	//std::generate_n(h_c, Vector_Size, [&]() mutable {return distribution(generator); });


	auto print_array = [&](const auto* array_ptr) -> void
	{
		for (std::size_t i = 0; i < Vector_Size; ++i)
		{
			std::cout << array_ptr[i] << ' ';
		}
		std::cout << '\n';
	};

	print_array(h_a);
	print_array(h_b);
	//print_array(h_c);


	//GPU
	array_type* d_a{};
	array_type* d_b{};
	array_type* d_c{};
	cudaMalloc(reinterpret_cast<void**>(&d_a), Vector_Size * sizeof(array_type));	//GPU interprets 2D array as a flat array !
	cudaMalloc(reinterpret_cast<void**>(&d_b), Vector_Size * sizeof(array_type));	//GPU interprets 2D array as a flat array !
	cudaMalloc(reinterpret_cast<void**>(&d_c), Vector_Size * sizeof(array_type));	//GPU interprets 2D array as a flat array !

	cudaMemcpy(d_a, h_a, Vector_Size * sizeof(array_type), HostToDevice);
	cudaMemcpy(d_b, h_b, Vector_Size * sizeof(array_type), HostToDevice);

	dim3 blocks{ 1 };
	dim3 threads{ Vector_Size };

	Vector_Addition << <blocks, threads >> > (d_a, d_b, d_c);
	cudaDeviceSynchronize();

	cudaMemcpy(h_c, d_c, Vector_Size * sizeof(array_type), DeviceToHost);

	std::cout << " \nResult \n";
	print_array(h_c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	delete[] h_a, h_b, h_c;
	h_a, h_b, h_c = nullptr;

	system("pause");
	return EXIT_SUCCESS;
}

//DEFINITIONS

__global__ void Vector_Addition(const float* const vec_a, const float* const vec_b, float* const vec_c)
{
	const unsigned int thread_idx = threadIdx.x;
	//printf("%d ", thread_idx);

	if (thread_idx < Vector_Size)
	{
		vec_c[thread_idx] = vec_a[thread_idx] + vec_b[thread_idx];
	}
}

__global__ void Vector_Addition(const int* const vec_a, const int* const vec_b, int* const vec_c)
{
	const unsigned int thread_idx = threadIdx.x;
	//printf("%d ", thread_idx);

	if (thread_idx < Vector_Size)
	{
		vec_c[thread_idx] = vec_a[thread_idx] + vec_b[thread_idx];
	}
}