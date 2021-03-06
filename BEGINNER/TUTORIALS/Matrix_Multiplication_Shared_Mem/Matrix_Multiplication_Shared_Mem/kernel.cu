﻿
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <iostream>
#include <random>
#include <memory>
#include <algorithm>
#include <vector>


#define HostToDevice	cudaMemcpyHostToDevice
#define DeviceToHost	cudaMemcpyDeviceToHost

//constexpr std::size_t Dim_X = 1llu << 9llu;
//constexpr std::size_t Dim_Y = 1llu << 9llu;


//constexpr std::size_t Dim_X = 1 << 10llu;
constexpr std::size_t Dim_X = 3;
//constexpr std::size_t Dim_Y = 1 << 10llu;
constexpr std::size_t Dim_Y = 3;


//CPU FUNCTIONS
void Print_Matrix(const std::unique_ptr<std::unique_ptr<int[]>[]>& Matrix);

//GPU FUNCTIONS
__global__ void Matrix_Multiplication_Shared(const int* const Matrix_A, const int* const Matrix_B, int* const Matrix_C);


int main(int argc, char* argv[])
{
	int nDevices{};
	int id = cudaGetDevice(&id);

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++)
	{
		cudaDeviceProp prop{};
		cudaGetDeviceProperties(&prop, i);
		printf("   --- General Information for device %d ---\n", i);
		printf("Name:  %s\n", prop.name);
		printf("Compute capability:  %d.%d\n", prop.major, prop.minor);
		printf("Clock rate:  %d\n", prop.clockRate);
		printf("   --- Memory Information for device %d ---\n", i);
		printf("Total global mem:  %ld\n", static_cast<long>(prop.totalGlobalMem));
		printf("Total constant Mem:  %ld\n", static_cast<long>(prop.totalConstMem));
		printf("Max mem pitch:  %ld\n", static_cast<long>(prop.memPitch));
		printf("Texture Alignment:  %ld\n", static_cast<long>(prop.textureAlignment));

		printf("   --- MP Information for device %d ---\n", i);
		printf("Multiprocessor count:  %d\n", prop.multiProcessorCount);
		printf("Shared mem per mp:  %ld\n", static_cast<long>(prop.sharedMemPerBlock));
		printf("Registers per mp:  %d\n", prop.regsPerBlock);
		printf("Threads in warp:  %d\n", prop.warpSize);
		printf("Max threads per block:  %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions:  (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions:  (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf_s("\n");
	}
	printf("%s Starting...\n\n", argv[0]);
	system("pause");

	//CPU
	using type = int;
	using ptr_type = std::unique_ptr<type*>;

	const std::unique_ptr<std::unique_ptr<type[]>[]> Matrix_A(new std::unique_ptr<type[]>[Dim_Y]);
	//type** Matrix_A = new type*[Dim_Y];
	const std::unique_ptr<std::unique_ptr<type[]>[]> Matrix_B(new std::unique_ptr<type[]>[Dim_Y]);
	const std::unique_ptr<std::unique_ptr<type[]>[]> Matrix_C(new std::unique_ptr<type[]>[Dim_Y]);


	for (std::size_t i = 0ull; i < Dim_Y; ++i)
	{
		Matrix_A[i] = std::make_unique<type[]>(Dim_X);
		Matrix_B[i] = std::make_unique<type[]>(Dim_X);
		Matrix_C[i] = std::make_unique<type[]>(Dim_X);

		for (std::size_t j = 0ull; j < Dim_X; ++j)
		{
			Matrix_A[i][j] = static_cast<type>(i * Dim_Y + j + 1);
			Matrix_B[i][j] = static_cast<type>(i * Dim_Y + j + 1);
		}
	}

	//GPU
	type* Dev_Matrix_A{};
	cudaMalloc(reinterpret_cast<void**>(&Dev_Matrix_A), Dim_X * Dim_Y * sizeof(type));	//GPU interprets 2D array as a flat array !
	type* Dev_Matrix_B{};
	cudaMalloc(reinterpret_cast<void**>(&Dev_Matrix_B), Dim_X * Dim_Y * sizeof(type));	//GPU interprets 2D array as a flat array !
	type* Dev_Matrix_C{};
	cudaMalloc(reinterpret_cast<void**>(&Dev_Matrix_C), Dim_X * Dim_Y * sizeof(type));	//GPU interprets 2D array as a flat array !


	type* a_d{};
	cudaMalloc(reinterpret_cast<void**>(&a_d), sizeof(type));	//GPU interprets 2D array as a flat array !

	//Copy memory from CPU to GPU
	for (std::size_t i = 0ull; i < Dim_Y; ++i)
	{
		cudaMemcpy(reinterpret_cast<void*>(Dev_Matrix_A + i * Dim_Y), reinterpret_cast<const void*>((Matrix_A.get() + i)->get()), sizeof(type) * Dim_Y, HostToDevice);
		cudaMemcpy(reinterpret_cast<void*>(Dev_Matrix_B + i * Dim_Y), reinterpret_cast<const void*>((Matrix_B.get() + i)->get()), sizeof(type) * Dim_Y, HostToDevice);
	}


	{
		dim3 threads(Dim_X);
		dim3 blocks(Dim_X, Dim_Y);
		Matrix_Multiplication_Shared << <blocks, threads >> > (Dev_Matrix_A, Dev_Matrix_B, Dev_Matrix_C);
		cudaDeviceSynchronize();
	}

	//copying data from GPU to CPU
	for (std::size_t i = 0ull; i < Dim_Y; ++i)
	{
		cudaMemcpy(reinterpret_cast<void*>((Matrix_C.get() + i)->get()), reinterpret_cast<const void*>(Dev_Matrix_C + i * Dim_Y), sizeof(type) * Dim_Y, DeviceToHost);
	}

	Print_Matrix(Matrix_C);

	cudaFree(Dev_Matrix_A);
	cudaFree(Dev_Matrix_B);
	cudaFree(Dev_Matrix_C);

	system("pause");
	return EXIT_SUCCESS;
}


//DEFINITIONS OF FUNCTIONS


//CPU
void Print_Matrix(const std::unique_ptr<std::unique_ptr<int[]>[]>& Matrix)
{
	for (std::size_t i = 0ull; i < Dim_Y; ++i)
	{
		for (std::size_t j = 0ull; j < Dim_X; ++j)
		{
			std::cout << Matrix[i][j] << ' ';
		}
		std::cout << '\n';
	}
}



//GPU
__global__ void Show_Matrix_GPU(const int* const Matrix)
{
	const unsigned int id_x = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int id_y = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int index = id_x + id_y * gridDim.x * gridDim.y;
	const unsigned int threads_amount = blockDim.x * gridDim.x * blockDim.y * gridDim.y;

	//printf("Threads amount %d ", threads_amount);

	while (index < Dim_X * Dim_Y)
	{
		printf("Thread idx: %d | Thread idy %d | Thread id: %d | Value: %d \n", id_x, id_y, index, Matrix[index]);
		index += threads_amount;
	}
	printf("Thread id: %d |\n", index);
}

__global__ void Matrix_Multiplication_Shared(const int* const Matrix_A, const int* const Matrix_B, int* const Matrix_C)
{
	__shared__ int Row[Dim_X];
	__shared__ int Col[Dim_Y];
	__shared__ int Values[Dim_Y];

	const unsigned row = blockIdx.y * blockDim.x + threadIdx.x;
	const unsigned col = blockDim.x * threadIdx.x + blockIdx.x;
	const unsigned block_index = gridDim.x * blockIdx.y + blockIdx.x;

	Row[threadIdx.x] = Matrix_A[row];
	Col[threadIdx.x] = Matrix_B[col];
	__syncthreads();

	Values[threadIdx.x] = (Row[threadIdx.x] * Col[threadIdx.x]);
	__syncthreads();

	for (std::size_t i = 0; i < blockDim.x; i++)
	{
		Matrix_C[block_index] += Values[i];
	}
}