
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


constexpr std::size_t Dim_X = 3;
constexpr std::size_t Dim_Y = 3;

//CPU FUNCTIONS
void Print_Matrix(const std::unique_ptr<std::unique_ptr<int[]>[]>& Matrix);

//GPU FUNCTIONS
__global__ void Show_Matrix_GPU(const int* const Matrix);
__global__ void Matrix_Multiplication(const int* const Matrix_A, const int* const Matrix_B, int* const Matrix_C);


int main(int argc, char* argv[])
{
	printf("%s Starting...\n", argv[0]);


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
		//Matrix_A[i] = new type[Dim_X];
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


	//Copy memory from CPU to GPU
	for (std::size_t i = 0ull; i < Dim_Y; ++i)
	{
		//std::cout << *(*(Matrix_A.get() + i)).get() << '\n';
		//std::cout << (*(Matrix_A.get() + i)).get() << '\n';
		//std::cout << (*(Matrix_A) + i)<< '\n';
		//cudaMemcpy(reinterpret_cast<void*>(Dev_Matrix_A + i * Dim_Y), reinterpret_cast<const void*>(Matrix_A.get() + i), sizeof(type) * Dim_Y, HostToDevice);
		//cudaMemcpy(reinterpret_cast<void*>(Dev_Matrix_B + i * Dim_Y), reinterpret_cast<const void*>(Matrix_B.get() + i), sizeof(type) * Dim_Y, HostToDevice);
		//cudaMemcpy(reinterpret_cast<void*>(Dev_Matrix_C + i * Dim_Y), reinterpret_cast<const void*>(Matrix_C.get() + i), sizeof(type) * Dim_Y, HostToDevice);
		std::cout << i + i * Dim_Y << '\n';
		//cudaMemcpy(Matrix_GPU_B + i * SIZE, *(Matrix_CPU_B + i), sizeof(type) * SIZE, HostToDevice);
		//cudaMemcpy(Matrix_GPU_C + i * SIZE, *(Matrix_CPU_C + i), sizeof(type) * SIZE, HostToDevice);
	}

	{
		dim3 blocks{ 1 };
		dim3 threads{ 3 };
		Show_Matrix_GPU << <blocks, threads >> > (Dev_Matrix_A);//v
		cudaDeviceSynchronize();
	}

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
	//unsigned int id_x = threadIdx.x + blockIdx.x * blockDim.x;
	//unsigned int id_y = threadIdx.y + blockIdx.y * blockDim.y;
	//while (id_y < Dim_Y)
	//{
	//	while (id_x < Dim_X)
	//	{
	//		printf("| %d ", Matrix[id_y * Dim_Y + id_x]);
	//		//printf("| %d ", static_cast<int>(id_x * Dim_Y + id_y));
	//		id_x += blockDim.x * gridDim.x;
	//	}
	//	id_y += blockDim.y * gridDim.y;
	//}
	unsigned int id_x = threadIdx.x + blockIdx.x * blockDim.x;
	while (id_x < Dim_X*Dim_Y)
	{
		printf("| %d ", Matrix[id_x]);
		//printf("| %d ", static_cast<int>(id_x * Dim_Y + id_y));
		id_x += blockDim.x * gridDim.x;
	}
	printf("\n");
}

__global__ void Matrix_Multiplication(const int* const Matrix_A, const int* const Matrix_B, int* const Matrix_C)
{

}
