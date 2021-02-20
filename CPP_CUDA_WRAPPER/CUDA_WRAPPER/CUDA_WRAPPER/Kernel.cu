

#include "Kernel.cuh"

#include "Cuda_Kernel.cuh"

void Cuda_Kernel::kernel_double(double* A, double* B, double* C, const std::size_t& array_size)
{
	//kernel<double>(A, B, C, array_size);

    double* d_A, * d_B, * d_C;


    cudaMalloc((void**)&d_A, array_size * sizeof(double));
    cudaMalloc((void**)&d_B, array_size * sizeof(double));
    cudaMalloc((void**)&d_C, array_size * sizeof(double));


    cudaMemcpy(d_A, A, array_size * sizeof(double), HostToDevice);
    cudaMemcpy(d_B, B, array_size * sizeof(double), DeviceToHost);


    dim3 blockSize(512, 1, 1);
    dim3 gridSize(512 / array_size + 1, 1);

    vector_addition_kernel<double> << <1, 1 >> > (d_A, d_B, d_C, array_size);


    cudaMemcpy(C, d_C, array_size * sizeof(double), DeviceToHost);
}
