/*
 *		Copyright (c) by Damian Andrysiak. All rights reserved.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../common/book.h"

#include <stdio.h>
#include <stdlib.h>

#define HostToDevice	cudaMemcpyHostToDevice
#define DeviceToHost	cudaMemcpyDeviceToHost
#define N 10



__global__ void Add(const int const* a, const int const* b, int* c);

int main(int argc, char* argv[])
{
	int a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;
	// Alokacja pamiêci na GPU
	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));
	//Zape³nienie tablic a i b na CPU

	for (int i = 0; i < N; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}
	//Kopiowanie tablic a i b do GPU

	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), HostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), HostToDevice));
	Add << <N, 1 >> > (dev_a, dev_b, dev_c);

	// Kopiowanie tablicy c z GPU do CPU
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), DeviceToHost));


	// Wyœwietlenie wyniku
	for (int i = 0; i < N; i++)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	// Zwolnienie pamiêci alokowanej na GPU
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return EXIT_SUCCESS;
}

__global__ void Add(const int const* a, const int const* b, int* c)
{
	int tid = blockIdx.x;
	if (tid < N)
	{
		c[tid] = a[tid] + b[tid];
	}
}