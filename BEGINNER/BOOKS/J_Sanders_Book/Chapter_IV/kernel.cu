/*
 *		Copyright (c) by Damian Andrysiak. All rights reserved.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../common/book.h"

#include <stdio.h>
#include <iostream>
#include <stdlib.h>

#define HostToDevice	cudaMemcpyHostToDevice
#define DeviceToHost	cudaMemcpyDeviceToHost
#define N 10



__global__ void Add(const int const* a, const int const* b, int* c);


#include "../../common/book.h"
#include "../../common/cpu_bitmap.h"

#define DIM 1000

struct cuComplex {
	float   r;
	float   i;
	// cuComplex( float a, float b ) : r(a), i(b)  {}
	__device__ cuComplex(float a, float b) : r(a), i(b) {} // Fix error for calling host function from device
	__device__ float magnitude2(void) {
		return r * r + i * i;
	}
	__device__ cuComplex operator*(const cuComplex& a) {
		return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a) {
		return cuComplex(r + a.r, i + a.i);
	}
};

__device__ int julia(int x, int y) {
	const float scale = 1.5;
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);

	int i = 0;
	for (i = 0; i < 200; i++) {
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}

	return 1;
}

__global__ void kernel(unsigned char* ptr) {
	// map from blockIdx to pixel position
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;

	// now calculate the value at that position
	int juliaValue = julia(x, y);
	ptr[offset * 4 + 0] = 255 * juliaValue;
	ptr[offset * 4 + 1] = 0;
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 3] = 255;
}

// globals needed by the update routine
struct DataBlock {
	unsigned char* dev_bitmap;
};

int main(int argc, char* argv[])
{
	//cudaDeviceProp  prop;

	//int count;
	//HANDLE_ERROR(cudaGetDeviceCount(&count));
	//for (int i = 0; i < count; i++) {
	//    HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
	//    printf("   --- General Information for device %d ---\n", i);
	//    printf("Name:  %s\n", prop.name);
	//    printf("Compute capability:  %d.%d\n", prop.major, prop.minor);
	//    printf("Clock rate:  %d\n", prop.clockRate);
	//    printf("Device copy overlap:  ");
	//    if (prop.deviceOverlap)
	//        printf("Enabled\n");
	//    else
	//        printf("Disabled\n");
	//    printf("Kernel execution timeout :  ");
	//    if (prop.kernelExecTimeoutEnabled)
	//        printf("Enabled\n");
	//    else
	//        printf("Disabled\n");

	//    printf("   --- Memory Information for device %d ---\n", i);
	//    printf("Total global mem:  %ld\n", prop.totalGlobalMem);
	//    printf("Total constant Mem:  %ld\n", prop.totalConstMem);
	//    printf("Max mem pitch:  %ld\n", prop.memPitch);
	//    printf("Texture Alignment:  %ld\n", prop.textureAlignment);

	//    printf("   --- MP Information for device %d ---\n", i);
	//    printf("Multiprocessor count:  %d\n",
	//        prop.multiProcessorCount);
	//    printf("Shared mem per mp:  %ld\n", prop.sharedMemPerBlock);
	//    printf("Registers per mp:  %d\n", prop.regsPerBlock);
	//    printf("Threads in warp:  %d\n", prop.warpSize);
	//    printf("Max threads per block:  %d\n",
	//        prop.maxThreadsPerBlock);
	//    printf("Max thread dimensions:  (%d, %d, %d)\n",
	//        prop.maxThreadsDim[0], prop.maxThreadsDim[1],
	//        prop.maxThreadsDim[2]);
	//    printf("Max grid dimensions:  (%d, %d, %d)\n",
	//        prop.maxGridSize[0], prop.maxGridSize[1],
	//        prop.maxGridSize[2]);
	//    printf("\n");
	//}

	//int a[N], b[N], c[N];
	//int* dev_a, * dev_b, * dev_c;
	//// Alokacja pamiêci na GPU
	//HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	//HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	//HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));
	////Zape³nienie tablic a i b na CPU

	//for (int i = 0; i < N; i++)
	//{
	//	a[i] = -i;
	//	b[i] = i * i;
	//}
	////Kopiowanie tablic a i b do GPU

	//HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), HostToDevice));
	//HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), HostToDevice));
	//Add << <N, 1 >> > (dev_a, dev_b, dev_c);

	//// Kopiowanie tablicy c z GPU do CPU
	//HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), DeviceToHost));


	//// Wyœwietlenie wyniku
	//for (int i = 0; i < N; i++)
	//{
	//	printf("%d + %d = %d\n", a[i], b[i], c[i]);
	//}

	//// Zwolnienie pamiêci alokowanej na GPU
	//cudaFree(dev_a);
	//cudaFree(dev_b);
	//cudaFree(dev_c);

	DataBlock   data;
	CPUBitmap bitmap(DIM, DIM, &data);
	unsigned char* dev_bitmap;
	//std::cout << sizeof(dev_bitmap) << '\n';
	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
	data.dev_bitmap = dev_bitmap;

	dim3    grid(DIM, DIM);
	kernel << <grid, 1 >> > (dev_bitmap);

	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap,
		bitmap.image_size(),
		cudaMemcpyDeviceToHost));

	HANDLE_ERROR(cudaFree(dev_bitmap));

	bitmap.display_and_exit();


	return EXIT_SUCCESS;
}

__global__ void Add(const int const* a, const int const* b, int* c)
{
	int tid = blockIdx.x;
	if (tid < N)
	{
		printf("%d \n", tid);
		c[tid] = a[tid] + b[tid];
	}
	//c[300] = 200;
	//printf("%d \n", tid);
	//c[tid] = a[tid] + b[tid];
}