/*
 *		Copyright (c) by Damian Andrysiak. All rights reserved.
 *					  *** End: 24/6/2020 ***
 *  First CUDA Project, HELLO WORLD
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../common/book.h"

#include <stdio.h>
#include <stdlib.h>

#define HostToDevice	cudaMemcpyHostToDevice
#define DeviceToHost	cudaMemcpyDeviceToHost


__global__ void Kernel(void);
__global__ void Add(const int a, const int b, int* c);
__host__ void Hello_From_Host(void);

int main(int argc, char* argv[])
{

	/* Kernel << < 1, 1 >> > (); //basic kernel function, called from host and launched at device
	 Hello_From_Host();*/    //basic kernel function, called from host and launched at host


	int c{};
	int* dev_c{};

	HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));	//alocate memory for dev_c variable

	Add << <1, 1 >> > (2, 7, dev_c);	//launch kernel function

	HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), DeviceToHost));	//copy value of dev_c to c, dev_c is device variable, c is a host device
	printf("2 + 7 = %d\n", c);	//print value of variable c

	cudaFree(dev_c);	//deallocate memory by using cudaFree function on pointer dev_c, cause dev_c is a dynamic allocated memory at the runtime
	return EXIT_SUCCESS;
}

__global__ void Kernel(void)
{
	//nothing here
}

__global__ void Add(const int a, const int b, int* c)
{
	*c = a + b;
}

__host__ void Hello_From_Host(void)
{
	printf_s("%s", "Hello World from CPU!");
}