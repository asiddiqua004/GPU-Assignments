// Race condition in CUDA

#include <stdio.h>
#include <cuda_runtime.h>


// includes, project
////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"

// my kernel
/**
 * CUDA Kernel Device code
 *
 * Computes cooperative additions
 */
// FILL HERE: Define a kernel function that accumulates 
//            the given input parameter without atomic operation.
//            And then, change the code to use atomic operation.
//            Check the final output value.


/**
 * Host main routine
 */
int 
main(void) 
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    int a = 0;

    int* d_a = NULL;	
    err = cudaMalloc((void**)&d_a, sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Kernel Invocation 
    int blocksPerGrid = 125;
    int threadsPerBlock = 1000;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    myKernel<<< blocksPerGrid, threadsPerBlock>>>(d_a);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch race kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaThreadSynchronize();
	
    // Copy the device result in device memory to the host result variable
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy A from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaThreadSynchronize();

    printf("a = %d\n", a);
    
    // Free device global memory
    cudaFree(d_a);

    return 0;
}

