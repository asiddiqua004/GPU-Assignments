/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
*/
// Error check function

void  Check (cudaError_t check)
{
    if (check != cudaSuccess)
    {
        fprintf(stderr, "Failure, exiting:(error code %s)!\n", cudaGetErrorString(check));
        exit(EXIT_FAILURE);
    }
}


// Kernel implemented with streams for overlapping
 __global__ void
vecAdd(const float *A, const float *B, float *C)
{        
    	int i= (blockIdx.x * blockDim.x) +  threadIdx.x; 
        C[i] = A[i] + B[i];
}

/**
 * Host main routine
 */
int
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 2097152;
    size_t size = numElements * sizeof(float);
    size_t size1 = (numElements/4) * sizeof(float);
    printf("[Vector addition of %d elements, size: %d size1: %d]\n", numElements, size, size1);
    
    // Create 4 streams
	cudaStream_t stream0, stream1, stream2, stream3;


	//Events
    cudaEvent_t overlapStart, overlapEnd;
    cudaEvent_t transin_or,stopout_or;
    
    cudaEventCreate(&overlapStart);
    cudaEventCreate(&overlapEnd);

	
    cudaEventCreate(&transin_or); 
    cudaEventCreate(&stopout_or);
	
	
	float seq_elapsed=0, over_elapsed=0;

    // Allocate the host input vector A B and C
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_Cseq = (float *)malloc(size);
    float *h_Cover = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_Cseq == NULL || h_Cover == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A B and C
    float *d_A = NULL;
    err = cudaMalloc ((void**)&d_A, size);
    Check(err);
    float *d_B = NULL;
    err = cudaMalloc((void**)&d_B, size);
    Check(err);
    float *d_C = NULL;
    err = cudaMalloc((void**)&d_C, size);
    Check(err);


    // Allocate the device input vector A for all the four streams d_A0, d_A1, d_A2, d_A3
    float *d_A0 = NULL;
    err = cudaMalloc ((void**)&d_A0, size1);
    Check(err);
    float *d_A1 = NULL;
    err = cudaMalloc ((void**)&d_A1, size1);
    Check(err);
    float *d_A2 = NULL;
    err = cudaMalloc ((void**)&d_A2, size1);
    Check(err);
    float *d_A3 = NULL;
    err = cudaMalloc ((void**)&d_A3, size1);
    Check(err);

    // Allocate the device input vector A for all the four streams d_B0, d_B1, d_B2, d_B3
    float *d_B0 = NULL;
    err = cudaMalloc ((void**)&d_B0, size1);
    Check(err);
    float *d_B1 = NULL;
    err = cudaMalloc ((void**)&d_B1, size1);
    Check(err);    
    float *d_B2 = NULL;
    err = cudaMalloc ((void**)&d_B2, size1);
    Check(err);    
    float *d_B3 = NULL;
    err = cudaMalloc ((void**)&d_B3, size1);
    Check(err);

    // Allocate the device input vector A for all the four streams d_C0, d_C1, d_C2, d_C3
    float *d_C0 = NULL;
    err = cudaMalloc ((void**)&d_C0, size1);
    Check(err);
    float *d_C1 = NULL;
    err = cudaMalloc ((void**)&d_C1, size1);
    Check(err);    
    float *d_C2 = NULL;
    err = cudaMalloc ((void**)&d_C2, size1);
    Check(err);    
    float *d_C3 = NULL;
    err = cudaMalloc ((void**)&d_C3, size1);
    Check(err);
   
    // Launch the Vector Add CUDA Kernel
    
    int blocksPerGrid = (numElements/128);
    int threadsPerBlock =128;
    int i =(numElements/4);
    int j =(blocksPerGrid/4);
	
    /*
     * OverLapped
     */
    cudaStreamCreate(&stream0); cudaStreamCreate(&stream1); cudaStreamCreate(&stream2);cudaStreamCreate(&stream3);

    cudaEventRecord(overlapStart);
    cudaMemcpyAsync(d_A0, h_A, (i * sizeof(float)),cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(d_B0, h_B, (i * sizeof(float)),cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(d_A1, h_A+i, (i * sizeof(float)),cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B1, h_B+i, (i * sizeof(float)),cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_A2, h_A+2*i, (i * sizeof(float)),cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_B2, h_B+2*i, (i * sizeof(float)),cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_A3, h_A+3*i, (i * sizeof(float)),cudaMemcpyHostToDevice, stream3);
    cudaMemcpyAsync(d_B3, h_B+3*i, (i * sizeof(float)),cudaMemcpyHostToDevice, stream3);
    
    vecAdd<<<j,128, 0, stream0>>>(d_A0, d_B0, d_C0);
    vecAdd<<<j,128, 0, stream1>>>(d_A1, d_B1, d_C1);
    vecAdd<<<j,128, 0, stream2>>>(d_A2, d_B2, d_C2);
    vecAdd<<<j,128, 0, stream3>>>(d_A3, d_B3, d_C3);
    
    cudaMemcpyAsync(h_Cover, d_C0,(i * sizeof(float)),cudaMemcpyDeviceToHost, stream0);
    cudaMemcpyAsync(h_Cover+i,d_C1, (i *sizeof(float)), cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(h_Cover+2*i, d_C2,(i * sizeof(float)),cudaMemcpyDeviceToHost, stream2);
    cudaMemcpyAsync(h_Cover+3*i, d_C3, (i * sizeof(float)),cudaMemcpyDeviceToHost, stream3);
    
    cudaEventRecord(overlapEnd);
    cudaEventSynchronize(overlapEnd);
    cudaEventElapsedTime(&over_elapsed, overlapStart, overlapEnd);
   
    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_Cover[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED for overlap\n");
 
    /*
     * SEQUENTIAL
     */
    cudaEventRecord(transin_or);
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    vecAdd <<<blocksPerGrid,threadsPerBlock>>>(d_A,d_B,d_C);
    printf("CUDA sequential kernel launched \n");
    err = cudaGetLastError();
    Check(err);
    cudaThreadSynchronize();
    err = cudaMemcpy(h_Cseq, d_C,  size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopout_or);
    cudaEventSynchronize(stopout_or);
    cudaEventElapsedTime(&seq_elapsed, transin_or, stopout_or);

    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_Cseq[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED for sequential\n");

    //Timing for the sequential execution
    printf("The sequential elapsed time is %.2f ms\n", seq_elapsed);
	
    //Timing for the overlapped execution
    printf("The overlapped elapsed time is %.2fms\n", over_elapsed);
 

    // Free device global memory
    err = cudaFree(d_A0);
    Check(err);
    err = cudaFree(d_B0);
        Check(err);
    err =cudaFree(d_C0);
        Check(err);
    err =cudaStreamDestroy(stream0);
        Check(err);

    err =cudaFree(d_A1);
        Check(err);
    err =cudaFree(d_B1);
        Check(err);
    err =cudaFree(d_C1);
        Check(err);
    err = cudaStreamDestroy(stream1);
            Check(err);

    err =cudaFree(d_A2);
        Check(err);
    err =cudaFree(d_B2);
        Check(err);
    err =cudaFree(d_C2);
        Check(err);
    err = cudaStreamDestroy(stream2);
            Check(err);

    err = cudaFree(d_A3);
        Check(err);
    err =cudaFree(d_B3);
        Check(err);
    err =cudaFree(d_C3);
	Check(err);
    err =cudaStreamDestroy(stream3);
    Check(err);



    // Free host memory
    free(h_A);
    free(h_B);
    free(h_Cseq);
    free(h_Cover);


    // Reset the device and exit
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    err = cudaDeviceReset();
	Check(err);
	printf("Total number of warps in the kernel = %d\n",((threadsPerBlock)/32)*blocksPerGrid);

	printf("Total number of warps in a block=%d\n",(threadsPerBlock)/32);
    
	printf("Done\n");
    return 0;
}

