
// Matrix Multiplication in CUDA

#include <stdio.h>
//#include <string.h>
//#include <assert.h>
//#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#define WIDTH 128
extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

// FILL HERE: define constant variable
const char* cublasGetErrorString(cublasStatus_t status)
{
switch(status)
{
case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
}
return "unknown error";
}

//Kernel
__global__ void
MatrixMul(float* A, float* B, float* C)
{
    unsigned long long start_time= clock64();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row = tid/WIDTH;
    int col = tid%WIDTH;
    float cl=0.0;

    for(int k = 0; k < WIDTH; k++)
    {
    cl += A[row*WIDTH+k] * B[k*WIDTH + col];
    }

    C[row*WIDTH+ col] = cl;

}

 // Host main routine

int main(void)
{
    // Error code to check return values for CUDA calls and CUBLAS calls
    cudaError_t err = cudaSuccess;
    cublasStatus_t status=CUBLAS_STATUS_SUCCESS;

    //Handle creation
    cublasHandle_t handle;
    status = cublasCreate(&handle);

    //Events
    cudaEvent_t Start, End, Start_ker, End_ker;
    cudaEventCreate(&Start);cudaEventCreate(&End);
    cudaEventCreate(&Start_ker);cudaEventCreate(&End_ker);
    float milisec =0, milisec_ker =0;

    // Print the matrix size to be used, and compute its size
    int size = WIDTH*WIDTH*sizeof(float);
    printf("[MatrixMul of %d x %d elements]\n", WIDTH, WIDTH);

    // Allocate the host input matrix h_A, h_B and h_C
    float  *h_A = (float *)malloc(size);
    float  *h_B = (float *)malloc(size);
    float  *h_C = (float *)malloc(size);
    float  *h_C_ker = (float *)malloc(size);

    // Allocate the host matrix for compute check 
    float  *reference = (float *)malloc(size);
    
    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL || h_C_ker == NULL || reference == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrices!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input matrices
    for (int i = 0; i < WIDTH; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
            h_A[i*WIDTH + j] = j/10;
            h_B[i*WIDTH + j] = i/10;
        }
    }
    /*printf("Matrix A\n");
    for (int i = 0; i < WIDTH; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
            printf("%f\t",h_A[i*WIDTH + j]);
        }
        printf("\n");
    }
    printf("\n");

    printf("Matrix  B\n");
    for (int i = 0; i < WIDTH; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
        printf("%f\t",h_B[i*WIDTH + j]);
        }
        printf("\n");
    }
    printf("\n");*/

    memset(h_C, 0, size);
    memset(h_C_ker, 0, size);
    memset(reference, 0, size);

    // compute the matrix multiplication on the CPU for comparison
    computeGold(reference, h_A, h_B, WIDTH, WIDTH, WIDTH);

	// Allocate device input matrices
	float* d_A = NULL;	
    err = cudaMalloc((void**)&d_A, size); 
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	float* d_B = NULL;	
    err = cudaMalloc((void**)&d_B, size); 
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float* d_A_ker = NULL;
    err = cudaMalloc((void**)&d_A_ker, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float* d_B_ker = NULL;
    err = cudaMalloc((void**)&d_B_ker, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	// Allocate the device output matrix
	float* d_C = NULL;
    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float* d_C_ker = NULL;
    err = cudaMalloc((void**)&d_C_ker, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* Initialize the device matrices with the host matrices */
    err = cudaMemcpy(d_A_ker, h_A, size, cudaMemcpyHostToDevice);// FILL HERE
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B_ker, h_B, size, cudaMemcpyHostToDevice);// FILL HERE
    //err = ;// FILL HERE
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    status = cublasSetMatrix(WIDTH,WIDTH, sizeof(float), h_A, WIDTH, d_A, WIDTH);
    if (status!= CUBLAS_STATUS_SUCCESS)
    {
        printf("error A %s\n", cublasGetErrorString(status));
    }

    status = cublasSetMatrix(WIDTH,WIDTH,sizeof(float),h_B,WIDTH,d_B ,WIDTH);
    if ( status!= CUBLAS_STATUS_SUCCESS)
    {
        printf("error B %s\n", cublasGetErrorString(status));
    }

    int blocksPerGrid = ((WIDTH*WIDTH)>1024) ? (((WIDTH*WIDTH)>4096) ? 16 : 4)  : 1;
    int threadsPerBlock = ((WIDTH*WIDTH)>1024) ? 1024 : (WIDTH*WIDTH) ;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    cudaEventRecord(Start_ker);
    MatrixMul <<<blocksPerGrid, threadsPerBlock>>> (d_A_ker, d_B_ker, d_C_ker);
    cudaEventRecord(End_ker);
    cudaEventSynchronize(End_ker);
    cudaEventElapsedTime(&milisec_ker, Start_ker, End_ker);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch matrixMul kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaThreadSynchronize();

    // Copy the device result matrix in device memory to the host result matrix in host memory.
    err = cudaMemcpy(h_C_ker, d_C_ker, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
    fprintf(stderr, "Failed to copy matrix C from device to host (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
    }
    cudaThreadSynchronize();
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cudaEventRecord(Start);
    status=cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,WIDTH,WIDTH,WIDTH, &alpha,d_A,WIDTH,d_B,WIDTH, &beta,d_C,WIDTH);
    cudaEventRecord(End);
    cudaEventSynchronize(End);
    cudaEventElapsedTime(&milisec, Start, End);

    if ( status!= CUBLAS_STATUS_SUCCESS)
    {
    printf("error B %s\n", cublasGetErrorString(status));
    }
    // Copy the device result matrix in device memory to the host result matrix in host memory.

    status=cublasGetMatrix(WIDTH,WIDTH,sizeof(float),d_C,WIDTH,h_C,WIDTH);
    if ( status!= CUBLAS_STATUS_SUCCESS)
    {
    printf("error B %s\n", cublasGetErrorString(status));
    }

    // Verify that the result matrix is correct
    bool res = 1;
	for (int i = 0; i < WIDTH*WIDTH; i++)
	{
		float diff = fabs(reference[i] - h_C[i]);
		if(diff > 0.001f)
		{
			res = 0;
           // printf("Broken link at reference[%d] - h_C[%d] = %f - %f", i, i, reference[i], h_C[i]);
			break;
		}
	}
	printf("Test for cublas %s\n", (res == 1) ? "PASSED" : "FAILED");

    bool res1 = 1;
    for (int i = 0; i < WIDTH*WIDTH; i++)
    {
        float diff = fabs(reference[i] - h_C_ker[i]);
        if(diff > 0.001f)
        {
            res1 = 0;
             printf("Broken link at reference[%d] - h_C_ker[%d] = %f - %f", i, i, reference[i], h_C_ker[i]);
            break;
        }
    }
    printf("Test for kernel %s\n", (res1 == 1) ? "PASSED" : "FAILED");

    //Timing for the execution
    printf("The cublas elapsed time is %0.9fms\n", milisec);
    printf("The kernel elapsed time is %0.9fms\n", milisec_ker);

	// Free device global memory

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    status = cublasDestroy(handle);

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);
	free(reference);


	return 0;
}

void
computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < wA; ++k) {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
        }

}
