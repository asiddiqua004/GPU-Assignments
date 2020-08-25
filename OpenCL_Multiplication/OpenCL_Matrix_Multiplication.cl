#define WIDTH 32 
__kernel void
MatrixMul(__global float* A, __global float* B, __global float* C, __global unsigned long long* runtime)
{

    // TODO : Kernel Function
    //        C = A * B
   ulong  start_time, stop_time;

   asm("mov.u64 %0, %%clock64;" : "=l" (start_time));

    uint tid = get_local_id(0);
    uint col = tid%WIDTH;
    uint row = tid/WIDTH;
    float cout = 0.0;

    for(int i = 0; i < WIDTH; i++)
    {
       cout += A[row*WIDTH+i] * B[col+i*WIDTH];
    }
	C[tid] = cout;

  asm("mov.u64 %0, %%clock64;" : "=l" (stop_time));

    runtime[tid] = (stop_time - start_time);
}

