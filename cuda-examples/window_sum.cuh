#ifndef WINDOW_SUM_CUH
#define WINDOW_SUM_CUH

#include <cuda.h>
#include <cuda_runtime.h>

#define RADIUS              3
#define THREADS_PER_BLOCK    512

__global__ void windowSumNaiveKernel(const float* A, float* B, int n);
__global__ void windowSumKernel(const float* A, float* B, int n);

void windowSumGpu(const float* A, float* B, int n);
void windowSumNaiveGpu(const float* A, float* B, int n);

#endif