#ifndef VEC_ADD_KERNEL
#define VEC_ADD_KERNEL

#include <cuda.h>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK    512

__global__ void vecAddKernel(const float* A, const float* B, float* C, int n);
void vecAddGpu(const float* A, const float* B, float* C, int n);

#endif