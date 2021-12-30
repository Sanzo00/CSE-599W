#include "window_sum.cuh"
#include <iostream>
#include <cassert>
#include <stdio.h>

__global__ void windowSumNaiveKernel(const float* A, float* B, int n) {
    int out_index = blockDim.x * blockIdx.x + threadIdx.x;
    int in_index = out_index + RADIUS;
    if (out_index < n) {
        float sum = 0;
        for (int i = -RADIUS; i <= RADIUS; ++i) {
            sum += A[in_index + i];
        }
        B[out_index] = sum;
    }
}

__global__ void windowSumKernel(const float* A, float* B, int n) {
  __shared__ float temp[THREADS_PER_BLOCK + 2 * RADIUS];
  int out_index = blockDim.x * blockIdx.x + threadIdx.x;
  int in_index = out_index + RADIUS;
  int local_index = threadIdx.x + RADIUS;

  // if (threadIdx.x == 0) {
  //   for (int i = 0; i < THREADS_PER_BLOCK + 2 * RADIUS; ++i) {
  //     temp[i] = 0;
  //   }
  // }
  // __syncthreads();

  if (out_index < n) {
    int num = min(THREADS_PER_BLOCK, n - blockIdx.x * blockDim.x);
    temp[local_index] = A[in_index];
    if (threadIdx.x < RADIUS) {
      temp[local_index - RADIUS] = A[in_index - RADIUS];
      // temp[local_index + THREADS_PER_BLOCK] = A[in_index +  THREADS_PER_BLOCK]; // error: not fit last block
      temp[local_index + num] = A[in_index +  num];
    }
    __syncthreads();

    float sum = 0.;
#pragma unroll
    for (int i = -RADIUS; i <= RADIUS; ++i) {
      sum += temp[local_index + i];
    }
    B[out_index] = sum;
  }
}

void windowSumNaiveGpu(const float* A, float* B, int n) {
    float *d_A, *d_B;
    int size = n * sizeof(float);
    cudaMalloc((void**)&d_A, size + 2 * RADIUS * sizeof(float));
    cudaMalloc((void**)&d_B, size);
    
    cudaMemset(d_A, 0, size + 2 * RADIUS * sizeof(float));
    cudaMemcpy(d_A + RADIUS, A, size, cudaMemcpyHostToDevice);
    
    dim3 threads(THREADS_PER_BLOCK, 1, 1);
    dim3 blocks((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
    windowSumNaiveKernel<<<blocks, threads>>> (d_A, d_B, n);
    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
}

void windowSumGpu(const float* A, float* B, int n) {
    float *d_A, *d_B;
    int size = n * sizeof(float);
    
    cudaMalloc((void**)&d_A, size + 2 * RADIUS * sizeof(float));
    cudaMalloc((void**)&d_B, size);
    
    cudaMemset(d_A, 0, size + 2 * RADIUS * sizeof(float));
    cudaMemcpy(d_A + RADIUS, A, size, cudaMemcpyHostToDevice);
    
    dim3 threads(THREADS_PER_BLOCK, 1, 1);
    dim3 blocks((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);
    windowSumKernel<<<blocks, threads>>> (d_A, d_B, n);
    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
}