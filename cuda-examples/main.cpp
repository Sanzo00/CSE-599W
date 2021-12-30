#include "vector_add.cuh"
#include "window_sum.cuh"
#include <iostream>
#include <cassert>

void vecAddCpu(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; ++i) {
        C[i] = A[i] + B[i];
    }
}

void test_vector_add() {
    srand(time(0));

    int n = INT_MAX / sizeof(float); 
    float* A = new float[n];
    float* B = new float[n];
    float* C_cpu = new float[n];
    float* C_gpu = new float[n];
    
    for (int i = 0; i < n; ++i) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    vecAddCpu(A, B, C_cpu, n);
    vecAddGpu(A, B, C_gpu, n);

    for (int i = 0; i < n; ++i) {
        assert(C_cpu[i] == C_gpu[i]);
    }
    std::cout << "vector add is right!" << std::endl;

    delete[] A;
    delete[] B;
    delete[] C_cpu;
    delete[] C_gpu;
}

void windowSumCpu(const float* A, float* B, int n) {
    for (int i = 0; i < n; ++i) {
        B[i] = 0;
        for (int j = std::max(0, i - RADIUS); j <= std::min(n-1, i + RADIUS); ++j) {
            B[i] += A[j];
        }
    }
}

void test_window_sum() {

    int n = (INT_MAX - 2 * RADIUS * sizeof(float)) / 4; // max n (cudaMalloc (void** devPtr, size_t size) )

    float* A = new float[n];
    float* B_cpu = new float[n];
    float* B_gpu1 = new float[n];
    float* B_gpu2 = new float[n];

    srand(time(0));
    for (int i = 0; i < n; ++i) {
        A[i] = rand() % 10;
    }

    windowSumCpu(A, B_cpu, n);
    windowSumNaiveGpu(A, B_gpu1, n);
    windowSumGpu(A, B_gpu2, n);

    for (int i = 0; i < n; ++i) {
        assert(B_cpu[i] == B_gpu1[i]);
        assert(B_cpu[i] == B_gpu2[i]);
    }
    std::cout << "window sum is right!" << std::endl;

    delete[] A;
    delete[] B_cpu;
    delete[] B_gpu1;
    delete[] B_gpu2;
}

int main() {
    
    // test_vector_add();

    test_window_sum();

    return 0;
}