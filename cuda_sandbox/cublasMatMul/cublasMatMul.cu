#include<stdio.h>
#include<stdlib.h>
#include<cublas_v2.h>

#define N (1<<10)
#define TILE_SIZE (1<<4)
#define SHMEM_SIZE (TILE_SIZE * TILE_SIZE * sizeof(int))

inline bool cudaGuard(cudaError_t candidate){
    if (candidate != cudaSuccess){
        fprintf("CUDA Runtime Error: %s\n", cudaGetErrorString(candidate));
        return true;
    }
    return false;
}

inline bool cublasGuard(cublasStatus_t candidate){
    if (candidate != CUBLAS_STATUS_SUCCESS){
        fprintf("cuBLAS Runtime Error: %s\n", cublasGetStatusString(candidate));
        return true;
    }
    return false;
}


void initMatrices(int* a, int* b){
    for (int i = 1; i <= N; i++){
        for (int j = 1; j <= N; j++){
            // cuBLAS arrays arranged in Fortran Column Major order with 1-based indexing
        }
    }
}

int main(){
    //TODO
    int *a, *b, *c, *device_a, *device_b, *device_c;
    size_t size = (N) * (N) * sizeof(int);
    cublasHandle_t handle;

    a = (int *)malloc(size);
    b = (int *)malloc(size);
    cudaMalloc(&device_a, size);
    cudaMalloc(&device_b, size);
    cudaMalloc(&device_c, size);

    initMatrices(a, b);
}
