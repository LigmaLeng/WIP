#include<stdio.h>
#include<stdlib.h>

#define N (1<<10)
#define TILE_SIZE (1<<4)
#define SHMEM_SIZE (TILE_SIZE * TILE_SIZE * sizeof(int))


void errorGuard(cudaError_t candidate){
	if (candidate != cudaSuccess){
		fprintf(stderr, "CUDA runtime error: %s\n", cudaGetErrorString(candidate));
        exit(1);
	}
}

void initMatrices(int* a, int* b){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            int idx = i * N + j;
            a[idx] = (int) rand()%100;
            b[idx] = (int) rand()%100;
        }
    }
}

void verifyMM(int* a, int* b, int* c){
    int vecMulValue;
    bool err = false;
    for (int row = 0; row < N; row++){
        for (int col = 0; col < N; col++){
            vecMulValue = 0;
            for (int k = 0; k < N; k++){
               vecMulValue += a[row * N + k] * b[k * N + col];
            }
            if (c[row * N + col] != vecMulValue){
                printf("Element mismatch at c[%d][%d]\n", row, col);
                err = true;
                break;
            }
        }
    }
    if (!err){
        printf("s u c c e s s ! ! !\n");
    }
}

__global__ void gpuMM(int* a, int* b, int* c){
    __shared__ int cache_a[SHMEM_SIZE];
    __shared__ int cache_b[SHMEM_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = ty + blockDim.y * blockIdx.y;
    int col = tx + blockDim.x * blockIdx.x;

    int temp = 0;

    for (int i = 0; i < N / TILE_SIZE; i++){
        cache_a[ty * TILE_SIZE + tx] = a[row * N + i * blockDim.x + tx];
        cache_b[ty * TILE_SIZE + tx] = b[col + i * blockDim.y * N + ty * N];

        // synchronize all threads in "tile"/grid before cache-ing values to result matrix
        __syncthreads();


        for (int k = 0; k < TILE_SIZE; k++){
            temp += cache_a[ty * blockDim.y + k] * cache_b[k * blockDim.x + tx];
        }

        // synchronize threads again to ensure earlier threads don't return and wipe values in shared memory
        __syncthreads();
    }

    c[row * N + col] = temp;
    return;
}

int main(){
    int *a, *b, *c, *device_a, *device_b, *device_c;
    size_t size = (N) * (N) * sizeof(int);
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    cudaMalloc(&device_a, size);
    cudaMalloc(&device_b, size);
    cudaMalloc(&device_c, size);

    initMatrices(a, b);

    cudaMemcpy(device_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, size, cudaMemcpyHostToDevice);

    dim3 bDim(TILE_SIZE, TILE_SIZE);
    dim3 gDim(N/TILE_SIZE, N/TILE_SIZE);

    gpuMM<<<gDim, bDim>>>(device_a, device_b, device_c);
    errorGuard(cudaGetLastError());
    errorGuard(cudaDeviceSynchronize());

    cudaFree(device_a);
    cudaFree(device_b);

    c = (int *)malloc(size);
    cudaMemcpy(c, device_c, size, cudaMemcpyDeviceToHost);

    verifyMM(a, b, c);

    free(a);
    free(b);
    free(c);

    cudaFree(device_c);

    errorGuard(cudaGetLastError());
    errorGuard(cudaDeviceSynchronize());
}
