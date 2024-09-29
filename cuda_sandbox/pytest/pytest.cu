#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>


void errorGuard(cudaError_t candidate){
	if (candidate != cudaSuccess){
		fprintf(stderr, "%s\n", cudaGetErrorString(candidate));
	}
}

__global__ void helloWorldGPU(){
	printf("IndexInGrid %d: Hello World\n", threadIdx.x + (blockIdx.x * blockDim.x));
}


// extern "C"{
// void hwg(){
int main(){
	helloWorldGPU<<<5,1>>>();
	cudaDeviceSynchronize();
	exit(0);
// }
}
