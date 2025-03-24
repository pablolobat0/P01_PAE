#include <cassert>
#include <stdio.h>
#include <cuda_runtime.h>

#define N 8192

__global__ void initKernel(int *x) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        x[i] = i;
    }
}

__global__ void reverseKernel(const int *x, int *y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        y[N - 1 - i] = x[i];
    }
}

int main() {
    int *d_x, *d_y;
    int h_y[N];

    cudaError_t err = cudaMalloc((void**)&d_x, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error al asignar memoria para d_x: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc((void**)&d_y, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error al asignar memoria para d_y: %s\n", cudaGetErrorString(err));
        cudaFree(d_x);
        return 1;
    }

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    initKernel<<<numBlocks, blockSize>>>(d_x);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error al lanzar initKernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    reverseKernel<<<numBlocks, blockSize>>>(d_x, d_y);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error al lanzar reverseKernel: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    err = cudaMemcpy(h_y, d_y, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error al copiar datos de GPU a CPU: %s\n", cudaGetErrorString(err));
        return 1;
    }

    for (int i = 0; i < N; i++) {
        assert(h_y[i] == (N - 1 - i));    
    }
    
    printf("Correct!\n");

    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
