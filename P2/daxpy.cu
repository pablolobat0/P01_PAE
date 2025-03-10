#include <cstdio>
#include <assert.h>
#include <math.h>

#define SIZE 6442450944
#define ALPHA 3
// CUDA kernel to add elements of two arrays
    __global__
void add(int n, double *x, double *y, int alpha)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i]*alpha;
}

int main(void)
{
    double *x, *y;
    int N = SIZE/8;
    // Allocate Unified Memory -- accessible from CPU or GPU
    cudaMallocManaged(&x, SIZE);
    cudaMallocManaged(&y, SIZE);

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = i+1.0f;
        y[i] = 0;
    }

    // Launch kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y,ALPHA);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++)
        assert(y[i] == x[i]*ALPHA);
    printf("Correct!\n");
    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
