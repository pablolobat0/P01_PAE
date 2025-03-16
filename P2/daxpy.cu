#include <assert.h>
#include <cstdio>
#include <math.h>
#include <sys/time.h>

#define SIZE 6442450944
#define ALPHA 3
// CUDA kernel to add elements of two arrays
__global__ void add(int n, double *x, double *y, int alpha) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] += x[i] * alpha;
}

int main(void) {
    double *x, *y;
    int N = SIZE / 8;
    struct timeval ex_start, ex_finish, init_start, init_finish;
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    double time = 0;




    gettimeofday(&init_start, NULL);
    // Allocate Unified Memory -- accessible from CPU or GPU
    cudaMallocManaged(&x, SIZE);
    cudaMallocManaged(&y, SIZE);

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = i + 1.0f;
        y[i] = i;
    }
    gettimeofday(&init_finish, NULL);


    
    gettimeofday(&ex_start, NULL);
    add<<<numBlocks, blockSize>>>(N, x, y, ALPHA);
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    gettimeofday(&ex_finish, NULL);

    for (int i = 0; i < N; i++)
        assert(y[i] == x[i] * ALPHA + i);
    printf("NumBlocks=%d ------ BlockSize=%d\n", numBlocks, blockSize);
    
    time = (init_finish.tv_sec - init_start.tv_sec +
            (init_finish.tv_usec - init_start.tv_usec) / 1.e6);

    printf("Reserva de memoria: %.10lf\n", time);

    time = (ex_finish.tv_sec - ex_start.tv_sec +
            (ex_finish.tv_usec - ex_start.tv_usec) / 1.e6);

    printf("Tiempo de Ejecucion: %.10lf\n", time);



    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
