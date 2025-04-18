// General utilities
#include <stdio.h>
#include <stdlib.h>
// Math functions
#include <math.h>
#include <sys/time.h>
// The file loadPGM.h will be used for defining load and export functions
#include "../pgmio.h"

#define SIZE 256
#define BLOCK_SIZE 256

__global__ void histograma(unsigned char *d_xu8, int *v_hist, int dim) {
    __shared__ int histo_s[SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Inicilizacion de la memoria compartida
    for (int i = tid; i < SIZE; i += blockDim.x) {
        histo_s[i] = 0;
    }

    __syncthreads();

    // loop por si el numero de hilos es menor que numero de pixeles
    for (int i = idx; i < dim; i += stride) {
        // usamos atomic para evitar carreras criticas
        atomicAdd(&histo_s[d_xu8[i]], 1);
    }

    __syncthreads();

    // Un solo hilo por nivel de intensidad actualiza la memoria global
    for (int i = tid; i < SIZE; i += blockDim.x) {
        atomicAdd(&v_hist[i], histo_s[i]);
    }
}

int main(int argc, char *argv[]) {
    unsigned char *h_xu8, *v_xu8;
    int width, height;
    int h_hist[SIZE] = {0};
    int *v_hist;
    int blockDim, blockCount;
    struct timeval ex_start, ex_finish, init_start, init_finish;
    double time = 0;
    // check for arguments
    if (argc < 2) {
        printf("Use %s file.pgm\n", argv[0]);
        exit(-1);
    }

    gettimeofday(&init_start, NULL);
    // Load pgm image
    h_xu8 = loadPGMu8(argv[1], &width, &height);

    // reservamos memoria para las variables en memoria GLOBAL
    cudaMalloc(&v_xu8, width * height * sizeof(unsigned char));
    cudaMalloc(&v_hist, SIZE * sizeof(int));

    // inicalizamos variables en device
    cudaMemset(v_hist, 0, SIZE * sizeof(int));
    cudaMemcpy(v_xu8, h_xu8, width * height * sizeof(unsigned char),
               cudaMemcpyHostToDevice);

    blockDim = BLOCK_SIZE;
    blockCount = (width * height + blockDim - 1) / blockDim;
    gettimeofday(&init_finish, NULL);


    gettimeofday(&ex_start, NULL);
    histograma<<<blockCount, blockDim>>>(v_xu8, v_hist, width * height);
    cudaDeviceSynchronize();
    gettimeofday(&ex_finish, NULL);

    cudaMemcpy(h_hist, v_hist, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    for (int i = 0; i < SIZE; i++) {
        printf("%i ", h_hist[i]);
    }
    printf("\n");
    printf("NumBlocks=%d ------ BlockSize=%d\n", blockCount, blockDim);
    
    time = (init_finish.tv_sec - init_start.tv_sec +
            (init_finish.tv_usec - init_start.tv_usec) / 1.e6);

    printf("Reserva de memoria: %.10lf\n", time);

    time = (ex_finish.tv_sec - ex_start.tv_sec +
            (ex_finish.tv_usec - ex_start.tv_usec) / 1.e6);

    printf("Tiempo de Ejecucion: %.10lf\n", time);




    free(h_xu8);
    cudaFree(v_xu8);
    cudaFree(v_hist);

    return 0;
}
