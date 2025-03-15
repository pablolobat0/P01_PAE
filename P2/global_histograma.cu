// General utilities
#include <stdio.h>
#include <stdlib.h>
// Math functions
#include <math.h>

// The file loadPGM.h will be used for defining load and export functions
#include "../pgmio.h"

#define SIZE 256
#define BLOCK_SIZE 256

__global__ void histograma(unsigned char *d_xu8, int *v_hist, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // hay que poner un mutex, si no va a haber carreras criticas
    for (int i = idx; i < dim; i += stride) {
        atomicAdd(&v_hist[d_xu8[i]], 1);
    }
}

int main(int argc, char *argv[]) {
    int blockDim, blockCount;
    int width, height;
    int h_hist[SIZE] = {0};
    unsigned char *h_xu8, *v_xu8;
    int *valores_hist;

    // check for arguments
    if (argc < 2) {
        printf("Use %s file.pgm\n", argv[0]);
        exit(-1);
    }

    // Load pgm image
    h_xu8 = loadPGMu8(argv[1], &width, &height);

    // reservamos memoria para las variables en memoria GLOBAL
    cudaMalloc(&v_xu8, width * height * sizeof(unsigned char));
    cudaMalloc(&valores_hist, SIZE * sizeof(int));

    // inicalizamos variables en device
    cudaMemset(valores_hist, 0, SIZE * sizeof(int));
    cudaMemcpy(v_xu8, h_xu8, width * height * sizeof(unsigned char),
               cudaMemcpyHostToDevice);

    blockDim = BLOCK_SIZE;
    blockCount = (width * height + blockDim - 1) / blockDim;
    histograma<<<blockCount, blockDim>>>(v_xu8, valores_hist, width * height);

    cudaMemcpy(h_hist, valores_hist, SIZE * sizeof(int),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; i++) {
        printf("%i ", h_hist[i]);
    }
    printf("\n");

    free(h_xu8);
    cudaFree(v_xu8);
    cudaFree(valores_hist);

    return 0;
}
