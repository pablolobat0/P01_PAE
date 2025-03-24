#include <nppi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "pgmio.h"

#define MODO_PRUEBA 1
#define MOD 10

void initKernel(float** kernel, int size);
void initKernelPrueba(float** kernel);

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Use %s file.pgm [KERNEL_TAM]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    float *kernel;
    int size = 3;
    if (argc == 3) {
        size = atoi(argv[2]);
    }
    
    int width, height;
    float* h_src = loadPGM32(argv[1], &width, &height);
    float* h_dst = (float*)malloc(sizeof(float) * width * height);
    
    if (MODO_PRUEBA) {
        size = 3;
        initKernelPrueba(&kernel);
    } else {
        initKernel(&kernel, size * size);
    }

    double init_start = get_time();
    float *d_src, *d_dst, *d_kernel;
    cudaMalloc((void**)&d_src, sizeof(float) * width * height);
    cudaMalloc((void**)&d_dst, sizeof(float) * width * height);
    cudaMalloc((void**)&d_kernel, sizeof(float) * size * size);

    cudaMemcpy(d_src, h_src, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, sizeof(float) * size * size, cudaMemcpyHostToDevice);
    double init_end = get_time();

    NppiSize oROI;
    oROI.width = width;
    oROI.height = height;
    NppiSize oMaskSize = {size, size};
    NppiPoint oAnchor = {size / 2, size / 2};

    double ex_start = get_time();
    NppStatus status = nppiFilter_32f_C1R(
        d_src, width * sizeof(Npp32f),
        d_dst, width * sizeof(Npp32f),
        oROI,
        d_kernel, oMaskSize, oAnchor
    );
    cudaDeviceSynchronize();
    double ex_end = get_time();

    cudaMemcpy(h_dst, d_dst, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_kernel);
    free(h_src);
    free(h_dst);
    free(kernel);

    printf("Initialization Time: %f seconds\n", init_end - init_start);
    printf("Execution Time: %f seconds\n", ex_end - ex_start);

    return 0;
}

void initKernel(float** kernel, int size) {
    (*kernel) = (float*)malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++) {
        (*kernel)[i] = (rand() % MOD) / (float)MOD;
    }
}

void initKernelPrueba(float** kernel) {
    (*kernel) = (float*)malloc(sizeof(float) * 9);
    float valores[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
    for (int i = 0; i < 9; i++) {
        (*kernel)[i] = valores[i];
    }
}
