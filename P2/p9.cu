#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// Kernel para inicializar la matriz
__global__ void initMatrixKernel(int *matrix, int width, int height) {
    // Cálculo de índices: cada hilo se encarga de una posición de la matriz
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        // Inicialización secuencial: fila * N + columna
        matrix[row * width + col] = row * width + col;
    }
}

int main() {
    int width = 16384;
    int height = 16384;
    size_t size = width * height * sizeof(int);

    // Reserva de memoria en el dispositivo
    int *d_matrix;
    cudaError_t err = cudaMalloc((void**)&d_matrix, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error al asignar memoria en el dispositivo: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // ---------------------------
    // Ejecución con bloques de tamaño de un warp (32 hilos)
    
    initMatrixKernel<<<width * height / 32, 32>>>(d_matrix, width, height);
    cudaDeviceSynchronize();
    printf("Correct\n");

    // Ejecución con bloques de tamaño maximo (1024 hilos)
    initMatrixKernel<<<width * height / 1024, 1024>>>(d_matrix, width, height);
    cudaDeviceSynchronize();
    printf("Correct\n");
    cudaFree(d_matrix);

    return 0;
}

