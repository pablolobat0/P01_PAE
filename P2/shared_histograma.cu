// General utilities
#include <stdlib.h>
#include <stdio.h>
// Math functions
#include <math.h>

// The file loadPGM.h will be used for defining load and export functions
#include "../pgmio.h"

#define SIZE 256
#define BLOCK_SIZE 1024

__global__ void histograma(unsigned char * d_xu8, int* v_hist, int dim){
    __shared__ int histo_s[SIZE];

    int tid = threadIdx.x;
    if(tid<SIZE)histo_s[tid]=0;
    __syncthreads();
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    while (i<dim){ //loop por si el numero de hilos es menor que numero de pixeles
        atomicAdd(&histo_s[d_xu8[i]],1); //usamos atomic para evitar carreras criticas
        i+= blockIdx.x*blockDim.x;
    }
    __syncthreads();

    // Un solo hilo por nivel de intensidad actualiza la memoria global
    if (tid < SIZE) {
        atomicAdd(&v_hist[tid], histo_s[tid]);
    }
}


int main(int argc, char *argv[])
{ 
    // check for arguments
    if (argc < 2) {
    	printf("Use %s file.pgm\n", argv[0]);
        exit(-1);
    }

    // image width x height
    int w, h;

    // Load pgm image
    unsigned char* h_xu8 = loadPGMu8(argv[1], &w, &h);

    int h_hist[SIZE]= {0};

    unsigned char* v_xu8;
    int* v_hist;

    //reservamos memoria para las variables en memoria GLOBAL
    cudaMalloc(&v_xu8,w*h*sizeof(unsigned char));
    cudaMalloc(&v_hist, SIZE*sizeof(int));

    //inicalizamos variables en device
    cudaMemset(v_hist,0,SIZE*sizeof(int));
    cudaMemcpy(v_xu8,h_xu8,w*h*sizeof(unsigned char),cudaMemcpyHostToDevice);
  
    int blockDim,blockCount;
    blockDim = BLOCK_SIZE;
    blockCount = (w * h + blockDim - 1) / blockDim;
    histograma<<<blockCount,blockDim>>>(v_xu8, v_hist, w*h);
    cudaDeviceSynchronize();
     
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }


    cudaMemcpy(h_hist,v_hist,SIZE*sizeof(int),cudaMemcpyDeviceToHost);

    free(h_xu8);
    cudaFree(v_xu8);
    cudaFree(v_hist);

    
    return 0;

}
