// OpenMP header
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "pgmio.h"
float *init_kernel(int size);

int main(int argc, char* argv[])
{
    // check for arguments
    if (argc < 2) {
    	printf("Use %s file.pgm [KERNEL_TAM]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    float *kernel;
    int i,j,tid,size=3; 
    // image width x height
    int width, height;

    // Load pgm image
    float* xu8 = loadPGM32(argv[1], &width, &height);
    float* result = calloc(width*height, sizeof(float));
    kernel = init_kernel(size);

    // Beginning of parallel region
    #pragma omp parallel private(tid) shared(width,height,size,result)
    {
        tid = omp_get_thread_num();
        #pragma omp for private(j) 
        for (i=0;i<20;i++){
            for(j=0;j<20;j++){
                //Funcion de convolucion

                int mid = size/ 2;
                int start_row = i - mid;
                int start_col = j - mid;

                float sum = 0;

                for (int r = start_row, a = 0; a < size; r++, a++) {
                    for (int c = start_col, b = 0; b < size; c++, b++) {
                        if (r >= 0 && r < height && c >= 0 && c < width) {
                            sum += xu8[r*width + c] * kernel[a*size+b];
                        }
                    }
                }
                result[i*width+j] = sum;
            }
        }
    }

    savePGM32("salida.pgm",result,width,height);

    free(result);
    free(kernel);
    free(xu8);
}

float *init_kernel(int size){
    float valores[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
    float *ker = (float*)malloc(size*size*sizeof(float));
    for(int i = 0;i<size*size;i++){
        ker[i] = valores[i];
    }
    return ker;
}

