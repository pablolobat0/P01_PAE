// OpenMP header
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "pgmio.h"


double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

float *init_kernel(int size);

int main(int argc, char* argv[])
{
    // check for arguments
    if (argc < 4) {
    	printf("Use %s file.pgm threads_num chunk\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    float *kernel;
    int i,j,tid,threads_num = 8,size=3; 
    int chunk = 0;
    // image width x height
    int width, height;

    double init_start = get_time();
    // Load pgm image
    float* xu8 = loadPGM32(argv[1], &width, &height);
    float* result = calloc(width*height, sizeof(float));
    kernel = init_kernel(size);
    threads_num = atoi(argv[2]);
    chunk = atoi(argv[3]);
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    double mid= get_time();
    // Beginning of parallel region
    #pragma omp parallel private(tid) shared(width,height,size,result,chunk) num_threads(threads_num)
    {
        tid = omp_get_thread_num();
        #pragma omp for private(j) schedule(dynamic,chunk)
        for (i=0;i<height;i++){
            for(j=0;j<height;j++){
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

    double ex_end= get_time();
    savePGM32("salida.pgm",result,width,height);

    printf("Initialization Time: %f seconds\n", mid- init_start);
    printf("Execution Time: %f seconds\n", ex_end - mid);

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

