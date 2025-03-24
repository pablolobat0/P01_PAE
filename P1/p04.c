#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include "../hist&conv/pgmio.h"

#define ROWS 3
#define COLS 3

#define MOD 10
#define MODO_PRUEBA 1 //CAMBIAR A 0 SI SE QUIERE PROBAR CON UN KERNEL  DISTINTO AL DEL ENUNCIADO

void initKernel(int**kernel,int size);
void initKernelPrueba(int**kernel);

float convulucion(float *matriz, int* kernel, int width, int height, int matrix_row, int matrix_col,int kernel_size) {
    int mid = kernel_size/ 2;
    int start_row = matrix_row- mid;
    int start_col = matrix_col- mid;

    float sum = 0;

    for (int i = start_row, a = 0; a < kernel_size; i++, a++) {
        for (int j = start_col, b = 0; b < kernel_size; j++, b++) {
            if (i >= 0 && i < height && j >= 0 && j < width) {
                sum += matriz[i *height + j] * kernel[a*kernel_size+b];
            }
        }
    }

    return sum;
}

int main(int argc, char *argv[]) {
    
    // check for arguments
    if (argc < 2) {
    	printf("Use %s file.pgm [KERNEL_TAM]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    int *kernel;
    int size = 3;
    struct timeval ex_start, ex_finish, init_start, init_finish;
    double ex_time, init_time;

    if(argc ==3){
        size = atoi(argv[2]);
    }
    gettimeofday(&init_start, NULL);

    if(MODO_PRUEBA){
        size = 3;
        initKernelPrueba(&kernel);
    }else{
        initKernel(&kernel,size);
    }
    /*printf("Size: %d*%d\n",size,size);
    for(int i = 0;i<size;i++){
        for(int j = 0;j<size;j++){
            printf("%d ",kernel[i*size +j]);
        }
        printf("\n");
    }*/
    
    
    // image width x height
    int width, height;

    // Load pgm image
    float* xu8 = loadPGM32(argv[1], &width, &height);
    float* salida = malloc(sizeof(float) * width * height);

    gettimeofday(&init_finish, NULL);

    gettimeofday(&ex_start, NULL);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          salida[i * height + j] = convulucion(xu8, kernel, width, height, i, j,size );
        }
    }
    gettimeofday(&ex_finish, NULL);

    savePGM32("hola.pgm", salida,  width,  height);

    ex_time = (ex_finish.tv_sec - ex_start.tv_sec + (ex_finish.tv_usec - ex_start.tv_usec) / 1.e6);
    init_time = (init_finish.tv_sec - init_start.tv_sec + (init_finish.tv_usec - init_start.tv_usec) / 1.e6);
    printf("TIEMPO DE INICIALIZACION: %.10lf\nTIEMPO DE EJECUCION: %.10lf\n", init_time,ex_time);


    free(xu8);
    free(salida);
    free(kernel);

    return EXIT_SUCCESS;
}


void initKernel(int**kernel,int size){
    (*kernel)=(int*)malloc(sizeof(int)*size*size);
    for(int i = 0;i<size;i++){
        for(int j = 0;j<size;j++){
            (*kernel)[i*size +j] = rand()%MOD;
        }
    }
}

void initKernelPrueba(int** kernel) {
    (*kernel)=(int*)malloc(sizeof(int)*9);

    int valores[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
    for (int i = 0; i < 9; i++) {
        (*kernel)[i] = valores[i];
    }
}


