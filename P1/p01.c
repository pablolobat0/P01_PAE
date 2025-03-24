#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define TAM 100663296
#define ALPHA 5
/**
 * @brief Template for Labs
 * 
 * PAE [G4011452] Labs
 * Last update: 
 * Issue date:  16/02/2025
 * 
 * Student name: David Carpintero Diaz 
 *
 */

// General utilities
void init_array(double *array, int tam) {
    for (int i = 0; i < tam; i++) {
         array[i] = i;
    }
}


void mult_array(double *array, int tam, int alpha) {
    for (int i = 0; i < tam; i++) {
         array[i] *= alpha;
    }
}

// Custon utilities (in case of need) 

// Implement the exercise in a function here
void euclidian_distance(double *a, double *b, int tam) {
    for (int i = 0; i < tam; i++) {
         a[i] = a[i] + b[i];
    }
}


// Main program
int main(int argc, char *argv[]) {
    double *a, *b;
    struct timeval ex_start, ex_finish, init_start, init_finish;
    double ex_time, init_time;
    int alpha, N;

    if (argc > 3) {
        printf("Número de parámetros incorrecto. Use %s [alpha] [N]\n",argv[0]);
        exit(EXIT_FAILURE);
    }

    alpha = argc >= 2? atoi(argv[1]) : ALPHA;
    N = argc == 3 ? atoi(argv[2]) : TAM;

    gettimeofday(&init_start, NULL);

    a = (double*) malloc(sizeof(double) * N);
    b = (double*) malloc(sizeof(double) * N);

    init_array(a, N);
    init_array(b, N);

    gettimeofday(&init_finish, NULL);
    // ex_start timer
    gettimeofday(&ex_start, NULL);
    
    // call the function
    mult_array(a, N, alpha);
    euclidian_distance(a, b, N);

    // stop timer
    gettimeofday(&ex_finish, NULL);
    ex_time = (ex_finish.tv_sec - ex_start.tv_sec + (ex_finish.tv_usec - ex_start.tv_usec) / 1.e6);
    init_time = (init_finish.tv_sec - init_start.tv_sec + (init_finish.tv_usec - init_start.tv_usec) / 1.e6);
    printf("TIEMPO DE INICIALIZACION: %.10lf\nTIEMPO DE EJECUCION: %.10lf\n", init_time,ex_time);

    free(a);
    free(b);

    return EXIT_SUCCESS;
}
