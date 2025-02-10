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
 * Issue date:  30/01/2022
 * 
 * Student name: Pablo Lobato Rey
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
    struct timeval start, finish;
    double time;
    int alpha, N;

    if (argc > 3) {
        printf("Número de parámetros incorrecto. Use ./ DAXPY alpha N\n");
        exit(EXIT_FAILURE);
    }

    alpha = argc == 2 || argc == 3 ? atoi(argv[1]) : ALPHA;
    N = argc == 3 ? atoi(argv[2]) : TAM;

    a = (double*) malloc(sizeof(double) * N);
    b = (double*) malloc(sizeof(double) * N);

    init_array(a, N);
    init_array(b, N);
    mult_array(a, N, alpha);
    // start timer
    gettimeofday(&start, NULL);
    
    // call the function
    euclidian_distance(a, b, N);

    // stop timer
    gettimeofday(&finish, NULL);
    time = (finish.tv_sec - start.tv_sec + (finish.tv_usec - start.tv_usec) / 1.e6);

    printf("%.10lf\n", time);

    free(a);
    free(b);

    return EXIT_SUCCESS;
}
