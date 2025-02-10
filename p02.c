#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define TAM 83886080
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
    float result;
    for (int i = 0; i < tam; i++) {
         result += ((a[i] - b[i]) * (a[i] - b[i]));
    }

    result = sqrtf(result);
}


// Main program
int main(int argc, char *argv[]) {
    double *a, *b;
    struct timeval start, finish;
    double time;
    int N;

    if (argc > 2) {
        printf("Número de parámetros incorrecto. Use ./ distancia N\n");
        exit(EXIT_FAILURE);
    }

    N = argc == 2 ? atoi(argv[1]) : TAM;

    a = (double*) malloc(sizeof(double) * N);
    b = (double*) malloc(sizeof(double) * N);

    init_array(a, N);
    init_array(b, N);
    mult_array(b, N, 2);
    // start timer
    gettimeofday(&start, NULL);
    
    // call the function
    euclidian_distance(a, b, N);

    // stop timer
    gettimeofday(&finish, NULL);
    time = (finish.tv_sec - start.tv_sec + (finish.tv_usec - start.tv_usec) / 1.e6);

    printf("%lf\n", time);

    free(a);
    free(b);

    return EXIT_SUCCESS;
}
