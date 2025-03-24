#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define TAM 83886080
/**
 * @brief Template for Labs
 *
 * PAE [G4011452] Labs
 * Last update:
 * Issue date:  07/02/2025
 *
 * Student name: Pablo Lobato Rey y David Carpintero Diaz
 *
 */

// General utilities
void init_array(double *array, int tam, int alpha) {
    for (int i = 0; i < tam; i++) {
        array[i] = i * alpha;
    }
}

// Custon utilities (in case of need)

// Implement the exercise in a function here
float euclidian_distance(double *a, double *b, int tam) {
    float result = 0;
    for (int i = 0; i < tam; i++) {
        result += ((a[i] - b[i]) * (a[i] - b[i]));
    }

    return sqrtf(result);
}

// Main program
int main(int argc, char *argv[]) {
    double *a, *b;
    struct timeval start, finish;
    double time;
    float distancia = 0;
    int N;

    if (argc > 2) {
        printf("Número de parámetros incorrecto. Use ./ distancia N\n");
        exit(EXIT_FAILURE);
    }

    N = argc == 2 ? atoi(argv[1]) : TAM;
    // start timer
    gettimeofday(&start, NULL);

    a = (double *)malloc(sizeof(double) * N);
    b = (double *)malloc(sizeof(double) * N);

    if (a == NULL || b == NULL) {
        printf("Error: No se pudo asignar memoria.\n");
        exit(EXIT_FAILURE);
    }

    init_array(a, N, 2);
    init_array(b, N, 1);

    // stop timer
    gettimeofday(&finish, NULL);
    time = (finish.tv_sec - start.tv_sec +
            (finish.tv_usec - start.tv_usec) / 1.e6);

    printf("Reserva de memoria: %.10lf\n", time);

    // start timer
    gettimeofday(&start, NULL);

    // call the function
    distancia = euclidian_distance(a, b, N);

    // stop timer
    gettimeofday(&finish, NULL);
    time = (finish.tv_sec - start.tv_sec +
            (finish.tv_usec - start.tv_usec) / 1.e6);

    printf("Ejecucion: %.10lf\n", time);
    // printf("Distancia: %f\n", distancia);

    free(a);
    free(b);

    return EXIT_SUCCESS;
}
