#include "./Lab1_material/pgmio.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZE 256
/**
 * @brief Template for Labs
 *
 * PAE [G4011452] Labs
 * Last update:
 * Issue date:  30/01/2022
 *
 * Student name: Pablo Lobato Rey y David Carpintero Diaz
 *
 */

// General utilities

// Custon utilities (in case of need)

// Implement the exercise in a function here
void histogram(unsigned char *image, int height, int width, int *count) {
    for (int i = 0; i < height * width; i++) {
        count[image[i]] += 1;
    }
}

// Main program
int main(int argc, char *argv[]) {
    int width, height;
    unsigned char *image;
    struct timeval start, finish;
    double time;
    int *count;
    // check for arguments
    if (argc < 2) {
        printf("Use %s file.pgm\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    // start timer
    gettimeofday(&start, NULL);

    count = (int *)calloc(SIZE, sizeof(int));

    if (count == NULL) {
        printf("Error: No se pudo asignar memoria.\n");
        exit(EXIT_FAILURE);
    }

    // Load pgm image
    image = loadPGMu8(argv[1], &width, &height);

    // stop timer
    gettimeofday(&finish, NULL);
    time = (finish.tv_sec - start.tv_sec +
            (finish.tv_usec - start.tv_usec) / 1.e6);

    printf("Reserva de memoria: %.10lf\n", time);

    // start timer
    gettimeofday(&start, NULL);

    // call the function
    histogram(image, height, width, count);

    // stop timer
    gettimeofday(&finish, NULL);
    time = (finish.tv_sec - start.tv_sec +
            (finish.tv_usec - start.tv_usec) / 1.e6);

    printf("Ejecucion: %.10lf\n", time);

    // for (int i = 0; i < SIZE; i++) {
    //     printf("%d ", count[i]);
    // }
    // printf("\n");

    free(image);
    free(count);

    return EXIT_SUCCESS;
}
