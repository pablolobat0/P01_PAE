#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "./Lab1_material/pgmio.h"

#define SIZE 256

int main(int argc, char *argv[]) {
    // check for arguments
    if (argc < 2) {
    	printf("Use %s file.pgm\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int *count;

    count = malloc(sizeof(int) * SIZE);

    for (int i = 0; i < SIZE; i++) {
        count[i] = 0;
    }

    // image width x height
    int width, height;

    // Load pgm image
    unsigned char* xu8 = loadPGMu8(argv[1], &width, &height);

    for (int i = 0; i < height * width; i++) {
        count[xu8[i]] += 1;
    }

    for (int i = 0; i<SIZE; i++) {
        printf("%d ", count[i]); 
    }
    printf("\n");

    free(xu8);
    free(count);

    return EXIT_SUCCESS;
}
