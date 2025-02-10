#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "./Lab1_material/pgmio.h"

#define ROWS 3
#define COLS 3

float convulucion(float *matriz, int kernel[3][3], int width, int height, int row, int col) {
    int mid_row = ROWS / 2;
    int mid_col = COLS / 2;
    int start_row = row - mid_row;
    int start_col = col - mid_col;

    float sum = 0;

    for (int i = start_row, a = 0; a < ROWS; i++, a++) {
        for (int j = start_col, b = 0; b < COLS; j++, b++) {
            if (i >= 0 && i < height && j >= 0 && j < width) {
                sum += matriz[i *height + j] * kernel[a][b];
            }
        }
    }

    return sum;
}

int main(int argc, char *argv[]) {
    int kernel[ROWS][COLS]= {
        {0, -1 ,0},
        {-1, 5, -1},
        {0, -1, 0}
    };
    // check for arguments
    if (argc < 2) {
    	printf("Use %s file.pgm\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // image width x height
    int width, height;

    // Load pgm image
    float* xu8 = loadPGM32(argv[1], &width, &height);
    float* temp = malloc(sizeof(float) * width * height);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          temp[i * height + j] = convulucion(xu8, kernel, width, height, i, j );
        }
    }

    savePGM32("hola.pgm", temp,  width,  height);

    free(xu8);
    free(temp);

    return EXIT_SUCCESS;
}
