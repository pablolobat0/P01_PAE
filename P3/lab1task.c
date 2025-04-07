#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define M 16384
#define N 16384

int main() {
    int* matriz = malloc(M * N * sizeof(int));
    double start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < M; i++) {
                #pragma omp task firstprivate(i)
                {
                    for (int j = 0; j < N; j++) {
                        matriz[i * N + j] = i * N + j;
                    }
                }
            }
        }
    }

    double end = omp_get_wtime();
    printf("Tiempo con tasks: %.4f s\n", end - start);

    free(matriz);
    return 0;
}
