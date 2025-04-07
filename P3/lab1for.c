#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define M 16384
#define N 16384

int main() {
    int* matriz = malloc(M * N * sizeof(int));
    double start, end;

    // Static, chunk size 64
    start = omp_get_wtime();
    #pragma omp parallel for schedule(static, 64)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matriz[i * N + j] = i * N + j;
        }
    }
    end = omp_get_wtime();
    printf("Tiempo con parallel for (static,64): %.4f s\n", end - start);

    free(matriz);
    return 0;
}
