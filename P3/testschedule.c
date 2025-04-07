#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define M 16384
#define N 16384

int main() {
    double start, end;
    int tid,nthreads;

    // Static, chunk size 64
    start = omp_get_wtime();
    omp_set_dynamic(0);
    #pragma omp parallel private(tid,nthreads) num_threads(5)
    {
        tid = omp_get_thread_num();
        if (tid == 0) {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }
        #pragma omp for
        for (int i = 0; i < 100; i++) {
            printf("Hilo: %d, it: %d\n", tid, i);
        }
    }
    end = omp_get_wtime();
    printf("Tiempo con parallel for (static,64): %.4f s\n", end - start);

    return 0;
}
