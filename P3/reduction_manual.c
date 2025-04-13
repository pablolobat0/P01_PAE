#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define TAM 671088640
#define ALPHA 5

void init_array(float *array, int tam, int alpha) {
#pragma omp parallel for schedule(runtime)
    for (int i = 0; i < tam; i++) {
        array[i] = (float)i * alpha;
    }
}

void euclidian_distance(float *a, float *b, int tam) {
    float sum = 0.0;

#pragma omp parallel
    {
        float local_sum = 0.0;

#pragma omp for schedule(runtime)
        for (int i = 0; i < tam; i++) {
            float diff = a[i] - b[i];
            local_sum += diff * diff;
        }

#pragma omp critical
        {
            sum += local_sum;
        }
    }

    sqrt(sum);
}

int main(int argc, char *argv[]) {
    float *a, *b;
    struct timeval ex_start, ex_finish, init_start, init_finish;
    double ex_time, init_time;
    int alpha, N;

    alpha = argc >= 2 ? atoi(argv[1]) : ALPHA;
    N = argc == 3 ? atoi(argv[2]) : TAM;

    gettimeofday(&init_start, NULL);

    a = (float *)malloc(sizeof(float) * N);
    b = (float *)malloc(sizeof(float) * N);

    init_array(a, N, 1);
    init_array(b, N, alpha);

    gettimeofday(&init_finish, NULL);

    gettimeofday(&ex_start, NULL);
    euclidian_distance(a, b, N);
    gettimeofday(&ex_finish, NULL);

    ex_time = (ex_finish.tv_sec - ex_start.tv_sec +
               (ex_finish.tv_usec - ex_start.tv_usec) / 1.e6);
    init_time = (init_finish.tv_sec - init_start.tv_sec +
                 (init_finish.tv_usec - init_start.tv_usec) / 1.e6);

    printf("TIEMPO DE INICIALIZACION: %.10lf\nTIEMPO DE EJECUCION: %.10lf\n",
           init_time, ex_time);

    free(a);
    free(b);

    return EXIT_SUCCESS;
}
