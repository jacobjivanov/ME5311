#include <stdio.h>
#include <stdlib.h>

double* wavenumbers(double *k, int M) {
    for (int i = 0; i < M/2 + 1; i++) {
        k[i] = i;
    }
    for (int i = M/2 + 1; i < M; i++) {
        k[i] = i - M;
    }

    return k;
}

int main(void) {
    int M = 16;
    double* k = (double*)malloc(M * sizeof(double));
    wavenumbers(k, M);

    for (int i = 0; i < M; i++) {
        printf("%+.2f\n", k[i]);
    }
}
