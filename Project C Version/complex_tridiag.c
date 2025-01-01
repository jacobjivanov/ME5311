#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <fftw3.h>

/*
gcc complex_tridiag.c -o complex_tridiag.out -I/opt/homebrew/Cellar/fftw/3.3.10_1/include -L/opt/homebrew/Cellar/fftw/3.3.10_1/lib -lfftw3 -pedantic -Wall -Wextra -Wconversion
*/

fftw_complex* tridiag(fftw_complex *x, double *a, double *b, double *c, fftw_complex *d, int N) {
    for (int i = 1; i < N; i++) {
        double w = a[i-1]/b[i-1];
        b[i] -= w*c[i-1];
        d[i] -= w*d[i-1];
    }

    x[N-1] = d[N-1]/b[N-1];
    for (int i = N-2; -1 < i; i--) {
        x[i] = (d[i] - c[i]*x[i+1])/b[i];
    }

    return x;
}

int main(void) {
    int N = 4;
    double a[] = {-9, -4, 2};
    double b[] = {-10, -3, -1, 5};
    double c[] = {-9, -5, -2};
    fftw_complex d[] = {-8+4*I, -7+3*I, -5+I, 4};

    fftw_complex* x = (fftw_complex*)malloc(N * sizeof(fftw_complex));
    tridiag(x, a, b, c, d, N);
    
    printf("Resulting array:\n");
    for (int i = 0; i < N; i++) {
        printf("%+.2f%+.2fi\n", creal(x[i]), cimag(x[i]));
    }

    free(x);
}
