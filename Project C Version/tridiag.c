#include <stdio.h>
#include <stdlib.h>

double* tridiag(double a[], double b[], double c[], double d[], int N) {
    for (int i = 1; i < N; i++) {
        double w = a[i-1]/b[i-1];
        b[i] -= w*c[i-1];
        d[i] -= w*d[i-1];
    }

    double* x = (double*)malloc(N * sizeof(double));
    x[N-1] = d[N-1]/b[N-1];
    for (int i = N-2; -1 < i; i--) {
        x[i] = (d[i] - c[i]*x[i+1])/b[i];
    }

    return x;
}

int main() {
    int N = 4;
    double a[] = {-9, -4, 2};
    double b[] = {-10, -3, -1, 5};
    double c[] = {-9, -5, -2};
    double d[] = {-8, -7, -5, 4};

    double* x = tridiag(a, b, c, d, N);
    printf("Resulting array:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", x[i]);
    }

    printf("\n");
    free(x);
}

/*
int N = 4;
double a[] = {1, 2, 3};
double b[] = {1, 2, 3, 4};
double c[] = {1, 2, 3};
double d[] = {1, 2, 3, 4};

x = {1.23, -0.23, 0.62, 0.54}
*/

/*
int N = 4;
double a[] = {-9, -4, 2};
double b[] = {-10, -3, -1, 5};
double c[] = {-9, -5, -2};
double d[] = {-8, -7, -5, 4};

x = [0.07, 0.81, 0.79, 0.49]
*/