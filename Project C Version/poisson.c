#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <fftw3.h>

/*
gcc poisson.c -o poisson.out -I/opt/homebrew/Cellar/fftw/3.3.10_1/include -L/opt/homebrew/Cellar/fftw/3.3.10_1/lib -lfftw3 -pedantic -Wall -Wextra -Wconversion
*/


fftw_complex* tridiag(fftw_complex *X, double *a, double *b, double *c, fftw_complex *D, int N) {
    for (int i = 1; i < N; i++) {
        double w = a[i-1]/b[i-1];
        b[i] -= w*c[i-1];
        D[i] -= w*D[i-1];
    }

    X[N-1] = X[N-1]/b[N-1];
    for (int i = N-2; -1 < i; i--) {
        X[i] = (D[i] - c[i]*X[i+1])/b[i];
    }

    return X;
}

fftw_plan create_forward_fft_plan(int M) {
    fftw_plan plan;
    plan = fftw_plan_dft_1d(M, NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);

    return plan;
}

fftw_plan create_inverse_fft_plan(int M) {
    fftw_plan plan;
    plan = fftw_plan_dft_1d(M, NULL, NULL, FFTW_BACKWARD, FFTW_ESTIMATE);

    return plan;
}

void destroy_fft_plan(fftw_plan plan) {
    fftw_destroy_plan(plan);
}

double wavenumber(int i, int M) {
    if (i < M/2 + 1) {
        return i;
    }
    else {
        return i-M;
    }
}

double* poisson(double *p, double *d, int M, int N, double L, fftw_plan for_plan, fftw_plan inv_plan) {
    double dx = L/M;
    double dy = 1/N;

    double* A = (double*) malloc((N+1)*sizeof(double));
    for (int j = 0; j < N-1; j++) {
        A[j] = 1/dy/dy;
    }
    A[N] = -1;

    double* C = (double*) malloc((N+1)*sizeof(double));
    C[0] = -1;
    for (int j = 1; j < N; j++) {
        C[j] = 1/dy/dy;
    }

    fftw_complex* D = (fftw_complex*) malloc(M*(N+2)*sizeof(fftw_complex));
    for (int i = 0; i < M; i++) {
        D[i] = 0;
        D[i + M*(N+1)] = 0;
    }

    fftw_complex* d_row = (fftw_complex*) malloc(M*sizeof(fftw_complex));
    fftw_complex* D_row = (fftw_complex*) malloc(M*sizeof(fftw_complex));

    for (int j = 1; j < N+1; j++) {
        for (int i = 0; i < M; i++) {
            d_row[i] = d[i + M*(j-1)] + 0*I;
        }
        
        fftw_execute_dft(for_plan, d_row, D_row);
        for (int i = 0; i < M; i++) {
            D[i + M*j] = D_row[i];
        }
    }
    free(d_row); free(D_row);

    double* B = (double*) malloc((N+2)*sizeof(double));
    B[0] = 1; B[N+1] = 1;

    fftw_complex* P = (fftw_complex*) malloc(M*(N+2)*sizeof(fftw_complex));
    for (int j = 0; j < N+2; j++) {
        P[M*j] = 0;
    }

    fftw_complex* D_col = (fftw_complex*) malloc((N+2)*sizeof(fftw_complex));
    fftw_complex* P_col = (fftw_complex*) malloc((N+2)*sizeof(fftw_complex));

    for (int i = 1; i < M; i++) {
        double k = wavenumber(i, M);

        for (int j = 1; j < N+1; j++) {
            B[j] = -2/dy/dy + (-2 + 2*cos(2*M_PI*k/M))/dx/dx;
        }

        for (int j = 0; j < N+2; j++) {
            D_col[j] = D[i + M*j];
        }

        tridiag(P_col, A, B, C, D_col, N);

        for (int j = 0; j < N+2; j++) {
            P[i + M*j] = P_col[j];
        }
    }
    free(P_col); free(D_col); free(A); free(B); free(C); free(D);

    fftw_complex* P_row = (fftw_complex*) malloc(M*sizeof(fftw_complex));
    fftw_complex* p_row = (fftw_complex*) malloc(M*sizeof(fftw_complex));
    for (int j = 1; j < N+1; j++) {
        for (int i = 0; i < M; i++) {
            P_row[i] = P[i + M*j];
        }

        fftw_execute_dft(inv_plan, P_row, p_row);

        for (int i = 0; i < M; i++) {
            p[i + M*(j-1)] = creal(p_row[i]);
        }
    }
    free(P_row); free(p_row); free(P);
    
    return p;
}

int main(void) {
    int M = 6; int N = 6; double L = 2*M_PI;

    fftw_plan for_plan = create_forward_fft_plan(M);
    fftw_plan inv_plan = create_inverse_fft_plan(M);

    double* p = (double*) malloc(M*N*sizeof(double));
    double* d = (double*) malloc(M*N*sizeof(double));

    double x; double y;
    for (int i = 0; i < M; i++) {
        x = i*L/M;
        for (int j = 0; j < N; j++) {
            y = (double)j/N;
            d[i + M*j] = cos(2*M_PI*x/L) * sin(M_PI*y);
        }
    }


    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            printf("%+.2f ", d[i + M*j]);
        }
        printf("\n");
    }
    printf("\n");
    p = poisson(p, d, M, N, L, for_plan, inv_plan);

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            printf("%+.2f ", p[i + M*j]);
        }
        printf("\n");
    }

    free(p);
}
