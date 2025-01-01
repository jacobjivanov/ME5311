#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>

/*
gcc fft_test.c -o fft_test.out -I/opt/homebrew/Cellar/fftw/3.3.10_1/include -L/opt/homebrew/Cellar/fftw/3.3.10_1/lib -lfftw3 -pedantic -Wall -Wextra
*/

fftw_plan create_fft_plan(int N) {
    fftw_plan plan;

    // Create FFTW plan
    plan = fftw_plan_dft_1d(N, NULL, NULL, FFTW_FORWARD, FFTW_ESTIMATE);

    return plan;
}

void destroy_fft_plan(fftw_plan plan) {
    fftw_destroy_plan(plan);
}

void forward_fft(double *input, int N, fftw_plan plan) {
    fftw_complex *in, *out;

    // Allocate memory for input and output arrays
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    // Copy input array to FFTW complex array
    for (int i = 0; i < N; ++i) {
        in[i][0] = input[i]; // Real part
        in[i][1] = 0;         // Imaginary part
    }

    // Assign input and output arrays to the plan
    // fftw_plan_with_nthreads(1); // Avoids thread safety issues
    fftw_execute_dft(plan, in, out);

    // Output the result
    printf("Result of forward FFT:\n");
    for (int i = 0; i < N; ++i) {
        printf("%f + %fi\n", out[i][0], out[i][1]);
    }

    // Free memory
    fftw_free(in);
    fftw_free(out);
}

int main(void) {
    // Example input array
    double input[] = {1, 2, 3, 4, 5};
    int N = sizeof(input) / sizeof(double);

    // Create FFTW plan
    fftw_plan plan = create_fft_plan(N);

    // Perform forward FFT
    forward_fft(input, N, plan);

    // Destroy FFTW plan
    destroy_fft_plan(plan);

    return 0;
}
