// simpson_integral.c

# include <stdio.h>
# include <math.h>
const double pi = 3.14159265358979323846;

double simpson_integral(double (*function)(double), double x_min, double x_max, double dx) {
    double A = 0;
    int N = (x_max - x_min)/dx;

    for (int n = 0; n < N; n += 2) {
        A += function(x_min + n*dx) + 4*function(x_min + n*dx + dx) + function(x_min + n*dx + 2*dx);
    }

    A *= dx/3;
    return A;
}

double f(double x) {
    return sin(x);
}

int main() {
    printf("N,error\n");
    for (int n = 1; n <= 30; n++) {
        double a = 0;
        double b = pi;
        int N = 1 << n;
        double dx = pi/N;

        double A_num = simpson_integral(f, a, b, dx);

        printf("%d,%.20f\n", N, fabs(A_num - 2));
    }
}