import numpy as np
import numba

@numba.njit()
def simpson_integral(f, x_min, x_max, dx):
    A = 0
    N = (x_max - x_min)//dx

    n = 0
    while n < N:
        A += f(x_min + n*dx) + 4*f(x_min + (n+1)*dx) + f(x_min + (n+2)*dx)
        n += 2
    
    A *= dx/3
    return A

@numba.njit()
def f(x):
    return x**(3/2)

A_ana = 2/5
n = 0
N = np.empty(20)
e = np.empty(20)
while n < 20:
    N[n] = 2**n

    A = simpson_integral(f, 0, 1, 1/N[n])
    e[n] = A - A_ana
    n += 1

import matplotlib.pyplot as plt
plt.loglog(N, e, 'o', color = 'blue', label = "Simpson Integration Error")
plt.loglog(N, 1/N**2, color = 'yellow', label = "2nd Order Convergence")
plt.loglog(N, 1/N**3, color = 'orange', label = "3rd Order Convergence")
plt.loglog(N, 1/N**4, color = 'red', label = "4th Order Convergence")
plt.legend()
plt.xlabel(r"$N$")
plt.ylabel(r"$e$")
plt.ylim([1e-18, 1e-2])
plt.title(r"Simpson Integration Error over $N$")
plt.savefig("Simpson Integration Error over N.png", dpi = 200)