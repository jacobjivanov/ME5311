import numpy as np
import matplotlib.pyplot as plt

def lp(e, dx, p):
    lp_error = dx * np.sum(np.power(np.abs(e), p))
    lp_error = np.power(lp_error, 1./p)
    
    return lp_error

def tridiag(a, b, c, d):
    # returns the solution x for the linear system described by:
    # a_i x_{i-1}  + b_i x_i  + c_i x_{i+1}  = d_i,
    # using the algorithm described here: 
    # https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

    N = b.size

    for i in range(1, N):
        w = a[i-1]/b[i-1]
        b[i] = b[i] - w*c[i-1]
        d[i] = d[i] - w*d[i-1]
    
    x = b
    x[N-1] = d[N-1]/b[N-1]
    for i in range(N-2, -1, -1):
        x[i] = (d[i] - c[i]*x[i+1])/b[i]

    return x

N = np.array([2**n for n in range(2, 20)])
l2e = np.empty(N.size)
for i in range(0, N.size):
    y = np.linspace(0, 1, N[i])
    dy = y[1]

    d2p = -16 * np.pi**2 * np.sin(4 * np.pi * y)

    a = np.ones(N[i]-1)
    b = np.ones(N[i])-3
    c = np.ones(N[i]-1)
    d = np.zeros(N[i])

    a[N[i]-2] = 0
    b[0] = 1; b[N[i]-1] = 1
    c[0] = 0
    d[1:N[i]-1] = dy*dy * d2p[1:N[i]-1]

    p = tridiag(a, b, c, d)
    l2e[i] = lp(
        p - np.sin(4 * np.pi * y),
        dy,
        2
    )

plt.figure(figsize = (7.5, 4))
plt.title("Convergence of Tridiagonal Linear System Solver")
plt.xlabel(r"$N$")
plt.ylabel(r"$\ell_2 [\mathbf{e}]$")
plt.loglog(N, l2e, ':.', color = 'blue')
plt.loglog(N, 1/N**2, color = 'red', linestyle = 'dashed', label = '2nd Order')
plt.legend()
plt.yticks(ticks = np.array([
    1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12
]))
plt.savefig("tridiagonal_convergence.png", dpi = 200)
plt.show()