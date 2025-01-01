import numpy as np
from numpy import real, imag
from numpy.fft import fft, ifft
import numba
# np.set_printoptions(precision = 1)

import matplotlib.pyplot as plt

@numba.njit()
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

G = np.array([
    [8, 8],
    [16, 16],
    [32, 32],
    [64, 64],
    [128, 128],
    [256, 256],
    [512, 512],
    [1024, 1024],
    [2048, 2048],
    [4096, 4096],
])

e = np.empty(G.shape[0])

for g in range(0, G.shape[0]):
    M, N, L = G[g, 0], G[g, 1], 2*np.pi/7
    dx, dy = L/M, 1/N

    x = np.empty(M)
    for i in range(0, M):
        x[i] = dx/2 + i*dx
    y = np.empty(N)
    for j in range(0, N):
        y[j] = dy/2 + j*dy

    x_grid, y_grid = np.meshgrid(x, y, indexing = 'ij')
    kp = np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0)))

    d = -16*np.pi**2 * (4 + L**2)/L**2 * np.sin(8*np.pi*x_grid/L) * np.cos(4*np.pi*y_grid)

    # set up constant coefficient vectors
    A = np.ones(N+1)/dy**2; A[N] = -1
    C = np.ones(N+1)/dy**2; C[0] = -1
    D = np.zeros(shape = (M, N+2), dtype = 'complex')
    D[:, 1:N+1] += fft(d, axis = 0)
    P = np.zeros(shape = d.shape, dtype = 'complex')

    for i in range(1, M):
        k = kp[i]
        # set up (changing) main diagonal coefficient vector
        B = np.ones(N+2, dtype = 'complex')
        B[0] = 1; B[N+1] = 1
        for j in range(0, N):
            B[j+1] = -2./dy**2 + (-2 + 2*np.cos(2*np.pi*k/M))/dx**2
        
        # solve tridiagonal system for each column
        P[i, :] = tridiag(A, B, C, D[i, :])[1:N+1]
    p = real(ifft(P, axis = 0))

    p_ana = np.sin(8*np.pi/L*x_grid) * np.cos(4*np.pi*y_grid)
    e[g] = np.max(np.abs(p - p_ana))
    print("Computational Grid: {0}✕{1} Completed".format(G[g, 0], G[g, 1]))

plt.plot(G[:, 0], 1./G[:, 0]**2, ':.', color = 'blue', label = '2nd Order Convergence')

plt.plot(G[:, 0], e, ':.', color = 'red', label = r"Domain N✕N")

plt.title("Convergence of Poisson Equation")
plt.xlabel(r"$N$")
plt.ylabel(r"$\max |p_\mathrm{num} - p_\mathrm{ana}|$")
plt.xscale('log', base = 2)
plt.yscale('log', base = 10)
plt.legend()
# plt.savefig("Poisson Convergence.png", dpi = 200)
plt.show()