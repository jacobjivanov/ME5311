import numpy as np
from numpy import real, imag
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt

def lp(e, dx, p):
    lp_error = dx * np.sum(np.power(np.abs(e), p))
    lp_error = np.power(lp_error, 1./p)
    
    return lp_error

N = np.array([2**n for n in range(3, 20)])
l2e = np.empty(N.size)
for i in range(0, N.size):
    x = np.linspace(0, 2, N[i], endpoint = False)
    d = -36*np.pi*np.pi * np.cos(6*np.pi*x)

    D = fft(d)
    P = np.zeros(N[i], dtype = 'complex')
    for p in range(1, N[i]):
        P[p] = x[1]**2 / (-2 + 2*np.cos(2*np.pi*p/N[i])) * D[p]

    p_num = real(ifft(P))
    p_ana = np.cos(6*np.pi*x)
    e = p_num - p_ana

    l2e[i] = lp(e, x[1], 2)


plt.figure(figsize = (7.5, 4))
plt.title("Convergence of Fourier Method Solver")
plt.xlabel(r"$N$")
plt.ylabel(r"$\ell_2 [\mathbf{e}]$")
plt.loglog(N, l2e, ':.', color = 'blue')
plt.loglog(N, 1/N**2, color = 'red', linestyle = 'dashed', label = '2nd Order')
plt.legend()
plt.savefig("fourier_convergence.png", dpi = 200)
plt.show()