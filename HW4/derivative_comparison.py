import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

def func_u(x):
    return np.exp(np.sin(x))

def func_u_prime(x):
    return np.cos(x) * np.exp(np.sin(x))

def finite_difference(u, i, dx):
    N = u.size
    dudx = -1/12 * (u[(i+2)%N] - u[(i-2)%N]) 
    dudx += 2/3 * (u[(i+1)%N] - u[(i-1)%N])
    dudx /= dx

    return dudx

def lp(e, dx, p):
    lp_error = 0
    for i in range(0, len(e)):
        lp_error += np.power(e[i], p)
    lp_error *= dx
    lp_error = np.power(lp_error, 1./p)

    return lp_error

N = np.array([4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40])
e_finite = np.empty(len(N))
e_spectral = np.empty(len(N))

for n in range(0, len(N)):
    x = np.linspace(0, 2*np.pi, N[n], endpoint = False)
    u = func_u(x)
    u_prime = func_u_prime(x)

    dudx_finite = np.empty(N[n])
    for i in range(0, N[n]):
        dudx_finite[i] = finite_difference(u, i, x[1])

    kx = np.empty(N[n])
    for i in range(0, N[n]//2):
        kx[i] = i
    for i in range(N[n]//2, N[n]):
        kx[i] = -N[n] + i

    dudx_spectral = ifft(1j * kx * fft(u))

    e_finite[n] = lp(np.abs(u_prime - dudx_finite), x[1], 2)
    e_spectral[n] = lp(np.abs(u_prime - dudx_spectral), x[1], 2)

plt.loglog(2*np.pi/N, e_finite, label = "4th-Order Finite Difference")
plt.loglog(2*np.pi/N, e_spectral, label = 'Spectral')
plt.legend()
plt.xlabel(r"$\Delta x$")
plt.ylabel(r"$\ell_2[e]$")
plt.title("Convergence of Finite Difference/Spectral\nDerivative of $u(x) = \exp(\sin(x))$")
plt.savefig("Convergence of Derivatives.png", dpi = 200)