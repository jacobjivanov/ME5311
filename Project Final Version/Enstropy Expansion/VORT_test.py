import numpy as np
from numpy import pi
import numba
import matplotlib.pyplot as plt

def INIT_GRID(M, N):
    x_cent = np.empty(shape = (M, N)); y_cent = np.empty(shape = (M, N))
    x_left = np.empty(shape = (M, N)); y_left = np.empty(shape = (M, N))
    for i in range(0, M):
        for j in range(0, N):
            x_cent[i, j] = (i + 1/2)*Δx
            x_left[i, j] = i*Δx
            
            y_cent[i, j] = (j + 1/2)*Δy
            y_left[i, j] = (j + 1/2)*Δy
    
    x_bott = np.empty(shape = (M, N+1)); y_bott = np.empty(shape = (M, N+1))
    for i in range(0, M):
        for j in range(0, N+1):
            x_bott[i, j] = (i + 1/2)*Δx
            y_bott[i, j] = j*Δy
    
    return x_cent, y_cent, x_left, y_left, x_bott, y_bott

@numba.njit()
def VORTICITY(u, v):
    dvdx = np.empty(shape = (M, N+1))
    for i in range(0, M):
        dvdx[i, :] = (v[i, :] - v[i-1, :])/Δx
    
    dudy = np.empty(shape = (M, N+1))
    dudy[:, 0] = 2*u[:, 0]/Δy
    dudy[:, N] = -2*u[:, N-1]/Δy
    for j in range(1, N):
        dudy[:, j] = (u[:, j] - u[:, j-1])/Δy
    
    vort = dvdx - dudy

    vort_left = np.empty(shape = (M, N))
    for j in range(0, N):
        vort_left[:, j] = (vort[:, j+1] + vort[:, j])/2
    
    vort_cent = np.empty(shape = (M, N))
    for i in range(0, M):
        vort_cent[i, :] = (vort_left[(i+1)%M, :] + vort_left[i, :])/2

    return vort_cent

if __name__ == '__main__':
    M = 1024; N = 1024; L = 2
    Δx = L/M; Δy = 1/N

    x_cent, y_cent, x_left, y_left, x_bott, y_bott = INIT_GRID(M, N)

    u = np.sin(2*pi*x_left/L)*np.sin(pi*y_left)
    v = np.sin(pi*y_bott)*np.cos(2*pi*x_bott/L)
    print(u.shape, v.shape)

    vort = VORTICITY(u, v)
    vort_ana = -pi/L * np.sin(2*pi*x_cent/L) * (L*np.cos(pi*y_cent) + 2*np.sin(pi*y_cent))

    plt.pcolor(x_left, y_left, vort - vort_ana)
    plt.colorbar()
    plt.show()