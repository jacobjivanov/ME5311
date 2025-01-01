import numpy as np
from numpy import pi
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

def DIFF(θ):
    diff_θ = np.zeros(shape = (M, N))

    diff_θ[:, 0] = (2 - 3*θ[:, 0] + θ[:, 1])/Δy**2
    diff_θ[:, N-1] = (θ[:, N-2] - 3*θ[:, N-1])/Δy**2
    for j in range(1, N-1):
        diff_θ[:, j] += (θ[:, j-1] -2*θ[:, j] + θ[:, j+1])/Δy**2
    
    
    """
    for i in range(0, M):
        diff_θ[i, :] += (θ[i-1, :] - 2*θ[i, :] + θ[(i+1)%M, :])/Δx**2
    """
    return diff_θ

if __name__ == '__main__':
    M = 16; N = 16; L = 8
    Δx = L/M; Δy = 1/N
    x_cent, y_cent, x_left, y_left, x_bott, y_bott = INIT_GRID(M, N)

    def θANA(x, y):
        return np.sin(2*pi*x/L) * np.sin(2*pi*y)/10 + 1/(y + 1) - y/2

    θ = np.sin(2*pi*x_cent/L) * np.sin(2*pi*y_cent)/10 + 1/(y_cent + 1) - y_cent/2

    diff_θ = DIFF(θ)

    def DIFFθ_ANA(x, y):
        return 2 /(1 + y)**3 - 2 * (1 + L**2) * np.pi**2 * np.sin((2 * np.pi * x) / L) * np.sin(2 * np.pi * y) / (5 * L**2)
    
    def DIFFθX_ANA(x, y):
        return - 2*pi**2/(5*L**2) * np.sin(2*pi*x/L) * np.sin(2*pi*y)

    def DIFFθY_ANA(x, y):
        return 2/(1+y)**3 - 2*pi**2/5 * np.sin(2*pi*x/L) * np.sin(2*pi*y)

    plt.pcolor(x_cent, y_cent, diff_θ - DIFFθY_ANA(x_cent, y_cent))
    # plt.pcolor(x_bott, y_bott, θANA(x_bott, y_bott))
    plt.colorbar()
    plt.show()