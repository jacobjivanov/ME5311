import numpy as np
from numpy import pi

import matplotlib.pyplot as plt

def INIT_GRID(M, N):
    x_cent = np.empty(shape = (M, N)); y_cent = np.empty(shape = (M, N))
    x_left = np.empty(shape = (M, N)); y_left = np.empty(shape = (M, N))
    for i in range(0, M):
        for j in range(0, N):
            x_cent[i, j] = (i + 1/2)*dx
            x_left[i, j] = i*dx
            
            y_cent[i, j] = (j + 1/2)*dy
            y_left[i, j] = (j + 1/2)*dy
    
    x_bott = np.empty(shape = (M, N+1)); y_bott = np.empty(shape = (M, N+1))
    for i in range(0, M):
        for j in range(0, N+1):
            x_bott[i, j] = (i + 1/2)*dx
            y_bott[i, j] = j*dy
    
    return x_cent, y_cent, x_left, y_left, x_bott, y_bott

def CALC_FLUX(u, v, θ):
    fu = np.empty(shape = (M, N))
    for i in range(0, M):
        fu[i, :] = (1/4)*(u[i, :] + u[(i+1)%M, :])**2
    
    gu = np.empty(shape = (M, N+1))
    gu[:, 0] = 0; gu[:, N] = 0 # flux on walls is zero
    for i in range(0, M):
        for j in range(1, N):
            gu[i, j] = (1/4)*(u[i, j-1] + u[i, j])*(v[i-1, j] + v[i, j])
    
    fv = np.empty(shape = (M, N+1))
    fv[:, 0] = 0; fv[:, N] = 0 # flux on walls is zero
    for i in range(0, M):
        for j in range(1, N):
            fv[i, j] = (1/4)*(u[(i+1)%M, j-1] + u[(i+1)%M, j])*(v[i, j] + v[(i+1)%M, j])

    gv = np.empty(shape = (M, N))
    for j in range(0, N):
        gv[:, j] = (1/4)*(v[:, j] + v[:, j+1])**2
    
    fθ = np.empty(shape = (M, N))
    for i in range(0, M):
        fθ[i, :] = (1/2)*u[i, :]*(θ[i, :] + θ[i-1, :])

    gθ = np.empty(shape = (M, N+1))
    gθ[:, 0] = 0; gθ[:, N] = 0 # flux on walls is zero
    for j in range(1, N):
        gθ[:, j] = (1/2)*v[:, j]*(θ[:, j-1] + θ[:, j])

    return fu, gu, fv, gv, fθ, gθ

def CALC_CONV(u, v, θ):
    fu, gu, fv, gv, fθ, gθ = CALC_FLUX(u, v, θ)

    conv_u = np.zeros(shape = (M, N))
    for i in range(0, M):
        conv_u[i, :] -= (fu[i, :] - fu[i-1, :])/dx
    for j in range(0, N):
        conv_u[:, j] -= (gu[:, j+1] - gu[:, j])/dy
    
    conv_v = np.zeros(shape = (M, N+1))
    for i in range(0, M):
        conv_v[i, :] -= (fv[i, :] - fv[i-1, :])/dx
    for j in range(1, N):
        conv_v[:, j] -= (gv[:, j] - gv[:, j-1])/dy

    conv_θ = np.zeros(shape = (M, N))
    for i in range(0, M):
        conv_θ[i, :] -= (fθ[(i+1)%M, :] - fθ[i, :])/dx
    for j in range(0, N):
        conv_θ[:, j] -= (gθ[:, j+1] - gθ[:, j])/dy
    
    return conv_u, conv_v, conv_θ

def DIFF(u, v, θ):
    diff_u = np.empty(shape = (M, N))
    diff_u[:, 0] = (-3*u[:, 0] + u[:, 1])/dy**2
    diff_u[:, N-1] = (u[:, N-2] - 3*u[:, N-1])/dy**2
    for j in range(1, N-1):
        diff_u[:, j] = (u[:, j-1] -2*u[:, j] + u[:, j+1])/dy**2

    diff_v = np.empty(shape = (M, N+1))
    diff_v[:, 0] = 0; diff_v[:, N] = 0
    for j in range(1, N):
        diff_v[:, j] = (v[:, j-1] - 2*v[:, j] + v[:, j+1])/dy**2

    diff_θ = np.empty(shape = (M, N))
    diff_θ[:, 0] = (2 - 3*θ[:, 0] + θ[:, 1])/dy**2
    diff_θ[:, N-1] = (θ[:, N-2] - 3*θ[:, N-1])/dy**2

    for i in range(0, M):
        diff_u[i, :] += (u[i-1, :] - 2*u[i, :] + u[(i+1)%M, :])/dx**2
        diff_v[i, :] += (v[i-1, :] - 2*v[i, :] + v[(i+1)%M, :])/dx**2
        diff_θ[i, :] += (θ[i-1, :] - 2*θ[i, :] + θ[(i+1)%M, :])/dx**2

    return diff_u, diff_v, diff_θ

if __name__ == '__main__':
    def U_ANA(x, y):
        # satisfies u(x, y=0) = 0, u(x, y=1) = 0
        # return np.zeros(shape = x.shape)
        return np.sin(2*pi*x/L)*np.sin(pi*y)
    def V_ANA(x, y):
        # satisfies v(x, y=0) = 0, v(x, y=1) = 0
        return np.sin(pi*y)
    def θ_ANA(x, y):
        # satisfies θ(x, y=0) = 1, θ(x, y=1) = 0
        return 1-y
    def UCONVX_ANA(x, y):
        return -2*pi/L*np.sin(4*pi*x/L)*np.sin(pi*y)**2
    def UCONVY_ANA(x, y):
        return -pi*np.sin(2*pi*x/L)*np.sin(2*pi*y)
    def UCONV_ANA(x, y):
        return UCONVX_ANA(x, y) + UCONVY_ANA(x, y)
    def VCONVX_ANA(x, y):
        return -2*pi/L*np.cos(2*pi*x/L)*np.sin(np.pi*y)**2
    def VCONVY_ANA(x, y):
        return -pi*np.sin(2*pi*y)
    def VCONV_ANA(x, y):
        return VCONVX_ANA(x, y) + VCONVY_ANA(x, y)
    def θCONVX_ANA(x, y):
        return 2*pi/L*(y-1)*np.cos(2*pi*x/L)*np.sin(pi*y)
    def θCONVY_ANA(x, y):
        return pi*(y-1)*np.cos(pi*y) + np.sin(pi*y)
    def θCONV_ANA(x, y):
        return θCONVX_ANA(x, y) + θCONVY_ANA(x, y)
    def UDIFF_ANA(x, y):
        return -(4+L**2)*pi**2 * np.sin(2*pi*x/L)*np.sin(pi*y)/L**2
    def VDIFF_ANA(x, y):
        return -pi**2 * np.sin(pi*y)
    def θDIFF_ANA(x, y):
        return np.zeros(x.shape)

    GRID = np.array([16, 32, 64, 128, 256, 512, 1024, 2048])
    Eu = np.zeros(GRID.size)
    Ev = np.zeros(GRID.size)
    Eθ = np.zeros(GRID.size)

    M, N, L = GRID[3], GRID[3], 2*pi
    dx, dy = L/M, 1/N
    x_cent, y_cent, x_left, y_left, x_bott, y_bott = INIT_GRID(M, N)

    u = U_ANA(x_left, y_left)
    v = V_ANA(x_bott, y_bott)
    θ = θ_ANA(x_cent, y_cent)

    diff_u, diff_v, diff_θ = DIFF(u, v, θ)

    # plt.pcolor(x_left, y_left, diff_u - UDIFF_ANA(x_left, y_left))
    # plt.pcolor(x_bott, y_bott, diff_v - VDIFF_ANA(x_bott, y_bott))
    plt.pcolor(x_cent, y_cent, diff_θ)

    plt.colorbar()
    plt.show()

    """
    for i in range(0, GRID.size):
        M, N, L = GRID[i], GRID[i], 2*pi
        dx, dy = L/M, 1/N
        x_cent, y_cent, x_left, y_left, x_bott, y_bott = INIT_GRID(M, N)

        u = U_ANA(x_left, y_left)
        v = V_ANA(x_bott, y_bott) # on ((i+1/2)*dx, j*dy) grid
        θ = θ_ANA(x_cent, y_cent)

        conv_u, conv_v, conv_θ = CALC_CONV(u, v, θ)

        Eu[i] = np.max(np.abs(conv_u - UCONV_ANA(x_left, y_left)))
        Ev[i] = np.max(np.abs(conv_v - VCONV_ANA(x_bott, y_bott)))
        Eθ[i] = np.max(np.abs(conv_θ - θCONV_ANA(x_cent, y_cent)))

        print(GRID[i])

    plt.loglog(GRID, Eu, ':.', color = 'blue', label = "$u$ conv error")
    plt.loglog(GRID, Ev, ':.', color = 'red', label = "$v$ conv error")
    plt.loglog(GRID, Eθ, ':.', color = 'green', label = r"$\theta$ conv error")
    plt.loglog(GRID, 1/GRID**2, '-', color = 'grey', label = '2nd Order Convergence')
    plt.xlabel("$x$")
    plt.ylabel(r"$\ell_\infty[\mathbf{e}]$")
    plt.legend()
    plt.title("Convergence of Convection Term Error")
    plt.show()
    """