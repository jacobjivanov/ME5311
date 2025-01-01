import numpy as np
from numpy import real, imag
from numpy.fft import fft, ifft

import matplotlib.pyplot as plt

def TRIDIAG(a, b, c, d):
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

def POISSON(d):
    """
    p is on the center of each grid cell
    p = np.empty(shape = (M, N))
    d is on the center of each grid cell
    d = np.empty(shape = (M, N))
    """

    kp = np.array(list(range(0, M//2 + 1)) + list(range(- M//2 + 1, 0)))

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
            B[j+1] = -2./dy**2 + (-2 + 2*np.cos(k*dx))/dx**2    
        
        # solve tridiagonal system for each column
        P[i, :] = TRIDIAG(A, B, C, D[i, :])[1:N+1]
    p = real(ifft(P, axis = 0))

    return p

def DIV(u, v):
    """
    u is on the left side of each grid cell
    u = np.empty(shape = (M, N))
    v is on the bottom edge of each grid cell, but must also include the top wall
    v = np.empty(shape = (M, N+1))
    """

    dudx = np.empty(shape = (M, N))
    for i in range(0, M-1):
        dudx[i, :] = (u[i+1, :] - u[i, :])/dx
    dudx[-1, :] = (u[0, :] - u[-1, :])/dx

    dvdy = np.empty(shape = (M, N))
    for j in range(0, N):
        dvdy[:, j] = (v[:, j+1] - v[:, j])/dy
    
    div = dudx + dvdy
    return div

def CURL(p):
    """
    p is on the center of each grid cell
    dp/dx is on the left edge of each grid cell
    dp/dy is on the bottom edge of each grid cell, and includes the top wall

    then both are offset and returned as centered on each grid cell
    """

    # calculate both dpdx and dpdy on the staggered grid
    dpdx = np.zeros(shape = (M, N))
    for i in range(1, M):
        dpdx[i, :] = (p[i, :] - p[i-1, :])/dx
    dpdx[0, :] = (p[0, :] - p[-1, :])/dx

    dpdy = np.zeros(shape = (M, N+1))
    dpdy[:, 0] = 0; dpdy[:, N] = 0
    for j in range(1, N):
        dpdy[:, j] = (p[:, j] - p[:, j-1])/dy

    return dpdx, dpdy

def verify_grid_loc(x, y, L): # successful
    """ 
    Grid Location Visual Test
    """
    plt.scatter(x, y, color = 'black')
    plt.plot(
        np.array([0, L, L, 0, 0]),
        np.array([0, 0, 1, 1, 0]),
        linestyle = 'dashed', color = 'blue'
    )
    plt.gca().set_aspect('equal')
    plt.show()

def verify_div(M, N): # successful
    """
    if u = sin(2πx/L), v = cos(πy)
    div[u] = 2*π/L cos(2πx/L) - π sin(πy)

    in order to test this, I will initialize seperate x/y grids for u/v, fill u/v, and compute the divergence in order to verify that it is correctly functioning. However, there are *three* seperate grids we need to keep track of
    """

    def DIV(u, v):
        """
        u is on the left side of each grid cell
        u = np.empty(shape = (M, N))
        v is on the bottom edge of each grid cell, but must also include the top wall
        v = np.empty(shape = (M, N+1))
        """

        dudx = np.empty(shape = (M, N))
        for i in range(0, M-1):
            dudx[i, :] = (u[i+1, :] - u[i, :])/dx
        dudx[-1, :] = (u[0, :] - u[-1, :])/dx

        dvdy = np.empty(shape = (M, N))
        for j in range(0, N):
            dvdy[:, j] = (v[:, j+1] - v[:, j])/dy
        
        div = dudx + dvdy
        return div
    
    L = 2*np.pi
    dx, dy = L/M, 1/N

    # initialize grid for u
    x = np.empty(shape = (M, N)); y = np.empty(shape = (M, N))
    for i in range(0, M):
        for j in range(0, N):
            x[i, j] = i*dx
            y[i, j] = (j + 1/2)*dy
    u = np.sin(2*np.pi*x/L)
    
    # initialize grid for v
    x = np.empty(shape = (M, N+1)); y = np.empty(shape = (M, N+1))
    for i in range(0, M):
        for j in range(0, N+1):
            x[i, j] = (i+1/2)*dx
            y[i, j] = j*dy
    v = np.sin(np.pi*y)

    div = DIV(u, v)

    # initialize grid for div
    x = np.empty(shape = (M, N)); y = np.empty(shape = (M, N))
    for i in range(0, M):
        for j in range(0, N):
            x[i, j] = (i+1/2)*dx
            y[i, j] = (j+1/2)*dy
    
    div_ana = (2*np.pi) * (np.cos(2*np.pi*x/L)/L + np.cos(np.pi*y)/2)

    plt.pcolor(x, y, div)
    plt.colorbar()
    plt.show()

def verify_curl(M, N): # successful
    """
    p = sin(2πx/L) cos(πy)
    dp/dx = 2π/L cos(2πx/L) cos(πy)
    dp/dy = -π sin(2πx/L) sin(πy)

    in order to to test this, I will initialize p on a central grid, and calculate the numeric curl, and compare it to the analytical on their respective staggered grids
    """

    def CURL(p):
        """
        p is on the center of each grid cell
        dp/dx is on the left edge of each grid cell
        dp/dy is on the bottom edge of each grid cell, and includes the top wall

        then both are offset and returned as centered on each grid cell
        """

        # calculate both dpdx and dpdy on the staggered grid
        dpdx = np.zeros(shape = (M, N))
        for i in range(1, M):
            dpdx[i, :] = (p[i, :] - p[i-1, :])/dx
        dpdx[0, :] = (p[0, :] - p[-1, :])/dx

        dpdy = np.ones(shape = (M, N+1))
        dpdy[:, 0] = 0; dpdy[:, N] = 0
        for j in range(1, N):
            dpdy[:, j] = (p[:, j] - p[:, j-1])/dy

        return dpdx, dpdy
    
    L = 2*np.pi
    dx, dy = L/M, 1/N

    # this is the centered, not staggered grid
    x = np.empty(shape = (M, N)); y = np.empty(shape = (M, N))
    for i in range(0, M):
        for j in range(0, N):
            x[i, j] = (i + 1/2)*dx
            y[i, j] = (j + 1/2)*dy
    
    p = np.sin(2*np.pi*x/L) * np.cos(np.pi*y)
    dpdx, dpdy = CURL(p)

    # this is the u-staggered grid
    x = np.empty(shape = (M, N)); y = np.empty(shape = (M, N))
    for i in range(0, M):
        for j in range(0, N):
            x[i, j] = i*dx
            y[i, j] = (j + 1/2)*dy
    dpdx_ana = 2*np.pi/L * np.cos(2*np.pi*x/L) * np.cos(np.pi*y)

    # plt.pcolor(x, y, dpdx-dpdx_ana)
    # plt.show()

    # plt.scatter(x, y, color = 'black')
    # plt.plot(
    #     np.array([0, L, L, 0, 0]),
    #     np.array([0, 0, 1, 1, 0]),
    #     color = 'blue', linestyle = 'dashed'
    # )
    # plt.colorbar()
    # plt.show()

    # this is the v-staggered grid
    x = np.empty(shape = (M, N+1)); y = np.empty(shape = (M, N+1))
    for i in range(0, M):
        for j in range(0, N+1):
            x[i, j] = (i+1/2)*dx
            y[i, j] = j*dy
    dpdy_ana = -np.pi * np.sin(2*np.pi*x/L) * np.sin(np.pi*y)

    plt.pcolor(x, y, dpdy - dpdy_ana)
    plt.colorbar()
    plt.show()

    e = np.max([
        np.max(np.abs(dpdx - dpdx_ana)),
        np.max(np.abs(dpdy - dpdy_ana)),
    ])
    return e

# verify_div(512, 512)
# verify_curl(64, 64)
def init_grids(M, N):
    xp = np.empty(shape = (M, N)); yp = np.empty(shape = (M, N))
    xu = np.empty(shape = (M, N)); yu = np.empty(shape = (M, N))
    for i in range(0, M):
        for j in range(0, N):
            xp[i, j] = (i + 1/2)*dx
            xu[i, j] = i*dx
            
            yp[i, j] = (j + 1/2)*dy
            yu[i, j] = (j + 1/2)*dy
    
    xv = np.empty(shape = (M, N+1)); yv = np.empty(shape = (M, N+1))
    for i in range(0, M):
        for j in range(0, N+1):
            xv[i, j] = (i + 1/2)*dx
            yv[i, j] = j*dy
        
    return xp, yp, xu, yu, xv, yv

# computational domain initialization
M, N, L = 512, 512, 2*np.pi
dx, dy = L/M, 1/N
xp, yp, xu, yu, xv, yv = init_grids(M, N)

# initialize velocity as random, with v = 0 at bottom/top boundary
# u = -L*np.sin(2*np.pi*xu/L) * np.cos(np.pi*yu)/(2*np.pi) + np.random.rand(*xu.shape)
# v = np.cos(2*np.pi*xv/L) * np.sin(np.pi*yv)/np.pi + np.random.rand(*xv.shape)

u = np.random.rand(M, N) - 1/2
v = np.random.rand(M, N+1) - 1/2

v[:, 0] = 0; v[:, N] = 0 # type: ignore
for j in range(1, N):
    v[:, j] -= np.mean(v[:, j]) # type: ignore

# plt.pcolor(xv, yv, v)
# plt.pcolor(xu, yu, u)

pre_div = DIV(u, v)
p = POISSON(-pre_div)
dpdx, dpdy = CURL(p)
u += dpdx; v += dpdy
post_div = DIV(u, v)

# plt.pcolor(xp, yp, pre_div)
plt.pcolor(xp, yp, post_div)

# plt.plot(
#     np.array([0, L, L, 0, 0]),
#     np.array([0, 0, 1, 1, 0]),
#     color = 'blue', linestyle = 'dashed'
# )

plt.colorbar()
plt.show()
