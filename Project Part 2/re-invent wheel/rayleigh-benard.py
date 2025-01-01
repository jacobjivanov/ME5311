import numpy as np
from numpy import pi, real, imag
from numpy.fft import fft, ifft
import numba
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

@numba.njit()
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

@numba.njit()
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

@numba.njit()
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

@numba.njit()
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

@numba.njit()
def CENTER_VELOCITY(u, v): # success
    u_cent = np.empty(shape = (M, N))
    for i in range(0, M):
        u_cent[i, :] = (u[i, :] + u[(i+1)%M, :])/2 # right-centered average
    
    v_cent = np.empty(shape = (M, N))
    for j in range(0, N):
        v_cent[:, j] = (v[:, j] + v[:, j+1])/2 # centered average
    
    return u_cent, v_cent

@numba.njit()
def CONV_TERM(u, v, θ):
    def ADD_GHOSTS(u_cent, v_cent, θ): # success
        u_add = np.empty(shape = (M, N+2))
        u_add[:, 1:N+1] = u_cent
        u_add[:, 0] = -u_cent[:, 0]; u_add[:, N+1] = -u_cent[:, N-1]
        
        v_add = np.empty(shape = (M, N+2))
        v_add[:, 1:N+1] = v_cent
        v_add[:, 0] = -v_cent[:, 0]; v_add[:, N+1] = -v_cent[:, N-1]

        θ_add = np.empty(shape = (M, N+2))
        θ_add[:, 1:N+1] = θ
        θ_add[:, 0] = 2-θ[:, 0]; θ_add[:, N+1] = -θ[:, N-1]

        return u_add, v_add, θ_add

    def FLUX_CENT(u_add, v_add, θ_add): # success
        uu_cent = u_add*u_add
        uv_cent = u_add*v_add
        vv_cent = v_add*v_add
        uθ_cent = u_add*θ_add
        vθ_cent = v_add*θ_add

        return uu_cent, uv_cent, vv_cent, uθ_cent, vθ_cent

    def FLUX_STAG(uu_cent, uv_cent, vv_cent, uθ_cent, vθ_cent): # success
        uu_stag = uu_cent[:, 1:N+1] # already in position, just remove ghosts

        vu_stag = np.empty(shape = (M, N+1)) # these two are really in the same place
        uv_stag = np.empty(shape = (M, N+1)) # one is an intermediate variable
        for j in range(0, N+1):
            vu_stag[:, j] = (uv_cent[:, j] + uv_cent[:, j+1])/2 # centered average
        for i in range(0, M):
            uv_stag[i, :] = (vu_stag[i-1, :] + vu_stag[i, :])/2 # left-centered average

        vv_stag = vv_cent # already in position

        uθ_stag = np.empty(shape = (M, N))
        for i in range(0, M):
            uθ_stag[i, :] = (uθ_cent[i-1, 1:N+1] + uθ_cent[i, 1:N+1])/2 # left-centered average
        
        vθ_stag = np.empty(shape = (M, N+1))
        for j in range(0, N+1):
            vθ_stag[:, j] = (vθ_cent[:, j] + vθ_cent[:, j+1])/2 # centered average

        return uu_stag, uv_stag, vv_stag, uθ_stag, vθ_stag
    
    u_cent, v_cent = CENTER_VELOCITY(u, v)
    u_add, v_add, θ_add = ADD_GHOSTS(u_cent, v_cent, θ)
    uu_cent, uv_cent, vv_cent, uθ_cent, vθ_cent = FLUX_CENT(u_add, v_add, θ_add)
    uu_stag, uv_stag, vv_stag, uθ_stag, vθ_stag = FLUX_STAG(uu_cent, uv_cent, vv_cent, uθ_cent, vθ_cent)

    u_conv = np.empty(shape = (M, N))
    for i in range(0, M):
        u_conv[i, :] = -(uu_stag[i, :] - uu_stag[i-1, :])/dx # left-sided derivative
    for j in range(0, N):
        u_conv[:, j] -= (uv_stag[:, j+1] - uv_stag[:, j])/dy # centered derivative
    
    v_conv = np.empty(shape = (M, N+1))
    for i in range(0, M):
        v_conv[i, :] = -(uv_stag[(i+1)%M, :] - uv_stag[i, :])/dx # right-sided derivative
    for j in range(0, N+1):
        v_conv[:, j] -= (vv_stag[:, j+1] - vv_stag[:, j])/dy # centered derivative
    
    θ_conv = np.empty(shape = (M, N))
    
    for i in range(0, M):
        θ_conv[i, :] = -(uθ_stag[(i+1)%M, :] - uθ_stag[i, :])/dx # right-sided derivative
    for j in range(0, N):
        θ_conv[:, j] -= (vθ_stag[:, j+1] - vθ_stag[:, j])/dy # centered derivative
    
    return u_conv, v_conv, θ_conv

@numba.njit()
def CORRECT(ustar, vstar):
    vstar[:, 0] = 0; vstar[:, N] = 0
    for j in range(1, N):
        vstar[:, j] -= np.mean(vstar[:, j])

    div = DIV(ustar, vstar)
    p = POISSON(-div)
    dpdx, dpdy = CURL(p)
    u = ustar + dpdx; v = vstar + dpdy

    return u, v

@numba.njit()
def RK3(f, u0, v0, θ0, Δt):
    dudt0, dvdt0, dθdt0 = f(u0, v0, θ0)
    u1star = u0 + Δt * (dudt0/2)
    v1star = v0 + Δt * (dvdt0/2)
    θ1 = θ0 + Δt * (dθdt0/2)
    u1, v1 = CORRECT(u1star, v1star)
    dudt0, dvdt0 = 2*(u1-u0)/Δt, 2*(v1-v0)/Δt

    dudt1, dvdt1, dθdt1 = f(u1, v1, θ1)
    u2star = u0 + Δt * (-dudt0 + 2*dudt1)
    v2star = v0 + Δt * (-dvdt0 + 2*dvdt1)
    θ2 = θ0 + Δt * (-dθdt0 + 2*dθdt1)
    u2, v2 = CORRECT(u2star, v2star)
    dudt1, dvdt1 = (u2-u0)/Δt, (v2-v0)/Δt

    dudt2, dvdt2, dθdt2 = f(u2, v2, θ2)
    u3star = u0 + Δt * (dudt0/6 + 2*dudt1/3 + dudt2/6)
    v3star = v0 + Δt * (dvdt0/6 + 2*dvdt1/3 + dvdt2/6)
    θ3 = θ0 + Δt * (dθdt0/6 + 2*dθdt1/3 + dθdt2/6)
    u3, v3 = CORRECT(u3star, v3star)

    return u3, v3, θ3

@numba.njit()
def ENERGY(u, v, θ):
    u, v = CENTER_VELOCITY(u, v)
    
    Ek = np.sum(u**2 + v**2)
    Et = np.sum(θ**2)
    return Ek, Et

if __name__ == '__main__':
    M, N, L = 64, 64, 2*pi
    dx, dy = L/M, 1/N
    cfl_max = 1.6
    Δt = 2*cfl_max/(1/dx + 1/dy)

    x_cent, y_cent, x_left, y_left, x_bott, y_bott = INIT_GRID(M, N)

    u = np.random.rand(M, N) - 1/2
    v = np.random.rand(M, N+1) - 1/2
    u, v = CORRECT(u, v)

    θ = np.random.rand(M, N)
    θ[:, 0] = 1; θ[:, N-1] = 0
    
    n = 0
    while n < 100:
        u_cent, v_cent = CENTER_VELOCITY(u, v)
        plt.pcolor(x_cent, y_cent, v_cent)
        plt.colorbar()
        plt.show()
        Ek, Et = ENERGY(u, v, θ)
        print("step: {0:02d}, Kinetic: {1:.5e}, Thermal: {2:.5e}".format(n, Ek, Et))
        
        u, v, θ = RK3(CONV_TERM, u, v, θ, Δt)
        n += 1

    # plt.pcolor(x_cent, y_cent, post_div)
    # plt.colorbar()
    # plt.show()