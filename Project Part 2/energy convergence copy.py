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
    A = np.ones(N+1)/Δy**2; A[N] = -1
    C = np.ones(N+1)/Δy**2; C[0] = -1
    D = np.zeros(shape = (M, N+2), dtype = 'complex')
    D[:, 1:N+1] += fft(d, axis = 0)
    P = np.zeros(shape = d.shape, dtype = 'complex')

    for i in range(1, M):
        k = kp[i]
        # set up (changing) main diagonal coefficient vector
        B = np.ones(N+2, dtype = 'complex')
        B[0] = 1; B[N+1] = 1
        for j in range(0, N):
            B[j+1] = -2./Δy**2 + (-2 + 2*np.cos(k*Δx))/Δx**2    
        
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
        dudx[i, :] = (u[i+1, :] - u[i, :])/Δx
    dudx[-1, :] = (u[0, :] - u[-1, :])/Δx

    dvdy = np.empty(shape = (M, N))
    for j in range(0, N):
        dvdy[:, j] = (v[:, j+1] - v[:, j])/Δy
    
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
        dpdx[i, :] = (p[i, :] - p[i-1, :])/Δx
    dpdx[0, :] = (p[0, :] - p[-1, :])/Δx

    dpdy = np.zeros(shape = (M, N+1))
    dpdy[:, 0] = 0; dpdy[:, N] = 0
    for j in range(1, N):
        dpdy[:, j] = (p[:, j] - p[:, j-1])/Δy

    return dpdx, dpdy

@numba.njit()
def CONV(u, v, θ):
    # CALCULATE CONVECTION FLUXES
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
    
    # CALCULATE CONVECTION TERMS
    conv_u = np.zeros(shape = (M, N))
    for i in range(0, M):
        conv_u[i, :] -= (fu[i, :] - fu[i-1, :])/Δx
    for j in range(0, N):
        conv_u[:, j] -= (gu[:, j+1] - gu[:, j])/Δy
    
    conv_v = np.zeros(shape = (M, N+1))
    for i in range(0, M):
        conv_v[i, :] -= (fv[i, :] - fv[i-1, :])/Δx
    for j in range(1, N):
        conv_v[:, j] -= (gv[:, j] - gv[:, j-1])/Δy

    conv_θ = np.zeros(shape = (M, N))
    for i in range(0, M):
        conv_θ[i, :] -= (fθ[(i+1)%M, :] - fθ[i, :])/Δx
    for j in range(0, N):
        conv_θ[:, j] -= (gθ[:, j+1] - gθ[:, j])/Δy
    
    return conv_u, conv_v, conv_θ

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
    # dudt0, dvdt0 = 2*(u1-u0)/Δt, 2*(v1-v0)/Δt

    dudt1, dvdt1, dθdt1 = f(u1, v1, θ1)
    u2star = u0 + Δt * (-dudt0 + 2*dudt1)
    v2star = v0 + Δt * (-dvdt0 + 2*dvdt1)
    θ2 = θ0 + Δt * (-dθdt0 + 2*dθdt1)
    u2, v2 = CORRECT(u2star, v2star)
    # dudt1, dvdt1 = (u2-u0)/Δt, (v2-v0)/Δt

    dudt2, dvdt2, dθdt2 = f(u2, v2, θ2)
    u3star = u0 + Δt * (dudt0/6 + 2*dudt1/3 + dudt2/6)
    v3star = v0 + Δt * (dvdt0/6 + 2*dvdt1/3 + dvdt2/6)
    θ3 = θ0 + Δt * (dθdt0/6 + 2*dθdt1/3 + dθdt2/6)
    u3, v3 = CORRECT(u3star, v3star)

    return u3, v3, θ3

@numba.njit()
def ENERGY(u, v, θ):
    Ek = np.sum(u**2) + np.sum(v**2)
    Et = np.sum(θ**2)
    return Ek, Et

if __name__ == '__main__':
    M, N, L = 400, 50, 2*pi
    Δx, Δy = L/M, 1/N

    x_cent, y_cent, x_left, y_left, x_bott, y_bott = INIT_GRID(M, N)

    dt = np.array([0.075, 0.05, 0.02, 0.01, 0.005, 0.0025])#, 0.0001])
    ΔEk = np.empty(dt.size)
    ΔEt = np.empty(dt.size)

    for i in range(0, dt.size):
        Δt = dt[i]

        u = np.random.rand(M, N) - 1/2
        v = np.random.rand(M, N+1) - 1/2
        u, v = CORRECT(u, v)

        θ = np.random.rand(M, N)
        θ[:, 0] = 1; θ[:, N-1] = 0
        
        Ek0, Et0 = ENERGY(u, v, θ)

        n = 0; t = 0
        while t < 10:
            Ek, Et = ENERGY(u, v, θ)
            D = np.max(np.abs(DIV(u, v)))
            print("step: {0:02d}, Kinetic: {1:.5e}, Thermal: {2:.5e}, Max Div: {3:.5e}".format(n, Ek, Et, D))
            
            u, v, θ = RK3(CONV, u, v, θ, Δt)
            n += 1
            t += Δt
        
        Ekf, Etf = ENERGY(u, v, θ)
        ΔEk[i] = Ek0 - Ekf
        ΔEt[i] = Et0 - Etf
    
    plt.title("Energy Decay Convergence")
    plt.xlabel(r"$\Delta t$")
    plt.ylabel(r"$\Delta E$")
    plt.loglog(dt, ΔEk, ':.', color = 'blue', label = 'kinetic decay')
    plt.loglog(dt, ΔEt, ':.', color = 'red', label = 'thermal decay')
    plt.loglog(dt, dt**3, ':.', color = 'grey', label = '3rd Order Convergence')
    plt.legend()
    plt.savefig("energy_convergence.png", dpi = 400)
    plt.show()