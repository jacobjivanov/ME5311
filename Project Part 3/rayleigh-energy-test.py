import numpy as np
from numpy import pi, real, imag
from numpy.fft import fft, ifft

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

"""
def DIFF(u, v, θ):
    diff_u = np.empty(shape = (M, N))
    diff_u[:, 0] = (-3*u[:, 0] + u[:, 1])/Δy**2
    diff_u[:, N-1] = (u[:, N-2] - 3*u[:, N-1])/Δy**2
    for j in range(1, N-1):
        diff_u[:, j] = (u[:, j-1] -2*u[:, j] + u[:, j+1])/Δy**2

    diff_v = np.empty(shape = (M, N+1))
    diff_v[:, 0] = 0; diff_v[:, N] = 0
    for j in range(1, N):
        diff_v[:, j] = (v[:, j-1] - 2*v[:, j] + v[:, j+1])/Δy**2

    diff_θ = np.empty(shape = (M, N))
    diff_θ[:, 0] = (2 - 3*θ[:, 0] + θ[:, 1])/Δy**2
    diff_θ[:, N-1] = (θ[:, N-2] - 3*θ[:, N-1])/Δy**2

    for i in range(0, M):
        diff_u[i, :] += (u[i-1, :] - 2*u[i, :] + u[(i+1)%M, :])/Δx**2
        diff_v[i, :] += (v[i-1, :] - 2*v[i, :] + v[(i+1)%M, :])/Δx**2
        diff_θ[i, :] += (θ[i-1, :] - 2*θ[i, :] + θ[(i+1)%M, :])/Δx**2

    return diff_u, diff_v, diff_θ

def BUOY(θ):
    buoy_v = np.empty(shape = (M, N+1))
    buoy_v[:, 0] = 0; buoy_v[:, N] = 0
    for j in range(1, N):
        buoy_v[:, j] = (θ[:, j] + θ[:, j-1])/2

    return buoy_v

def RHS(u, v, θ):
    conv_u, conv_v, conv_θ = CONV(u, v, θ)
    diff_u, diff_v, diff_θ = DIFF(u, v, θ)
    buoy_v = BUOY(θ)

    return conv_u, conv_v, conv_θ
    # return conv_u + diff_u, conv_v + diff_v, conv_θ + diff_θ
    # return conv_u + diff_u, conv_v + diff_v + buoy_v, conv_θ + diff_θ
"""

def CORRECT(ustar, vstar):
    vstar[:, 0] = 0; vstar[:, N] = 0
    for j in range(1, N):
        vstar[:, j] -= np.mean(vstar[:, j])

    div = DIV(ustar, vstar)
    p = POISSON(-div)
    dpdx, dpdy = CURL(p)
    u = ustar + dpdx; v = vstar + dpdy

    return u, v

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

def ENERGY(u, v, θ):
    Ek = np.sum(u**2) + np.sum(v**2)
    Et = np.sum(θ**2)
    return Ek, Et

"""
def FINDΔt(u, v):
    Δt_diff = cfl_diff/Pr * (Δx**2 * Δy**2)/(Δx**2 + Δy**2)
    Δt_conv = 1e10
    for i in range(0, M):
        for j in range(0, N):
            Δt_conv = np.min([
                Δt_conv,
                cfl_conv/(np.abs(u[i, j]/Δx + np.abs(v[i, j]/Δy)))
            ])
    
    Δt_next = np.min([Δt_diff, Δt_conv])
    return Δt_next

def CENTER_VELOCITY(u, v):
    u_cent = np.empty(shape = (M, N))
    for i in range(0, M-1):
        u_cent[i, :] = (u[i+1, :] + u[i, :])/2
    u_cent[M-1, :] = (u[0, :] + u[M-1, :])/2

    v_cent = np.empty(shape = (M, N))
    for j in range(0, N):
        v_cent[:, j] = (v[:, j+1] + v[:, j])/2

    return u_cent, v_cent
"""

if __name__ == '__main__':
    # runtime parameters
    M, N, L = 64, 64, 2*pi
    Δt = 0.01

    # initialize grid
    Δx, Δy = L/M, 1/N
    x_cent, y_cent, x_left, y_left, x_bott, y_bott = INIT_GRID(M, N)

    # initialize flow fields
    u = np.random.rand(M, N)/10 - 1/20
    v = np.random.rand(M, N+1)/10 - 1/220
    u, v = CORRECT(u, v)
    θ = 1 - y_cent
    θ[:, 1:N-1] += np.random.rand(M, N-2)/10 - 1/20
    

    # timestepping
    n = 0; t = 0; t_end = 10
    while t < t_end:
        Ek, Et = ENERGY(u, v, θ)
        D = np.max(np.abs(DIV(u, v)))
        print("time: {0:5f}, step: {1:02d}, Kinetic: {2:.5e}, Thermal: {3:.5e}, Max Div: {4:.5e}".format(t, n, Ek, Et, D))
        
        u, v, θ = RK3(CONV, u, v, θ, Δt)
        n += 1
        t += Δt

    """
    # fig, ax = plt.subplots(2, 1)
    # ax[0].pcolor(x_cent, y_cent, θ, cmap = 'gist_heat', vmin = 0, vmax = 1)
    
    # u_cent, v_cent = CENTER_VELOCITY(u, v)    
    # ax[1].streamplot(x_cent.T, y_cent.T, u_cent.T, v_cent.T, density = 0.75)
    ax[1].set_xlim(0, L)
    ax[1].set_ylim(0, 1)
    
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    plt.show()
    """