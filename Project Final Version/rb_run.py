"""
The following script was developed entirely by Jacob Ivanov, in order to study the effects of increasing the Rayleigh Number for the Rayleigh-Benard Instability. It utilizes a Second Order Spatial, Third Order Time accurate method overall.

The script should be run as follows:
python3 rb_run.py M N L Pr Ra data_int

where `M`, `N` are the computational domain dimensions in the x-, and y-direction. The domain is periodic in x over length `L`. `Pr` is the Prandtl Number (usually 0.7 or 1). `Ra` is the Rayleigh Number which can range from 0 to 1e8. The script saves the data every `data_int` timesteps in a temporary folder. It continues until the `t_end` parameter, which has been hard-coded to 0.02, which is where the intability is fully developed.

Reccomended runs:
python3 rb_run.py 128 64 2.0 0.7 1e+06 1 (generates 138 MB of data)
python3 rb_run.py 512 256 2.0 0.7 1e+07 10 (generates 3.44 GB of data)
python3 rb_run.py 1024 512 2.0 0.7 1e+08 50 (generates 20.74 GB of data)

Must install `rocket-fft` package or disable `numba.njit()` JIT compilation.
"""

import numpy as np
from numpy import pi, real, imag
from numpy.fft import fft, ifft

import numba
import sys
import os
import shutil
import matplotlib.pyplot as plt

@numba.njit()
def INIT_GRID(M, N):
    """
    Generates the grids for plotting purposes:
    `M`, `N` are the computational domain dimensions
    """
    
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
    """
    Solves the tridiagonal linear system
    a_i x_{i-1}  + b_i x_i  + c_i x_{i+1}  = d_i,
    using the algorithm described here: 
    https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    
    Note: vectors `a` and `c` should be one index shorter than vectors `b` and `d`.
    """

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
    Solves the Poisson Equation assuming x-periodicity
    `p` is on the center of each grid cell
    `d` is on the center of each grid cell
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
            B[j+1] = -2./Δy**2 + (-2 + 2*np.cos(2*pi*k/M))/Δx**2
        
        # solve tridiagonal system for each column
        P[i, :] = TRIDIAG(A, B, C, D[i, :])[1:N+1]
    p = real(ifft(P, axis = 0))

    return p

@numba.njit()
def DIV(u, v):
    """
    Computes the divergence of the staggered velocity field
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
    Computes the curl of the pressure field
    `p` is on the center of each grid cell
    `dpdx` is on the left center of each grid cell
    `dpdy` is on the bottom center of each grid cell, and includes the top wall
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
    """
    Calculates the convection component of the RHS.
    Computes the convection fluxes in divergence form (conservative) according to the method described in "Fully Conservative Higher Order Finite Difference Schemes for Incompressible Flow" by Morinishi et al.

    `fu` is on the center of each grid cell
    `gu` is on the bottom left corner of each grid cell
    `fv` is on the bottom right corner of each grid cell
    `gv` is on the center of each grid cell
    `fθ` is on the left center of each grid cell
    `gθ` is on the bottom center of each grid cell

    Computes the convection terms using a finite difference such that each convection term lines up with the velocity it is convecting.
    `conv_u` is on the left center of each grid cell
    `conv_v` is on the bottom center of each grid cell
    `conv_θ` is on the center of each grid cell
    """
    
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
def DIFF(u, v, θ):
    """
    Calculates the diffusion component of the RHS

    Computes the diffusion terms using a finite difference such that each diffusion term lines up with the velocity it is diffusing.
    `diff_u` is on the left center of each grid cell
    `diff_v` is on the bottom center of each grid cell
    `diff_θ` is on the center of each grid cell
    """

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
    for j in range(1, N-1):
        diff_θ[:, j] = (θ[:, j-1] -2*θ[:, j] + θ[:, j+1])/Δy**2
    
    for i in range(0, M):
        diff_u[i, :] += (u[i-1, :] - 2*u[i, :] + u[(i+1)%M, :])/Δx**2
        diff_v[i, :] += (v[i-1, :] - 2*v[i, :] + v[(i+1)%M, :])/Δx**2
        diff_θ[i, :] += (θ[i-1, :] - 2*θ[i, :] + θ[(i+1)%M, :])/Δx**2

    return diff_u, diff_v, diff_θ

@numba.njit()
def BUOY(θ):
    """
    Calculates the buoyancy component of the RHS, 
    Uses the Boussinesq Approximation. Lines up with the y-velocity terms.
    """
    
    buoy_v = np.empty(shape = (M, N+1))
    buoy_v[:, 0] = 0; buoy_v[:, N] = 0
    for j in range(1, N):
        buoy_v[:, j] = (θ[:, j] + θ[:, j-1])/2

    return buoy_v

@numba.njit()
def RHS(u, v, θ):
    """
    Calculates the full RHS using `Pr` and `Ra` to non-dimensionalize all the variables.
    """
    
    conv_u, conv_v, conv_θ = CONV(u, v, θ)
    diff_u, diff_v, diff_θ = DIFF(u, v, θ)
    buoy_v = BUOY(θ)

    return conv_u + Pr*diff_u, conv_v + Pr*diff_v + Pr*Ra*buoy_v, conv_θ + diff_θ

@numba.njit()
def CORRECT(ustar, vstar):
    """
    Corrects the velocity fields
    In order to maintain divergence free flow, the pressure field is computed, and the the curl of the pressure field is added to the velocity field.
    """
    
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
    """
    Timesteps using Kutta's 3rd Order method
    Maintains divergence free flow intermediate to each substep.

    """
    
    dudt0, dvdt0, dθdt0 = f(u0, v0, θ0)
    u1star = u0 + Δt * (dudt0/2)
    v1star = v0 + Δt * (dvdt0/2)
    θ1 = θ0 + Δt * (dθdt0/2)
    u1, v1 = CORRECT(u1star, v1star)

    dudt1, dvdt1, dθdt1 = f(u1, v1, θ1)
    u2star = u0 + Δt * (-dudt0 + 2*dudt1)
    v2star = v0 + Δt * (-dvdt0 + 2*dvdt1)
    θ2 = θ0 + Δt * (-dθdt0 + 2*dθdt1)
    u2, v2 = CORRECT(u2star, v2star)

    dudt2, dvdt2, dθdt2 = f(u2, v2, θ2)
    u3star = u0 + Δt * (dudt0/6 + 2*dudt1/3 + dudt2/6)
    v3star = v0 + Δt * (dvdt0/6 + 2*dvdt1/3 + dvdt2/6)
    θ3 = θ0 + Δt * (dθdt0/6 + 2*dθdt1/3 + dθdt2/6)
    u3, v3 = CORRECT(u3star, v3star)

    return u3, v3, θ3

@numba.njit()
def ENERGY(u, v, θ):
    """
    Computes the total kinetic and thermal energy
    Used as a diagnostic to make sure run is not 'blowing up'.
    """
    
    Ek = np.sum(u**2) + np.sum(v**2)
    Et = np.sum(θ**2)
    return Ek, Et

@numba.njit()
def FINDΔt(u, v):
    """
    Calculates the next Δt to maintain stability
    Computes the maximum Δt to maintain diffusion stability, and compares it to the minimum, maximum Δt value to maintian convection stability. Returns the minimum. 
    """


    Δt_diff = cfl_diff/Pr * (Δx**2 * Δy**2)/(Δx**2 + Δy**2)
    Δt_conv = 1e10
    for i in range(0, M):
        for j in range(0, N):
            Δt_conv = min([
                Δt_conv,
                cfl_conv/(np.abs(u[i, j]/Δx + np.abs(v[i, j]/Δy))) if u[i, j] + v[i, j] != 0 else 1e10
            ])
    
    Δt_next = min([Δt_diff, Δt_conv])
    return Δt_next

if __name__ == '__main__':
    # read in runtime parameters
    M = int(sys.argv[1])
    N = int(sys.argv[2])
    L = float(sys.argv[3])
    Pr = float(sys.argv[4])
    Ra = float(sys.argv[5])
    data_int = int(sys.argv[6])
    
    cfl_diff = 0.2; cfl_conv = 1.0

    # initialize grid
    Δx, Δy = L/M, 1/N
    x_cent, y_cent, x_left, y_left, x_bott, y_bott = INIT_GRID(M, N)

    # initialize flow fields
    u = np.random.rand(M, N)/10 - 1/20
    v = np.random.rand(M, N+1)/10 - 1/20
    u, v = CORRECT(u, v)

    θ = 1 - y_cent
    θ[:, 1:N-1] += np.random.rand(M, N-2)/10 - 1/20
    
    # initialize data folder
    folder_name = "{0} {1} {2} {3} {4:.0e}".format(M, N, L, Pr, Ra)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print("No Previous Run. Folder Created")
    else:
        for filename in os.listdir(folder_name):
            file_path = os.path.join(folder_name, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        print("Previous Run Existed. Folder Cleared")

    # timestepping
    n = 0; t = 0; t_end = 0.02; Δt = 0
    while t < t_end:
        if n % data_int == 0:
            np.savez("{0}/data n = {1}.npz".format(folder_name, n), u = u, v = v, θ = θ, t = t)
        
        if n % 5 == 0:
            Ek, Et = ENERGY(u, v, θ)
            D = np.max(np.abs(DIV(u, v)))
            print("n: {0}, t: {1:5f}, Δt: {2:.5e}, Ek: {3:.5e}, Et: {4:.5e}, D: {5:.5e}".format(n, t, Δt, Ek, Et, D), end = "\r")

        Δt = FINDΔt(u, v)
        u, v, θ = RK3(RHS, u, v, θ, Δt)
        n += 1
        t += Δt
    
    # final message
    print("n: {0}, t: {1:5f}, Δt: {2:.5e}, Ek: {3:.5e}, Et: {4:.5e}, D: {5:.5e}".format(n, t, Δt, Ek, Et, D), end = "\n")
