import numpy as np
import sys
import numba

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

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
def CENTER_VELOCITY(u, v):
    u_cent = np.empty(shape = (M, N))
    for i in range(0, M-1):
        u_cent[i, :] = (u[i+1, :] + u[i, :])/2
    u_cent[M-1, :] = (u[0, :] + u[M-1, :])/2

    v_cent = np.empty(shape = (M, N))
    for j in range(0, N):
        v_cent[:, j] = (v[:, j+1] + v[:, j])/2

    return u_cent, v_cent

# @numba.njit()
def ENS(u, v):
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

    return vort_cent**2

if __name__ == '__main__':
    # read in runtime parameters
    M = int(sys.argv[1])
    N = int(sys.argv[2])
    L = float(sys.argv[3])
    Pr = float(sys.argv[4])
    Ra = float(sys.argv[5])
    data_int = int(sys.argv[6])
    data_end = int(sys.argv[7])

    folder_name = "{0} {1} {2} {3} {4:.0e}".format(M, N, L, Pr, Ra)

    Δx = L/M; Δy = 1/N
    x_cent, y_cent, x_left, y_left, x_bott, y_bott = INIT_GRID(M, N)

    # initialize figure
    n = 0
    data = np.load("{0}/data n = {1}.npz".format(folder_name, n))
    u, v = data['u'], data['v']
    θ = data['θ']
    t = data['t']

    u_cent, v_cent = CENTER_VELOCITY(u, v)
    ens = np.log10(ENS(u, v))
    umag = (u_cent**2 + v_cent**2)**0.25

    fig, ax = plt.subplots(3, 1, figsize = (6, 8))
    fig.suptitle("Rayleigh-Benard Convection, Pr = {0}, Ra = {1:.0e}\n".format(Pr, Ra) + r"$t = $" + " {0:.6f}".format(t))

    vel_norm = Normalize(vmin = 0, vmax = umag.max())
    ens_norm = Normalize(vmin = 0, vmax = ens.max())

    temp = ax[0].pcolormesh(x_cent, y_cent, θ, clim = (0, 1), cmap = 'gist_heat')
    vel = ax[1].pcolormesh(x_cent, y_cent, umag, cmap = 'GnBu', norm = vel_norm)
    enstr = ax[2].pcolor(x_cent, y_cent, ens, cmap = 'viridis', norm = ens_norm)

    temp_bar = fig.colorbar(temp)
    temp_bar.set_label(r"$\theta$")
    vel_bar = fig.colorbar(vel, norm = vel_norm)
    vel_bar.set_label(r"$\sqrt{|\vec{u}|}$")
    ens_bar = fig.colorbar(enstr, norm = ens_norm)
    ens_bar.set_label(r"$\log_{10} [\omega^2]$")

    ax[2].set_xlabel(r"$x$")
    ax[0].set_ylabel(r"$y$")
    ax[1].set_ylabel(r"$y$")
    ax[2].set_ylabel(r"$y$")

    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[2].set_aspect('equal')
    
    def update_plot(n):
        print("frame {0}/{1}".format(n, data_end), end = '\r')

        data = np.load("{0}/data n = {1}.npz".format(folder_name, n))
        u, v = data['u'], data['v']
        θ = data['θ']
        t = data['t']
       
        u_cent, v_cent = CENTER_VELOCITY(u, v)
        umag = (u_cent**2 + v_cent**2)**0.25
        ens = np.log10(ENS(u, v))

        fig.suptitle("Rayleigh-Benard Convection, Pr = {0}, Ra = {1:.0e}\n".format(Pr, Ra) + r"$t = $" + " {0:.6f}".format(t))

        temp.set_array(θ)
        vel.set_array(umag)
        enstr.set_array(ens)

        vel.set_norm(Normalize(vmin = 0, vmax = umag.max()))
        enstr.set_norm(Normalize(vmin = 0, vmax = ens.max()))

    ANI = FuncAnimation(fig, update_plot, init_func = lambda : None, frames = range(0, data_end, data_int))
    ANI.save("{0} ens.mp4".format(folder_name), dpi = 200, fps = 60, writer = 'ffmpeg')
