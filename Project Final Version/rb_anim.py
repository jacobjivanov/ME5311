import numpy as np
import sys
import numba

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

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
    u, v = CENTER_VELOCITY(data['u'], data['v'])
    θ = data['θ']
    t = data['t']

    fig, ax = plt.subplots(2, 1)
    temp = ax[0].pcolormesh(x_cent, y_cent, θ, clim = (0, 1), cmap = 'gist_heat')

    start = np.array([
        np.concatenate((x_cent[:, 0], x_cent[:, 0], x_cent[:, 0])),
        np.concatenate((np.zeros(M) + 1/4, np.zeros(M) + 1/2, np.zeros(M) + 3/4))
    ]).T

    umag = (u**2 + v**2)**0.5
    vel = ax[1].pcolormesh(x_cent, y_cent, umag, cmap = 'Blues')
    color = (u**2 + v**2)**0.5
    norm = Normalize(vmin = 0, vmax = umag.max())
    stream = ax[1].streamplot(x_cent.T, y_cent.T, u.T, v.T, linewidth = 0.2, color = 'grey', start_points = start)

    temp_bar = fig.colorbar(temp)
    temp_bar.set_label(r"$\theta$")
    vel_bar = fig.colorbar(vel, norm = norm)
    vel_bar.set_label(r"$|\vec{u}|$")

    ax[1].set_xlabel(r"$x$")
    ax[0].set_ylabel(r"$y$")
    ax[1].set_ylabel(r"$y$")

    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    
    fig.suptitle("Rayleigh-Benard Convection, Pr = {0}, Ra = {1:.0e}\n".format(Pr, Ra) + r"$t = $" + " {0:.6f}".format(t))

    def update_plot(n, vel, stream):
        print("frame {0}/{1}".format(n, data_end), end = '\r')

        data = np.load("{0}/data n = {1}.npz".format(folder_name, n))
        u, v = CENTER_VELOCITY(data['u'], data['v'])
        θ = data['θ']
        t = data['t']
       
        fig.suptitle("Rayleigh-Benard Convection, Pr = {0}, Ra = {1:.0e}\n".format(Pr, Ra) + r"$t = $" + " {0:.6f}".format(t))

        temp.set_array(θ)
        
        umag = (u**2 + v**2)**0.5
        
        if n % (data_int * 4) == 0:
            ax[1].cla()
            stream = ax[1].streamplot(x_cent.T, y_cent.T, u.T, v.T, linewidth = 0.2, color = 'grey', start_points = start)

            vel = ax[1].pcolormesh(x_cent, y_cent, umag, cmap = 'Blues')

        vel.set_array(umag)
        vel.set_norm(Normalize(vmin = 0, vmax = umag.max()))

        ax[1].set_xlim(0, L)
        ax[1].set_ylim(0, 1)

        ax[1].set_xlabel(r"$x$")
        ax[0].set_ylabel(r"$y$")
        ax[1].set_ylabel(r"$y$")

        return vel, stream

    ANI = FuncAnimation(fig, update_plot, fargs = (vel, stream), init_func = lambda : None, frames = range(0, data_end, data_int))
    ANI.save("{0}.mp4".format(folder_name), dpi = 200, fps = 60, writer = 'ffmpeg')