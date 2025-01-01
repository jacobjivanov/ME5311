import numpy as np
from numpy import pi
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
        
    x_gu = np.empty(shape = (M, N+1)); y_gu = np.empty(shape = (M, N+1))
    for i in range(0, M):
        for j in range(0, N+1):
            x_gu[i, j] = i*dx
            y_gu[i, j] = j*dy
    
    x_gv = np.empty(shape = (M, N+2)); y_gv = np.empty(shape = (M, N+2))
    for i in range(0, M):
        for j in range(0, N+2):
            x_gv[i, j] = (i + 1/2)*dx
            y_gv[i, j] = (j - 1/2)*dy

    return x_cent, y_cent, x_left, y_left, x_bott, y_bott, x_gu, y_gu, x_gv, y_gv

# @numba.njit()


def CONV_TERM(u, v, θ):
    def CONV_FLUX(u, v, θ): # Calculate the convection fluxes
        """Calculate the convection fluxes

        Inputs:
            u (M x N): x-component of velocity
                on (i*dx, (j+1/2)*dy) grid
            v (M x N+1): y-component of velocity
                on ((i+1/2)*dx, j*dy) grid
            θ (M x N): non-dimensional temperature
                on ((i+1/2)*dx, (j+1/2)*dy) grid

        Outputs:
            f_u (M x N): x-component of u-flux
                on ((i+1/2)*dx, (j+1/2)*dy) grid
            g_u (M x N): y-component of u-flux
                on ((i+1/2)*dx, (j+1/2)*dy) grid
            f_v (M x N): x-component of v-flux

        """

        def CENTER_VELOCITY(u, v): # Calculate component velocites on centered grid
            """Calculate component velocites on centered grid

            Inputs:
                u (M x N): x-component of velocity
                    on (i*dx, (j+1/2)*dy) grid
                v (M x N+1): y-component of velocity
                    on ((i+1/2)*dx, j*dy) grid

            Outputs:

            """
            u_cent = np.empty(shape = (M, N))
            for i in range(0, M-1):
                u_cent[i, :] = (u[i+1, :] + u[i, :])/2
            u_cent[M-1, :] = (u[0, :] + u[M-1, :])/2

            v_cent = np.empty(shape = (M, N))
            for j in range(0, N):
                v_cent[:, j] = (v[:, j+1] + v[:, j])/2

            return u_cent, v_cent

        def ADD_GHOSTS(u, v, θ):
            """Add grid values for centered ghost cells above and below walls, 
            in order to satisfy u(x, y=0) = u(x, y=1) = 0, v(x, y=0) = v(x, y=0) = 0
            and θ(x, y=0) = 1, θ(x, y=1) = 0
            
            """
            
            u_add = np.empty(shape = (M, N+2))
            u_add[:, 1:N+1] = u
            u_add[:, 0] = - u[:, 0]
            u_add[:, N+1] = - u[:, N-1]

            v_add = np.empty(shape = (M, N+2))
            v_add[:, 1:N+1] = v
            v_add[:, 0] = - v[:, 0]
            v_add[:, N+1] = - v[:, N-1]

            θ_add = np.empty(shape = (M, N+2))
            θ_add[:, 1:N+1] = θ
            θ_add[:, 0] = 1 - θ[:, 0]
            θ_add[:, N+1] = -θ[:, N-1]

            return u_add, v_add, θ_add

        u, v = CENTER_VELOCITY(u, v)
        u, v, θ = ADD_GHOSTS(u, v, θ)

        f_u = u**2
        g_u = v * u

        f_v = u * v
        g_v = v**2

        f_θ = u * θ 
        g_θ = v * θ

        def STAG_FLUX(f_u, g_u, f_v, g_v, f_θ, g_θ):
            """In order to make taking the x- and y-derivatives of the convective
            fluxes more logical, the previously calculated fluxes will be staggered such that:
            f_u (M x N):
                on ((i+1/2)*dx, (j+1/2)*dy) grid
            g_u (M x N+1):
                on (i*dx, j*dy) grid
            f_v (M x N+1):
                on (i*dx, j*dy) grid
            g_v (M x N+2):
                on ((i+1/2)*dx, (j-1/2)*dy) grid
            f_θ (M x N):
                on (i*dx, (j+1/2)*dy) grid
            g_θ (M x N+1):
                on ((i+1/2)*dx, j*dy) grid
            """

            # fix boundary conditions so that there is no flux on top and bottom wall
            f_u[:, 0] = -f_u[:, 1]; f_u[:, N+1] = -f_u[:, N]
            g_u[:, 0] = -g_u[:, 1]; g_u[:, N+1] = -g_u[:, N]
            f_v[:, 0] = -f_v[:, 1]; f_v[:, N+1] = -f_v[:, N]
            g_v[:, 0] = -g_v[:, 1]; g_v[:, N+1] = -g_v[:, N]
            f_θ[:, 0] = -f_θ[:, 1]; f_θ[:, N+1] = -f_θ[:, N]
            g_θ[:, 0] = -g_θ[:, 1]; g_θ[:, N+1] = -g_θ[:, N]


            fu = f_u[:, 1:N+1]
            
            gu = np.empty(shape = (M, N+1))
            fv = np.empty(shape = (M, N+1))
            for j in range(0, N+1):
                gu[:, j] = (g_u[:, j] + g_u[:, j+1])/2
            for i in range(0, M):
                fv[i, :] = (gu[i-1, :] + gu[i, :])/2
            gu = fv

            gv = g_v

            fθ = np.empty(shape = (M, N+2))
            for i in range(0, M):
                fθ[i, :] = (f_θ[i-1, :] + f_θ[i, :])/2
            fθ = fθ[:, 1:N+1]
            
            gθ = np.empty(shape = (M, N+1))
            for j in range(0, N+1):
                gθ[:, j] = (g_θ[:, j] + g_θ[:, j+1])/2
            
            return fu, gu, fv, gv, fθ, gθ

        f_u, g_u, f_v, g_v, f_θ, g_θ = STAG_FLUX(f_u, g_u, f_v, g_v, f_θ, g_θ)

        return f_u, g_u, f_v, g_v, f_θ, g_θ
    
    f_u, g_u, f_v, g_v, f_θ, g_θ = CONV_FLUX(u, v, θ)

    conv_u = np.empty(shape = (M, N))
    for i in range(0, M):
        conv_u[i, :] = (f_u[i-1, :] - f_u[i, :])/dx
    for j in range(0, N):
        conv_u[:, j] += (g_u[:, j] - g_u[:, j+1])/dy

    conv_v = np.empty(shape = (M, N+1))
    for i in range(0, M):
        conv_v[i, :] = (f_v[i, :] - f_v[(i+1)%M, :])/dx
    for j in range(0, N+1):
        conv_v[:, j] += (g_v[:, j] - g_v[:, j+1])/dy
    
    conv_θ = np.empty(shape = (M, N))
    for i in range(0, M):
        conv_θ[i, :] = (f_θ[i, :] - f_θ[(i+1)%M, :])/dx
    for j in range(0, N):
        conv_θ[:, j] += (g_θ[:, j] - g_θ[:, j+1])/dy
    
    return conv_u, conv_v, conv_θ

if __name__ == '__main__':
    M, N, L = 64, 64, 2*np.pi
    dx, dy = L/M, 1/N
    x_cent, y_cent, x_left, y_left, x_bott, y_bott, x_gu, y_gu, x_gv, y_gv = INIT_GRID(M, N)

    def u_func(x):
        return np.cos(2*pi/L*x)
    def v_func(y):
        return np.sin(pi*y)
    def θ_func(y):
        return y + np.sin(2*pi*y)

    u = u_func(x_left)
    v = v_func(y_bott)
    θ = θ_func(y_cent)

    conv_u, conv_v, conv_θ = CONV_TERM(u, v, θ)

    conv_uana = 4*pi/L * np.cos(2*pi*x_left/L) * np.sin(2*pi*x_left/L) - pi*np.cos(2*pi*x_left/L)*np.cos(pi*y_left)
    conv_vana = 2*pi/L*np.sin(pi*y_bott)*np.sin(2*pi*x_bott/L) - 2*pi*np.cos(pi*y_bott)*np.sin(pi*y_bott)
    conv_θana = 2*pi/L * (np.sin(2*pi*y_cent) + y_cent)*np.sin(2*pi*x_cent/L) - pi*np.cos(pi*y_cent) * (np.sin(2*pi*y_cent) + y_cent) - np.sin(pi*y_cent)*(2*pi*np.cos(2*pi*y_cent) + 1)

    # plt.pcolor(x_left, y_left, conv_u - conv_uana)
    # plt.pcolor(x_bott, y_bott, conv_v - conv_vana)
    # plt.pcolor(x_cent, y_cent, conv_θ - conv_θana)
    plt.colorbar()
    plt.show()