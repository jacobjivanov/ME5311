import numpy as np
from numpy import pi

import matplotlib.pyplot as plt

def U_ANA(x, y):
    # satisfies u(x, y=0) = 0, u(x, y=1) = 0
    return np.sin(2*pi*x/L)*np.sin(pi*y)
def V_ANA(x, y):
    # satisfies v(x, y=0) = 0, v(x, y=1) = 0
    return np.cos(2*pi*x/L)*np.sin(pi*y)
def θ_ANA(x, y):
    # satisfies θ(x, y=0) = 1, θ(x, y=1) = 0
    return np.cos(2*pi*x/L + 1)*np.sin(pi*y) - y+1

def UCONV_ANA(x, y):
    return np.cos(2*pi*x/L)*np.sin(2*pi*x/L)*np.sin(pi*y) * (-4*pi*np.sin(pi*y)/L - 2*pi*np.cos(pi*y))
def VCONV_ANA(x, y):
    return 2*pi/L * np.sin(pi*y)**2 * (np.sin(2*pi*x/L)**2 - np.cos(2*pi*x/L)**2) - 2*pi*np.cos(2*pi*x/L)**2 * np.cos(pi*y) * np.sin(pi*y)
def θCONV_ANA(x, y):
    return -(2*pi*np.sin(pi*y)*np.cos(2*pi*x/L)*(np.sin(pi*y)*np.cos(2*pi*x/L+1)-y+1) - 2*pi*np.sin(pi*y)**2*np.sin(2*pi*x/L)*np.sin(2*pi*x/L+1))/L - np.cos(2*np.pi*x/L) * ((2*np.pi*np.cos(2*np.pi*x/L+1)*np.cos(np.pi*y) - 1)*np.sin(np.pi*y) + (np.pi - np.pi*y)*np.cos(np.pi*y))
def θCONVX_ANA(x, y):
    return -(2*pi*np.sin(pi*y)*np.cos(2*pi*x/L)*(np.sin(pi*y)*np.cos(2*pi*x/L+1)-y+1) - 2*pi*np.sin(pi*y)**2*np.sin(2*pi*x/L)*np.sin(2*pi*x/L+1))/L
def θCONVY_ANA(x, y):
    return -np.cos(2*np.pi*x/L) * ((2*np.pi*np.cos(2*np.pi*x/L+1)*np.cos(np.pi*y) - 1)*np.sin(np.pi*y) + (np.pi - np.pi*y)*np.cos(np.pi*y))

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

def CENTER_VELOCITY(u, v): # success
    u_cent = np.empty(shape = (M, N))
    for i in range(0, M):
        u_cent[i, :] = (u[i, :] + u[(i+1)%M, :])/2 # right-centered average
    
    v_cent = np.empty(shape = (M, N))
    for j in range(0, N):
        v_cent[:, j] = (v[:, j] + v[:, j+1])/2 # centered average
    
    return u_cent, v_cent

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

def CONV_TERM(uu_stag, uv_stag, vv_stag, uθ_stag, vθ_stag):
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

if __name__ == '__main__':
    M, N, L = 512, 512, 2*pi
    dx, dy = L/M, 1/N

    x_cent, y_cent, x_left, y_left, x_bott, y_bott, x_gu, y_gu, x_gv, y_gv = INIT_GRID(M, N)

    u = U_ANA(x_left, y_left)
    v = V_ANA(x_bott, y_bott)
    θ = θ_ANA(x_cent, y_cent)

    u_cent, v_cent = CENTER_VELOCITY(u, v)
    u_add, v_add, θ_add = ADD_GHOSTS(u_cent, v_cent, θ)
    uu_cent, uv_cent, vv_cent, uθ_cent, vθ_cent = FLUX_CENT(u_add, v_add, θ_add)
    uu_stag, uv_stag, vv_stag, uθ_stag, vθ_stag = FLUX_STAG(uu_cent, uv_cent, vv_cent, uθ_cent, vθ_cent)
    u_conv, v_conv, θ_conv = CONV_TERM(uu_stag, uv_stag, vv_stag, uθ_stag, vθ_stag)

    # plt.pcolor(x_cent, y_cent, uu_cent[:, 1:N+1] - U_ANA(x_cent, y_cent)**2) # success
    # plt.pcolor(x_cent, y_cent, uv_cent[:, 1:N+1] - U_ANA(x_cent, y_cent)*V_ANA(x_cent, y_cent)) # success
    # plt.pcolor(x_cent, y_cent, vv_cent[:, 1:N+1] - V_ANA(x_cent, y_cent)**2) # success
    # plt.pcolor(x_cent, y_cent, uθ_cent[:, 1:N+1] - U_ANA(x_cent, y_cent)*θ_ANA(x_cent, y_cent)) # success
    # plt.pcolor(x_cent, y_cent, vθ_cent[:, 1:N+1] - V_ANA(x_cent, y_cent)*θ_ANA(x_cent, y_cent)) # success

    # plt.pcolor(x_left, y_left, u_conv - UCONV_ANA(x_left, y_left)) # success
    # plt.pcolor(x_bott, y_bott, v_conv - VCONV_ANA(x_bott, y_bott)) # success
    plt.pcolor(x_cent, y_cent, θ_conv - θCONV_ANA(x_cent, y_cent)) # success
    plt.colorbar()
    plt.show()