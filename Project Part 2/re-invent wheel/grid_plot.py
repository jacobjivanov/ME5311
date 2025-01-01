import numpy as np
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

if __name__ == '__main__':
    M, N, L = 7, 4, 2*np.pi
    dx, dy = L/M, 1/N
    x_cent, y_cent, x_left, y_left, x_bott, y_bott, x_gu, y_gu, x_gv, y_gv = INIT_GRID(M, N)

    # plt.scatter(x_cent, y_cent, color = 'black', marker = '+', label = r"$p, \nabla \cdot \vec{u}$")
    # plt.scatter(x_left, y_left, color = 'red', marker = 'o', label = r"$u, \frac{dp}{dx}$")
    # plt.scatter(x_bott, y_bott, color = 'blue', marker = 'x', label = r"$v, \frac{dp}{dy}$")
    # plt.plot(
    #     np.array([0, L, L, 0, 0]),
    #     np.array([0, 0, 1, 1, 0]),
    #     linestyle = 'dashed', color = 'grey',
    #     label = 'domain'
    # )
    # plt.legend(bbox_to_anchor = (1, 1.15), ncols = 4)
    # plt.show()

    # plt.scatter(x_left, y_left, color = 'black', marker = '+', label = r"$u$")
    # plt.scatter(x_cent, y_cent, color = 'red', marker = 'o', label = r"$f_u$")
    # plt.scatter(x_gu, y_gu, color = 'blue', marker = 'o', label = r"$g_u$")
    # plt.plot(
    #     np.array([0, L, L, 0, 0]),
    #     np.array([0, 0, 1, 1, 0]),
    #     linestyle = 'dashed', color = 'grey',
    #     label = 'domain'
    # )
    # plt.legend(bbox_to_anchor = (1, 1.15), ncols = 4)
    # plt.savefig("u_locs.png", dpi = 200)

    # plt.scatter(x_bott, y_bott, color = 'black', marker = '+', label = r"$v$")
    # plt.scatter(x_gu, y_gu, color = 'red', marker = 'o', label = r"$f_v$")
    # plt.scatter(x_gv, y_gv, color =  'blue', marker = 'o', label = r"$g_v$")
    # plt.plot(
    #     np.array([0, L, L, 0, 0]),
    #     np.array([0, 0, 1, 1, 0]),
    #     linestyle = 'dashed', color = 'grey',
    #     label = 'domain'
    # )
    # plt.legend(bbox_to_anchor = (1, 1.15), ncols = 4)
    # plt.savefig("v_locs.png", dpi = 200)

    plt.scatter(x_cent, y_cent, color = 'black', marker = '+', label = r"$\theta$")
    plt.scatter(x_left, y_left, color = 'red', marker = 'o', label = r"$f_\theta$")
    plt.scatter(x_bott, y_bott, color = 'blue', marker = 'o', label = r"$g_\theta$")
    plt.plot(
        np.array([0, L, L, 0, 0]),
        np.array([0, 0, 1, 1, 0]),
        linestyle = 'dashed', color = 'grey',
        label = 'domain'
    )
    plt.legend(bbox_to_anchor = (1, 1.15), ncols = 4)
    plt.savefig("θ_locs.png", dpi = 200)

