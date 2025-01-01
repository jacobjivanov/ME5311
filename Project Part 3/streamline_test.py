import numpy as np
from numpy import pi

M, N, L = 256, 256, 2
Δx = L/M; Δy = 1/N

x_cent = np.empty(shape = (M, N))
y_cent = np.empty(shape = (M, N))
for i in range(0, M):
    for j in range(0, N):
        x_cent[i, j] = (i + 1/2)*Δx
        y_cent[i, j] = (j + 1/2)*Δy
u_cent = np.sin(2*pi*x_cent/L)
v_cent = np.cos(2*pi*y_cent)

import matplotlib.pyplot as plt
# plt.scatter(x_cent, y_cent)
# plt.plot(np.array([0, L, L, 0, 0]), np.array([0, 0, 1, 1, 0]), linestyle = 'dashed')

eo = 16
plt.pcolor(x_cent, y_cent, (u_cent**2 + v_cent**2)**1/2)
plt.quiver(x_cent[::eo, ::eo], y_cent[::eo, ::eo], u_cent[::eo, ::eo], v_cent[::eo, ::eo])
plt.streamplot(x_cent.T, y_cent.T, u_cent.T, v_cent.T)
plt.show()