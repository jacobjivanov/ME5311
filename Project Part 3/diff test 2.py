import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

N = 16
Δy = 1/N

y_cent = np.empty(N)
for j in range(0, N):
    y_cent[j] = (j + 1/2)*Δy

# u_cent = 1/(y_cent + 1) - y_cent/2
u_cent = 1 - y_cent + np.sin(2*pi*y_cent)


u_diff = np.empty(N)
u_diff[0] = (2 - 3*u_cent[0] + u_cent[1])/Δy**2
u_diff[N-1] = (u_cent[N-2] - 3*u_cent[N-1])/Δy**2
for j in range(1, N-1):
    u_diff[j] = (u_cent[j-1] - 2*u_cent[j] + u_cent[j+1])/Δy**2

plt.plot(y_cent, u_diff, ':.')
plt.show()