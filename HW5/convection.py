import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 101)
u = np.zeros(101)

dx = 0.01
dt = 0.01

def rk3_step(dudt, t0, u0, dt):
    dudt0 = dudt(
        t0, 
        u0)
    
    dudt1 = dudt(
        t0 + dt/2,
        u0 + dt/2 * dudt0)
    
    dudt2 = dudt(
        t0 + dt,
        u0 + dt * (-dudt0 + 2*dudt1))

    u3 = u0 + dt * (dudt0/6 + 2*dudt1/3 + dudt2/6)

    return u3

def dudt(t, u):
    dudt = np.zeros(shape=u.shape)
    dudt[0] = 8 * np.cos(8*t)
    for i in range(1, len(u) - 1):
        dudt[i] = (u[i-1] - u[i+1]) / (2 * dx)
    dudt[-1] = (u[-2] - u[-1]) / dx
    return dudt

t = 0
while t < 0.5:
    u = rk3_step(dudt, t, u, dt)
    t += dt
u05 = u

while t < 1.0:
    u = rk3_step(dudt, t, u, dt)
    t += dt
u10 = u

while t < 1.5:
    u = rk3_step(dudt, t, u, dt)
    t += dt
u15 = u

while t < 2.0:
    u = rk3_step(dudt, t, u, dt)
    t += dt
u20 = u


plt.title(r"Solution of Model Advection PDE")
plt.xlabel("$x$")
plt.ylabel("$u(t, x)$")
plt.plot(x, u05, label = "$t = 0.5$")
plt.plot(x, u10, label = "$t = 1.0$")
plt.plot(x, u15, label = "$t = 1.5$")
plt.plot(x, u20, label = "$t = 2.0$")
plt.legend(loc = 'upper right')
plt.savefig("Solution of Model Advection PDE.png", dpi = 200)