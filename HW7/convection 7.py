import numpy as np
import matplotlib.pyplot as plt

def rk(dx, rk_type):
    dt = dx/10

    N = int(1//dx + 1)
    x = np.linspace(0, 1, N)
    u = np.cos(4 * np.pi * x)

    def dudt_v1(u):
        dudt = np.zeros(N)
        for i in range(0, N):
            dudt[i] = 1/dx * (-1/12 * (u[(i+2)%N] - u[(i-2)%N]) + 2/3 * (u[(i+1)%N] - u[(i-1)%N]))

        return dudt

    def dudt_v2(u):
        dudt = np.zeros(N)
        for i in range(0, N):
            dudt[i] = 1/dx * (-1/4 * (u[(i+2)%N] - u[(i-2)%N]) + (u[(i+1)%N] - u[(i-1)%N]))

        return dudt

    def rk3_step(dudt, u0, dt):
        dudt0 = dudt(
            u0)
        
        dudt1 = dudt(
            u0 + dt/2 * dudt0)
        
        dudt2 = dudt(
            u0 + dt * (-dudt0 + 2*dudt1))

        u3 = u0 + dt * (dudt0/6 + 2*dudt1/3 + dudt2/6)

        return u3

    t = 0
    while t < 1:
        if rk_type == 1:
            u = rk3_step(dudt_v1, u, dt)
        if rk_type == 2:
            u = rk3_step(dudt_v2, u, dt)
        t += dt
    
    return x, u

# Part A
# x1, u1 = rk(0.05, 1)
# x2, u2 = rk(0.05, 2)

# plt.plot(x1, u1, label = "Scheme 1")
# plt.plot(x2, u2, label = "Scheme 2")
# plt.plot(x1, np.cos(4 * np.pi * (x1 - 1)), label = "Exact")

# Part B
x1, u1 = rk(0.05, 2)
x2, u2 = rk(0.025, 2)
x3, u3 = rk(0.0125, 2)
x4, u4 = rk(0.00625, 2)

plt.plot(x1, u1, label = "$\Delta x = 0.05$")
plt.plot(x2, u2, label = "$\Delta x = 0.025$")
plt.plot(x3, u3, label = "$\Delta x = 0.0125$")
plt.plot(x4, u4, label = "$\Delta x = 0.00625$")
plt.plot(x4, np.cos(4 * np.pi * (x4 - 1)), label = "Exact")

plt.title(r"Solution of Model Advection PDE")
plt.xlabel("$x$")
plt.ylabel("$u(t, x)$")




plt.legend(loc = 'upper right')
plt.savefig("Periodic Advection Part B.png", dpi = 200)
# plt.show()