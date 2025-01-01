import numpy as np
import matplotlib.pyplot as plt

def rk3_step(dudt, u0, dt):
    dudt0 = dudt(u0)
    u1 = u0 + dt/2 * dudt0
    
    dudt1 = dudt(u1)
    u2 = u0 + dt * (-dudt0 + 2*dudt1)

    dudt2 = dudt(u2)
    u3 = u0 + dt * (dudt0/6 + 2*dudt1/3 + dudt2/6)

    return u3

def lp(e, dt, p):
    lp_error = 0
    for i in range(0, len(e)):
        lp_error += np.power(e[i], p)
    lp_error *= dt
    lp_error = np.power(lp_error, 1./p)

    return lp_error

def f(u): 
    beta = 8./3
    rho = 26
    sigma = 11

    u_t = np.array([
        sigma*(u[1] - u[0]),
        u[0]*(rho - u[2]) - u[1],
        u[0]*u[1] - beta*u[2]
    ])
    return u_t

def integrate(f, u0, t_end, T):
    t = np.linspace(0, t_end, T)
    u = np.zeros(shape = (T, 3))
    u[0, :] = u0
    n = 0
    while n < T-1:
        u[n+1, :] = rk3_step(f, u[n, :], t[1])
        n += 1
    
    return t, u

u0 = np.array([1.17, 1.1, 0.92])
t, u = integrate(f, u0, 40, 40000)

plt.plot(t, u[:, 0], label = r"$x$")
plt.plot(t, u[:, 1], label = r"$y$")
plt.plot(t, u[:, 2], label = r"$z$")
plt.xlabel("$t$")
plt.ylabel("position")
plt.title("RK3 Integration of Lorenz System, " + r"$\vec{u}(0) = [1.17, 1.1, 0.92]$")
plt.legend()
plt.savefig("RK3 Integration of Lorenz System.png", dpi = 200)

plt.cla()
t, u1 = integrate(f, 1.01*u0, 40, 40000)
plt.plot(t, u1[:, 0] - u[:, 0], label = r"$\Delta x$")
plt.plot(t, u1[:, 1] - u[:, 1], label = r"$\Delta y$")
plt.plot(t, u1[:, 2] - u[:, 2], label = r"$\Delta z$")
plt.xlabel("$t$")
plt.ylabel("position")
plt.title("Difference in 1% Perturbed Lorenz System, " + r"$\vec{u}(0) = [1.17, 1.1, 0.92]$")
plt.legend()
plt.savefig("Difference in 1% Perturbed Lorenz System.png", dpi = 200)