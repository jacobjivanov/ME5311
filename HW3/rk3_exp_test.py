import numpy as np
import matplotlib.pyplot as plt

def rk3_step(dydt, y0, dt):
    dydt0 = dydt(y0)
    y1 = y0 + dt/2 * dydt0
    
    dydt1 = dydt(y1)
    y2 = y0 + dt * (-dydt0 + 2*dydt1)

    dydt2 = dydt(y2)
    y3 = y0 + dt * (dydt0/6 + 2*dydt1/3 + dydt2/6)

    return y3

def lp(e, dt, p):
    lp_error = 0
    for i in range(0, len(e)):
        lp_error += np.power(e[i], p)
    lp_error *= dt
    lp_error = np.power(lp_error, 1./p)

    return lp_error

def f(y): return y

def integrate(f, y0, t_end, T):
    t = np.linspace(0, t_end, T)
    y = np.zeros(T)
    y[0] = y0

    n = 0
    while n < T-1:
        y[n+1] = rk3_step(f, y[n], t[1])
        n += 1
    
    return y

T = np.linspace(2, 20, 19)
T = np.power(2, T)
l1e = np.zeros(len(T))
l2e = np.zeros(len(T))
for i in range(0, len(T)):
    y = integrate(f, 1, 2, int(T[i]))
    t = np.linspace(0, 2, int(T[i]))
    y_ana = np.exp(t)

    e = np.abs(y - y_ana)
    l1e[i] = lp(e, t[1], 1)
    l2e[i] = lp(e, t[1], 2)

plt.loglog(T, l1e, ':o', color = 'green', label = r"$\ell_1 [e]$")
plt.loglog(T, l2e, ':o', color = 'red', label = r"$\ell_2 [e]$")
plt.loglog(T, 1/T**3, ':o', color = 'blue', label = "3rd Order Convergence")
plt.xlabel(r"$N$")
plt.ylabel("Error")
plt.ylim(1e-14, 1e-1)
plt.title("RK3 Error for " + r"$y' = y, \, y(0) = 1$")
plt.legend()
plt.savefig("RK3 Error Convergence (exp).png", dpi = 200)