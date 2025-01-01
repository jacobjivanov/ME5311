import numpy as np
import sys

import matplotlib.pyplot as plt

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
    
    t = np.empty(data_end//data_int)
    q_top = np.empty(data_end//data_int)
    q_bot = np.empty(data_end//data_int)

    for n in range(0, data_end, data_int):
        print("timestep {0}/{1}".format(n, data_end), end = '\r')

        data = np.load("{0}/data n = {1}.npz".format(folder_name, n))
        θ = data['θ']
        t[n//data_int] = data['t']

        q_top[n//data_int] = N/L * np.sum(θ[:, 0])
        q_bot[n//data_int] = N/L * np.sum(1 - θ[:, N-1])
    
    plt.title("Heat Flux From Domain Walls over Time")
    plt.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0,0))
    plt.plot(t, q_top, color = 'blue', label = r"$y = 1$")
    plt.plot(t, q_bot, color = 'red', label = r"$y = 0$")
    plt.xlabel(r"time, $t$")
    plt.ylabel(r"lengthwise averaged heat flux, $|\bar{q}''|$")
    plt.legend()
    plt.savefig("{0} heat flux.png".format(folder_name), dpi = 400)