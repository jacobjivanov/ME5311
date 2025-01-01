import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv("HW2/simpson_integral_results.csv")
fourth = np.empty(len(results))
for i in range(len(results)):
    fourth[i] = float(results['N'][i]) ** -4

plt.loglog(results['N'], results['error'], 'o', color = 'blue', label = "Simpson Integration Error")
plt.loglog(results['N'], fourth, ":", color = 'red', label = '4th Order Convergence')
plt.legend()
plt.xlabel(r"$N$")
plt.ylabel(r"$e$")
plt.ylim([1e-18, 1e-2])
plt.title(r"Simpson Integration Error over $N$")
plt.savefig("HW2/Simpson Integration Error over N.png", dpi = 200)