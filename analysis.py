from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import lst

_, n_backround = lst.load("data/background.lst")
n_backround_avg = np.average(n_backround)

t, n = lst.load("data/measurement.lst")
t = 5 * t
n = n - n_backround_avg

t_max = 55
t_min_fast = 0
t_max_fast = 10
t_min_slow = 15
t_max_slow = 50

# Method 1
k_fast, m_fast = np.polyfit(t[t_min_fast:t_max_fast], np.log(n[t_min_fast:t_max_fast]), deg=1)
k_slow, m_slow = np.polyfit(t[t_min_slow:t_max_slow], np.log(n[t_min_slow:t_max_slow]), deg=1)

t_fast = 5 * np.arange(t_min_fast, t_max_fast)
n_fast = [np.exp(k_fast*t + m_fast) for t in t_fast]
t_slow = 5 * np.arange(t_min_slow, t_max_slow)
n_slow = [np.exp(k_slow*t + m_slow) for t in t_slow]

t_half_fast = - np.log(2) / k_fast
t_half_slow = - np.log(2) / k_slow

def linear_model(t, k, m):
    return k*t + m

_, pcov = curve_fit(linear_model, 
                    t[t_min_fast:t_max_fast], 
                    np.log(n[t_min_fast:t_max_fast]))
k_fast_err, _ = np.sqrt(np.diag(pcov))
t_half_fast_err = np.log(2) * np.abs(1 / (k_fast**2)) * k_fast_err


_, pcov = curve_fit(linear_model, 
                    t[t_min_slow:t_max_slow], 
                    np.log(n[t_min_slow:t_max_slow]))
k_slow_err, _ = np.sqrt(np.diag(pcov))
t_half_slow_err = np.log(2) * np.abs(1 / (k_slow**2)) * k_slow_err

print(f"t_half_fast = {t_half_fast:.1f}s ± {t_half_fast_err:.1f}s")
print(f"t_half_slow = {t_half_slow:.1f}s ± {t_half_slow_err :.1f}s")

plt.figure(figsize=(10, 6))

# Poissant error and actual data plotting for Method 1
for i in range(-10, 10):
    factor = i / 10
    plt.semilogy(t[:t_max], n[:t_max] + factor * np.sqrt(n[:t_max]), color="lightgrey", linewidth=5, linestyle="-")

plt.semilogy(t[:t_max], n[:t_max] + np.sqrt(n[:t_max]), color="lightgrey", linewidth=5, linestyle="-", label="Poissant error")
plt.semilogy(t[:t_max], n[:t_max], color="black", label="Measured beta-particles")
plt.semilogy(t_fast, n_fast, color="red", linewidth=2, label=f"e^({k_fast:.2f}*t+{m_fast:.2f})")
plt.semilogy(t_slow, n_slow, color="blue", linewidth=2, label=f"e^({k_slow:.2f}*t+{m_slow:.2f})")

plt.xlabel("Time [seconds]", fontsize=16)
plt.ylabel("Number of Beta- particles [0.2 N/s]", fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()

# Method 2
def model(t, n1, tau1, n2, tau2):
    return n1*np.exp(-t/tau1) + n2*np.exp(-t/tau2)

popt, pcov = curve_fit(model, t[:t_max], n[:t_max])
_, tau1_opt, _, tau2_opt = popt
_, tau1_err, _, tau2_err = np.sqrt(np.diag(pcov))

print(f"t_half_1 = {np.log(2)*tau1_opt:.1f}s ± {np.log(2)*tau1_err:.1f}s")
print(f"t_half_2 = {np.log(2)*tau2_opt:.1f}s ± {np.log(2)*tau2_err:.1f}s")

n_fit = model(t[:t_max], *popt)

plt.figure(figsize=(10, 6))

# Poissant error and actual data plotting for Method 2
for i in range(-10, 10):
    factor = i / 10
    plt.semilogy(t[:t_max], n[:t_max] + factor * np.sqrt(n[:t_max]), color="lightgrey", linewidth=5, linestyle="-")

plt.semilogy(t[:t_max], n[:t_max] + np.sqrt(n[:t_max]), color="lightgrey", linewidth=5, linestyle="-", label="Poissant error")
plt.semilogy(t[:t_max], n[:t_max], color="black", label="Measured beta-particles")
plt.semilogy(t[:t_max], n_fit, color="red", linewidth=2, label=f"{popt[0]:.0f}*exp(-t/{popt[1]:.1f})+{popt[2]:.0f}*exp(-t/{popt[3]:.1f})")

plt.xlabel("Time [seconds]", fontsize=16)
plt.ylabel("Number of Beta- particles [0.2 N/s]", fontsize=16)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()
