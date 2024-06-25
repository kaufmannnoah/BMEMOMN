import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

imp_data = np.load("data/output_bayesrisk_rankcomp_240624.npy")
#imp_data = np.load("output.npy")

fid_data = imp_data[:, 0, :, :, 0, :]
ess_data = imp_data[:, 1, :, :, 0, :]
wma_data = imp_data[:, 2, :, :, 0, :]
dur_data = imp_data[:, 3, :, :, 0, :]

n_meas = np.array([1, 4, 16, 64, 256, 1024])
nq = np.array([5, 4, 3, 2, 1])

# Plot fidelities and duration
fig, ax1 = plt.subplots(figsize=(10, 7))

#fit exponential
coef = np.zeros((len(nq), 4, 2))
coef_std = np.zeros((len(nq), 4, 2))
for j in range(len(nq)):
    for i in range(4):
        coef[j][i] = sp.optimize.curve_fit(lambda t, b, c: b * np.exp(-c * t), n_meas, np.average(1 - fid_data[j, i, :, :], axis=1), method= 'lm')[0]
        coef_std[j][i] = np.sqrt(np.diag(sp.optimize.curve_fit(lambda t, b, c: b * np.exp(-c * t), n_meas, np.average(1 - fid_data[j, i, :, :], axis=1))[1]))

m_s = 6
l_w = 2
c_meas = ['purple', 'red', 'orange', 'green', 'blue']

x_t = np.linspace(1, n_meas[-1], 1000)
for j in range(len(nq)):
    for i in range(4):
        fid_std = np.std(1 - fid_data[j, i, :, :], axis=1) / np.sqrt(1000)
        ax1.errorbar(n_meas, np.average(1 - fid_data[j, i, :, :], axis=1), yerr= fid_std, c= c_meas[j], lw= 1, ls= "", marker='o', ms= m_s, zorder= 1)
for j in range(len(nq)):
    for i in range(4):
        temp_temp = 0
        #ax1.plot(x_t, coef[j, i, 0] * np.exp(-coef[j, i, 1] * x_t), ls='-', c=c_meas[j], lw= l_w, alpha= 0.5, zorder=0)

fs = 12
ax1.set_xscale('log', base=10)
ax1.set_xlim(0.8, 1200)
ax1.set_xlabel(r'number of measurements $M$', fontsize=fs)
ax1.set_xticks([1, 10, 100, 1000])
ax1.set_ylabel(r'infidelity ($1-F$)', fontsize=fs)
ax1.set_yticks(np.linspace(0, 0.30, 7))
#ax1.set_ylim(-0.025, 0.325)
ax1.grid()
ax1.legend(fontsize= 10)

plt.show()