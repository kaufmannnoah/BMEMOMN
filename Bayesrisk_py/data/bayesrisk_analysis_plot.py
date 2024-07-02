import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

imp_data = np.load("data/output_bayesrisk_240619.npy")

fid_data = imp_data[0][:, :, 0, :]
ess_data = imp_data[1][:, :, 0, :]
wma_data = imp_data[2][:, :, 0, :]
dur_data = imp_data[3][:, :, 0, :]

n_meas = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])

# Plot fidelities and duration
fig, ax1 = plt.subplots(figsize=(10, 7))
ax2 = ax1.twinx()

#fit exponential
coef = np.zeros((3, 2))
coef_std = np.zeros((3, 2))
for i in range(3):
    coef[i] = sp.optimize.curve_fit(lambda t, b, c: b * np.exp(-c * t), n_meas, np.average(1 - fid_data[i, :, :], axis=1))[0]
    coef_std[i] = np.sqrt(np.diag(sp.optimize.curve_fit(lambda t, b, c: b * np.exp(-c * t), n_meas, np.average(1 - fid_data[i, :, :], axis=1))[1]))

m_s = 6
l_w = 2
c_meas = ['seagreen', 'indigo', 'firebrick']
label_meas = ["Pauli Basis", "Random Projector", "2-Outcome Projector"]

x_t = np.linspace(1, n_meas[-1], 1000)
for i in range(3):
    fid_std = np.std(1 - fid_data[i, :, :], axis=1) / np.sqrt(10000)
    ax1.errorbar(n_meas, np.average(1 - fid_data[i, :, :], axis=1), yerr= fid_std, c= c_meas[i], lw= 1, ls= "", marker='o', ms= m_s, label= label_meas[i], zorder= 1)
for i in range(3):
    ax1.plot(x_t, coef[i, 0] * np.exp(-coef[i, 1] * x_t), ls='-', c=c_meas[i], lw= l_w, alpha= 0.5, label= "coef= " + '{:.3E}'.format(coef[i, 1]) + r' $\pm$' + '{:.1E}'.format(coef_std[i, 1]), zorder= 0)

# RUNTIME
coef_t = np.polyfit(n_meas[:7], np.average(dur_data[1, :7, :], axis=1), 1)[0] 
x_t = np.linspace(1, n_meas[-1], 1000)
ax2.plot(x_t, coef_t * x_t, ls='-', c='gray', lw= l_w / 2, alpha= 0.5)
ax2.plot(n_meas, np.average(dur_data[1, :, :], axis=1),  c= 'gray', ls='-', marker='x', ms= m_s, lw= 0, alpha= 1, label='t Pauli meas')    

fs = 12
ax1.set_xscale('log', base=10)
ax1.set_xlim(0.8, 1200)
ax1.set_xlabel(r'number of measurements $M$', fontsize=fs)
ax1.set_xticks([1, 10, 100, 1000])
ax1.set_ylabel(r'infidelity ($1-F$)', fontsize=fs)
ax1.set_yticks(np.linspace(0, 0.30, 7))
ax1.set_ylim(-0.025, 0.325)
ax1.grid()
ax1.legend(fontsize= 10)
ax2.set_ylabel(r'runtime [s]', fontsize=fs)
ax2.grid(visible=0)
ax2.set_yticks(np.linspace(0, 8, 5))
ax2.set_ylim(-1.25, 16.25)
ax2.legend(fontsize= 10, loc='upper left')

plt.show()