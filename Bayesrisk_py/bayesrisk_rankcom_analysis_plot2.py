import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

imp_data = np.load("data/output_bayesrisk_rankcomp_240624.npy")

fid_data = imp_data[:, 0, :, :, 0, :]
ess_data = imp_data[:, 1, :, :, 0, :]
wma_data = imp_data[:, 2, :, :, 0, :]
dur_data = imp_data[:, 3, :, :, 0, :]

n_meas = np.array([1, 4, 16, 64, 256, 1024])
nq = np.array([5, 4, 3, 2, 1])
meas = ['Pauli', 'Random', 'Random2POVM', 'RandomSeperable']

# Plot fidelities and duration
fig, ax1 = plt.subplots(figsize=(10, 7))

m_s = 6
l_w = 1
c_meas = ['purple', 'red', 'blue', 'green']

'''
for ind_i, i in enumerate([2, 0, 3, 1]):
    fid_std = np.std(1 - fid_data[:, i, 3, :], axis=1) / np.sqrt(1000)
    ax1.errorbar(nq + 0.1 * (ind_i-1.5), np.average(1 - fid_data[:, i, 3, :], axis=1), yerr= fid_std, c= c_meas[i], label= meas[i], lw= l_w, capsize= 6 * l_w, capthick=l_w, ls= "", marker='o', ms= m_s, zorder= 1)
'''
    
for i in range(len(n_meas)):
    fid_std = np.std(1 - fid_data[:, 0, i, :], axis=1) / np.sqrt(1000)
    ax1.errorbar(nq, np.average(1 - fid_data[:, 0, i, :], axis=1), yerr= fid_std, label= str(n_meas[i]), lw= l_w, capsize= 6 * l_w, capthick=l_w, ls= "", marker='o', ms= m_s, zorder= 1)

fs = 12
ax1.set_xlim(0.7, 5.3)
ax1.set_xlabel(r'number of qubits $nq$', fontsize=fs)
ax1.set_xticks([1, 2, 3, 4, 5])
ax1.set_ylabel(r'infidelity ($1-F$)', fontsize=fs)
ax1.set_yticks(np.linspace(0, 0.30, 7))
ax1.grid()
ax1.legend(fontsize= 10)

plt.show()