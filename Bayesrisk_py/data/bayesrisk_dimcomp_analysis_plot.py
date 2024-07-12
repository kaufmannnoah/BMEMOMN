import numpy as np
import matplotlib.pyplot as plt

imp_data = np.load("output_bayesrisk_dimcomp_240626.npy")

# Load data and average over repetitions of measurement
# [dimension][fid / NESS / w / t][type of measurement][N][rho_0][rep]
fid_data = imp_data[::-1, 0, :, :, :, :]
ess_data = np.average(imp_data[::-1, 1, :, :, :, :], axis= (3, 4))
wma_data = np.average(imp_data[::-1, 2, :, :, :, :], axis= (3, 4))
dur_data = np.average(imp_data[::-1, 3, :, :, :, :], axis= (3, 4))
fid_data = fid_data.reshape(*fid_data.shape[:-2], -1)

n_meas = np.array([1, 10, 100])
nq = np.array([1, 2, 3, 4])
meas = ['Pauli', 'Random', 'Random2POVM', 'RandomSeparable']
n_sample = len(fid_data[0, 0, 0])

# Plot fidelities and duration
fig, ax1 = plt.subplots(figsize=(10, 7))

m_s = 6
l_w = 1
m_style = ['o', 's', 'D']
c_meas = ['firebrick', 'goldenrod', 'dodgerblue', 'olive']

fid_std = np.std(fid_data, axis= 3) / np.sqrt(n_sample) # average over ensembl
for indj, j in enumerate([1, 3, 0, 2]):
    for i in range(len(n_meas)):
        ax1.errorbar(nq + 0.12 * (indj-1.5), np.average(fid_data[:, j, i, :], axis=1), yerr= fid_std[:, j, i], c= c_meas[indj], lw= l_w, capsize= 6 * l_w, capthick= l_w, ls= "", marker= m_style[i], ms= m_s, zorder= 1)

# Legend
for i in range(len(n_meas)):
    ax1.errorbar([10], [10], yerr= [10], c= 'black', lw= l_w, capsize= 6 * l_w, capthick= l_w, ls= "", marker= m_style[i], label= "N = " + str(n_meas[i]), ms= m_s, zorder= 1)    
ax1.errorbar([10], [10], yerr= [10], c= 'white', lw= 0, label= " ", alpha= 1)
for indj, j in enumerate([1, 3, 0, 2]):
    ax1.errorbar([10], [10], yerr= [10], c= c_meas[indj], lw= l_w, capsize= 6 * l_w, capthick= l_w, ls= "", marker= 'o', label= meas[j], ms= m_s, zorder= 1)

fs = 12

ax1.set_xlim(0.7, 4.3)
ax1.set_xlabel(r'number of qubits $n_q$', fontsize=fs)
ax1.set_xticks([1, 2, 3, 4])
ax1.set_ylabel(r'average risk ($1-\bar{F}$)', fontsize=fs)
ax1.set_ylim(-0.025, 0.325)
ax1.set_ylim(0.95, 1.0)
#ax1.set_yticks(np.linspace(0, 0.30, 7))
ax1.grid()
ax1.legend(fontsize= 10)

#plt.savefig("240626_Bayesrisk_dimcomp.png", dpi= 300)

plt.show()
print( np.average(1 - fid_data[0, :, 0, :]))