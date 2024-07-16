import numpy as np
import matplotlib.pyplot as plt

#imp_data = np.load("output_bayesrisk_dimcomp_pure_240711.npy")
#imp_data = np.load("output_bayesrisk_dimcomp_pure_240710.npy")
imp_data = np.load("output_bayesrisk_dimcomp_240710.npy")
#imp_data = np.load("output_bayesrisk_dimcomp_240709.npy")

pure = False

# Load data and average over repetitions of measurement
# [dimension][fid / NESS / w / t][type of measurement][N][samples]
fid_data = imp_data[:, 0, [0, 3, 2, 1], :, :1000]
ess_data = imp_data[:, 1, [0, 3, 2, 1], :, :]
wma_data = imp_data[:, 2, [0, 3, 2, 1], :, :]
dur_data = imp_data[:, 3, [0, 3, 2, 1], :, :]

n_meas = np.array([1, 10, 100])
nq = np.array([1, 2, 3, 4, 5])
meas = ['Random', 'RandomSeparable', 'Pauli', 'Random2POVM']
n_sample = len(fid_data[0, 0, 0])

# Plot fidelities and duration
fig, ax1 = plt.subplots(figsize=(10, 7))

m_s = 6
l_w = 1
m_style = ['o', 's', 'D']
c_meas = ['firebrick', 'goldenrod', 'dodgerblue', 'olive']

fid_std = np.std(fid_data, axis= 3) / np.sqrt(n_sample) # average over ensemble
for j in range(4):
    for i in range(len(n_meas)):
        ax1.errorbar(nq + 0.12 * (j-1.5), np.average(fid_data[:, j, i, :], axis=1), yerr= fid_std[:, j, i], c= c_meas[j], lw= l_w, capsize= 6 * l_w, capthick= l_w, ls= "", marker= m_style[i], ms= m_s, zorder= 1)

# Legend
for i in range(len(n_meas)):
    ax1.errorbar([10], [10], yerr= [10], c= 'black', lw= l_w, capsize= 6 * l_w, capthick= l_w, ls= "", marker= m_style[i], label= "N = " + str(n_meas[i]), ms= m_s, zorder= 1)    
ax1.errorbar([10], [10], yerr= [10], c= 'white', lw= 0, label= " ", alpha= 1)
for j in range(4):
    ax1.errorbar([10], [10], yerr= [10], c= c_meas[j], lw= l_w, capsize= 6 * l_w, capthick= l_w, ls= "", marker= 'o', label= meas[j], ms= m_s, zorder= 1)

if pure:
    # Analytical solution for N = 1
    dim = 2**nq
    ax1.scatter(nq, 1 - (dim + 3) / ((dim + 1) **2), c= 'black', marker= 'X', label=r'$1-\frac{3+d}{(1+d)^2}$', zorder=2)
    ax1.scatter([10], [10], c= 'white', lw= 0, label= " ", alpha= 1)


fs = 12

ax1.set_xlim(0.7, 5.3)
ax1.set_xlabel(r'number of qubits $n_q$', fontsize=fs)
ax1.set_xticks([1, 2, 3, 4, 5])
ax1.set_ylabel(r'average risk $(1-\bar{F})$', fontsize=fs)
if pure: ax1.set_ylim(-0.025, 1.025)
else: ax1.set_ylim(-0.025, 0.325)
ax1.set_ylim(0.6, 1.025)
ax1.grid()
ax1.legend(fontsize= 10)

#plt.savefig("240709_Bayesrisk_dimcomp.png", dpi= 300)

plt.show()

print(np.average(1 - fid_data[0, :, 0, :]))