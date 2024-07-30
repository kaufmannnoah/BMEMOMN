import numpy as np
import matplotlib.pyplot as plt

title = ['BDS_dirichlet']
name = "MLE_HS.npy"
meas = ['Pauli_bds']
dim = [4]
n_meas = np.array([3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45])
n_sample = 1000

recon =['bayesian', 'MLE', 'direct']
metric =['fidelity', 'HS']

c_meas = ['firebrick', 'goldenrod', 'dodgerblue', 'olive']
m_s = 8 #markersize
l_w = 3 #linewidth
f_s = 12 #fontsize

fid = np.load(name)[[0, 2, 4], 0, 0, 0] #[estimator][nmeas][sample]
HS = np.load(name)[[1, 3, 5], 0, 0, 0]

fig, axs = plt.subplots(nrows=1, ncols=len(metric), figsize=(12, 4), layout="constrained")

x = np.linspace(3, 45, 1000)

for j in range(len(recon)):
    fid_std = np.std(1 - fid[j], axis=1) / np.sqrt(n_sample)
    HS_std = np.std(1 - HS[j], axis=1) / np.sqrt(n_sample)
    axs[0].errorbar(n_meas, np.average(1 - fid[j], axis=1), yerr= fid_std, c= c_meas[j], lw=l_w, ls= "", marker='o', ms= m_s, label= recon[j], alpha=1, zorder= 1)
    axs[1].errorbar(n_meas, np.average(HS[j], axis=1), yerr= fid_std, c= c_meas[j], lw=l_w, ls= "", marker='o', ms= m_s, label= recon[j], alpha=1, zorder= 1)
    axs[1].plot(x, 9/(5*x), c= c_meas[2], ls= ":")

    axs[0].set_xlim(2, 46)
    axs[1].set_xlim(2, 46)
    axs[0].legend(fontsize= 10, loc='lower left')
    axs[1].legend(fontsize= 10, loc='lower left')

    axs[0].grid()
    axs[1].grid()

plt.savefig("BDS_PauliMeas_Analysis", dpi= 300)
plt.show()