import numpy as np
import matplotlib.pyplot as plt

title = ['BDS_dirichlet']
name = "MLE_HS.npy"

meas = ['Bell', 'XX_YY_ZZ (Pauli BDS)','MUB4', 'Pauli', 'Haar Random', 'Haar Random Bipartite']
c_meas = ['firebrick', 'goldenrod', 'dodgerblue', 'olive', 'red', 'blue']
lims = [[-0.01, 0.26], [-0.02, 0.52], [-0.03, 0.78]]
yticks = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25])

dim = [4]
n_meas = [np.arange(3, 91, 3, dtype= int), np.arange(3, 91, 3, dtype= int), np.arange(5, 91, 5, dtype= int), np.arange(9, 91, 9, dtype= int), np.arange(3, 91, 3, dtype= int), np.arange(3, 91, 3, dtype= int)]
n_sample = 10000

recon =['Bayesian estimation', 'maximum likelihood estimaton', 'direct inversion']
metric =['fidelity', 'HS']

markers = ['o', 'x', 'd']
m_s = 6 #markersize
l_w = 2 #linewidth
f_s = 12 #fontsize


fid = 1 - np.load(name)[[0, 2, 4], 0, 0] #[estimator][nmeas][meas][sample]
HS = np.load(name)[[1, 3, 5], 0, 0] #[estimator][nmeas][meas][sample]

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 5), layout="constrained")
pos = [[0], [1], [2]]

for indj, j in enumerate([0, 2, 1]):
    for i in range(len(meas)):
        temp = fid[j][i][:len(n_meas[i])]
        temp_std = np.std(1 - temp, axis=1) / np.sqrt(n_sample)
        if indj== 0: axs[*pos[indj]].errorbar(n_meas[i], np.average(temp, axis=1), yerr= temp_std, c= c_meas[i], lw=l_w, ls= "", marker= markers[j], ms= m_s, label= meas[i], alpha=1, zorder= 1)
        else: axs[*pos[indj]].errorbar(n_meas[i], np.average(temp, axis=1), yerr= temp_std, c= c_meas[i], lw=l_w, ls= "", marker= markers[j], ms= m_s, alpha=1, zorder= 1)

    x = np.linspace(3, 90, 1000)

    axs[*pos[indj]].set_title(recon[j])
    axs[*pos[indj]].set_xlim(2, 91)
    axs[*pos[indj]].set_ylim(*lims[j])
    axs[*pos[indj]].set_xticks(n_meas[0][1::2])
    axs[*pos[indj]].set_yticks(yticks * (j+1))
    axs[*pos[indj]].legend(fontsize= f_s, loc='upper right')
    axs[*pos[indj]].grid()
    axs[0].set_xlabel(r'number of measurements $N$', fontsize=f_s)
    axs[1].set_xlabel(r'number of measurements $N$', fontsize=f_s)
    axs[0].set_ylabel(r'average risk (1-F)', fontsize=f_s)

plt.savefig("BDS_meas_infid", dpi= 600)
plt.show()
