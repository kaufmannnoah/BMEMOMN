import numpy as np
import matplotlib.pyplot as plt

title = ['BDS_dirichlet']
name = "adapt_bell.npy"

meas = ['bell', 'pauli_BDS', 'pauli_BDS_adapt']
c_meas = ['firebrick', 'goldenrod', 'dodgerblue', 'olive', 'red', 'blue']
lims = [[-0.001, 0.026], [-0.001, 0.026], [-0.001, 0.026]]
yticks = np.array([0, 0.005, 0.01, 0.015, 0.02, 0.025])

dim = [4]
n_meas = [np.arange(30, 181, 30, dtype= int)]*3
n_sample =1000

recon =['Bayesian estimation', 'maximum likelihood estimaton', 'direct inversion']
metric =['fidelity', 'HS']

markers = ['o', 'x', 'd']
m_s = 6 #markersize
l_w = 2 #linewidth
f_s = 12 #fontsize


fid = np.load(name)[[0, 2, 4], 0, 0] #[estimator][nmeas][meas][sample]
HS = np.load(name)[[1, 3, 5], 0, 0] #[estimator][nmeas][meas][sample]

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 5), layout="constrained")
pos = [[0], [1], [2]]

for indj, j in enumerate([0, 2, 1]):
    for i in range(len(meas)):
        temp = HS[j][i][:len(n_meas[i])]
        HS_std = np.std(1 - temp, axis=1) / np.sqrt(n_sample)
        if indj== 0: axs[*pos[indj]].errorbar(n_meas[i]+ 2*i, np.average(temp, axis=1), yerr= HS_std, c= c_meas[i], lw=l_w, ls= "", marker= markers[j], ms= m_s, label= meas[i], alpha=1, zorder= 1)
        else: axs[*pos[indj]].errorbar(n_meas[i] + 2*i, np.average(temp, axis=1), yerr= HS_std, c= c_meas[i], lw=l_w, ls= "", marker= markers[j], ms= m_s, alpha=1, zorder= 1)

    x = np.linspace(30, 180, 1000)
    if j == 0: 
        axs[*pos[indj]].plot(x, 12/((x+4)**2), c= c_meas[0], ls= ":", label= r'Bell: $\frac{12}{(N+4)^2}$')

    axs[*pos[indj]].set_title(recon[j])
    axs[*pos[indj]].set_xlim(-8, 188)
    axs[*pos[indj]].set_ylim(*lims[j])
    axs[*pos[indj]].set_xticks([0, 60, 120, 180])
    axs[*pos[indj]].set_yticks(yticks)
    axs[*pos[indj]].legend(fontsize= f_s, loc='upper right')
    axs[*pos[indj]].grid()
    axs[j].set_xlabel(r'number of measurements $N$', fontsize=f_s)
axs[0].set_ylabel(r'average risk (HS)', fontsize=f_s)

plt.savefig("BDS_meas", dpi= 600)
plt.show()
