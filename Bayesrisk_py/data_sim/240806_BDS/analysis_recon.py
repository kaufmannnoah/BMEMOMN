import numpy as np
import matplotlib.pyplot as plt

title = ['BDS_dirichlet']
name = "MLE_HS.npy"

meas = ['bell', 'pauli_BDS','MUB4', 'pauli', 'random', 'random_bipartite']
c_meas = ['firebrick', 'goldenrod', 'dodgerblue', 'olive', 'red', 'blue']
lims = [[-0.02, 0.252], [-0.02, 0.652], [-0.02, 0.452], [-0.02, 0.452]]

dim = [4]
n_meas = np.arange(3, 63, 3, dtype= int)
n_sample = 1000

recon =['bayesian', 'MLE', 'direct']
metric =['fidelity', 'HS']

markers = ['o', 'x', 'd']
m_s = 6 #markersize
l_w = 2 #linewidth
f_s = 10 #fontsize


fid = np.load(name)[[0, 2, 4], 0, 0] #[estimator][nmeas][meas][sample]
HS = np.load(name)[[1, 3, 5], 0, 0] #[estimator][nmeas][meas][sample]

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), layout="constrained")
pos = [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]]

x = np.linspace(3, 60, 1000)

for j in range(len(meas)):
    for i in range(len(recon)):
        HS_std = np.std(1 - HS[i][j], axis=1) / np.sqrt(n_sample)
        if not(i > 1 and j > 3):
            axs[*pos[j]].errorbar(n_meas, np.average(HS[i][j], axis=1), yerr= HS_std, c= c_meas[j], lw=l_w, ls= "", marker= markers[i], ms= m_s, label= recon[i], alpha=1, zorder= 1)
   
    if j == 1: axs[*pos[j]].plot(x, 9/(5*x), c= c_meas[j], ls= ":")
    if j == 0: 
        axs[*pos[j]].plot(x, 3/(5*x), c= c_meas[j], ls= ":")
        axs[*pos[j]].plot(x, 3/(5*(x+4)), c= c_meas[j], ls= ":")
        
    axs[*pos[j]].set_title(meas[j])
    axs[*pos[j]].set_xlim(2, 61)
    axs[*pos[j]].set_ylim(*lims[1])
    axs[*pos[j]].set_xticks(n_meas[0::3])
    axs[*pos[j]].legend(fontsize= 10, loc='upper right')
    axs[*pos[j]].grid()

axs[1, 0].set_xlabel(r'number of measurements $M$', fontsize=f_s)
axs[1, 1].set_xlabel(r'number of measurements $M$', fontsize=f_s)
axs[1, 2].set_xlabel(r'number of measurements $M$', fontsize=f_s)
axs[0, 0].set_ylabel(r'average risk (HS)', fontsize=f_s)
axs[1, 0].set_ylabel(r'average risk (HS)', fontsize=f_s)

plt.savefig("BDS_recon", dpi= 300)
plt.show()