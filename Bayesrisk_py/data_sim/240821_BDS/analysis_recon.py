import numpy as np
import matplotlib.pyplot as plt

title = ['BDS_dirichlet']
name = "MLE_HS.npy"

meas = ['Bell', 'XX_YY_ZZ (Pauli BDS)','MUB4', 'Pauli', 'Haar Random', 'Haar Random Bipartite']
c_meas = ['firebrick', 'goldenrod', 'dodgerblue', 'olive', 'red', 'blue']
lims = [[-0.02, 0.252], [-0.02, 0.652], [-0.02, 0.452], [-0.02, 0.452]]

dim = [4]
n_meas = [np.arange(3, 91, 3, dtype= int), np.arange(3, 91, 3, dtype= int), np.arange(5, 91, 5, dtype= int), np.arange(9, 91, 9, dtype= int), np.arange(3, 91, 3, dtype= int), np.arange(3, 91, 3, dtype= int)]
n_sample = 10000

recon =['bayesian', 'MLE', 'direct']
metric =['fidelity', 'HS']

markers = ['o', 'x', 'd']
m_s = 6 #markersize
l_w = 2 #linewidth
f_s = 12 #fontsize


fid = np.load(name)[[0, 2, 4], 0, 0] #[estimator][nmeas][meas][sample]
HS = np.load(name)[[1, 3, 5], 0, 0] #[estimator][nmeas][meas][sample]

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 5), layout="constrained")
pos = [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]]

for j in range(len(meas)):
    for i in range(len(recon)):
        temp = HS[i][j][:len(n_meas[j])]
        HS_std = np.std(1 - temp, axis=1) / np.sqrt(n_sample)
        if not(i > 1 and j > 3):
            axs[*pos[j]].errorbar(n_meas[j], np.average(temp, axis=1), yerr= HS_std, c= c_meas[j], lw=l_w, ls= "", marker= markers[i], ms= m_s, label= recon[i], alpha=1, zorder= 1)
    
    x = np.linspace(3, 90, 1000)
    if j == 0: 
        axs[*pos[j]].plot(x, 3/(5*x), c= c_meas[j], ls= ":")
        axs[*pos[j]].plot(x, 3/(5*(x+4)), c= c_meas[j], ls= ":")
    if j == 1: axs[*pos[j]].plot(x, 9/(5*x), c= c_meas[j], ls= ":")
    x = np.linspace(5, 90, 1000)
    if j == 2: axs[*pos[j]].plot(x, 3/x, c= c_meas[j], ls= ":")
    x = np.linspace(9, 90, 1000)
    if j == 3: axs[*pos[j]].plot(x, 27/(5*x), c= c_meas[j], ls= ":")

    axs[*pos[j]].set_title(meas[j])
    axs[*pos[j]].set_xlim(0, 93)
    axs[*pos[j]].set_ylim(*lims[1])
    axs[*pos[j]].set_xticks(n_meas[0][0::3])
    axs[*pos[j]].legend(fontsize= f_s, loc='upper right')
    axs[*pos[j]].grid()

axs[1, 0].set_xlabel(r'number of measurements $N$', fontsize=f_s)
axs[1, 1].set_xlabel(r'number of measurements $N$', fontsize=f_s)
axs[1, 2].set_xlabel(r'number of measurements $N$', fontsize=f_s)
axs[0, 0].set_ylabel(r'average risk (HS)', fontsize=f_s)
axs[1, 0].set_ylabel(r'average risk (HS)', fontsize=f_s)

plt.savefig("BDS_recon", dpi= 600)
plt.show()