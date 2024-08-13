import numpy as np
import matplotlib.pyplot as plt

title = ['BDS_dirichlet']
name = "MLE_HS.npy"

meas = ['bell', 'pauli_BDS','MUB4', 'pauli']
c_meas = ['firebrick', 'goldenrod', 'dodgerblue', 'olive']
lims = [[-0.01, 0.26], [-0.02, 0.52], [-0.03, 0.78]]
yticks = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25])

dim = [4]
n_meas = np.arange(3, 48, 3, dtype= int)
n_sample = 4000

recon =['bayesian', 'MLE', 'direct']
metric =['fidelity', 'HS']

markers = ['o', 'x', 'd']
m_s = 8 #markersize
l_w = 3 #linewidth
f_s = 12 #fontsize


fid = np.load(name)[[0, 2, 4], 0, 0] #[estimator][nmeas][meas][sample]
HS = np.load(name)[[1, 3, 5], 0, 0] #[estimator][nmeas][meas][sample]

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 5), layout="constrained")
pos = [[0], [1], [2]]

x = np.linspace(3, 45, 1000)

for j in range(len(recon)):
    for i in range(len(meas)):
        HS_std = np.std(1 - HS[j][i], axis=1) / np.sqrt(n_sample)
        axs[*pos[j]].errorbar(n_meas, np.average(HS[j][i], axis=1), yerr= HS_std, c= c_meas[i], lw=l_w, ls= "", marker= markers[j], ms= m_s, label= meas[i], alpha=1, zorder= 1)
        
    if j == 0: 
        axs[*pos[j]].plot(x, 3/(5*(x+4)), c= c_meas[0], ls= ":")
        k= 2
        axs[*pos[j]].plot(x, 3/5*(x/3 + k**2) / (x/3+2*k)**2, c= c_meas[1], ls= ":")

    if j == 2: 
        axs[*pos[j]].plot(x, 3/(5*x), c= c_meas[0], ls= ":")
        axs[*pos[j]].plot(x, 9/(5*x), c= c_meas[1], ls= ":")

    axs[*pos[j]].set_title(recon[j])
    axs[*pos[j]].set_xlim(2, 46)
    axs[*pos[j]].set_ylim(*lims[j])
    axs[*pos[j]].set_xticks(n_meas[1::2])
    axs[*pos[j]].set_yticks(yticks * (j+1))
    axs[*pos[j]].legend(fontsize= 10, loc='upper right')
    axs[*pos[j]].grid()

plt.savefig("BDS_comp_meas", dpi= 300)
plt.show()
