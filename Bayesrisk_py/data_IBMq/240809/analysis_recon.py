import numpy as np
import matplotlib.pyplot as plt

title = ['BDS_dirichlet']
name = "out.npy"

sim = ['brisbane', 'ideal']
prep = ['exact', 'mixture']
meas = ['bell', 'pauli_BDS']
labels = [["exact_bell", "exact_pauli"], ["mixture_bell", "mixture_pauli"]]
c_meas = ['firebrick', 'goldenrod', 'dodgerblue', 'olive']
lims = [[-0.01, 0.26], [-0.02, 0.52], [-0.03, 0.78]]
yticks = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25])

dim = [4]
n_meas = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45] 
n_sample = 400

recon =['bayesian', 'MLE', 'direct']
metric =['HS']

markers = ['o', 'x', 'd']
m_s = 6 #markersize
l_w = 2 #linewidth
f_s = 12 #fontsize

HS = np.load(name) #[estimator][simulation][preparation][meas][nmeas][sample]

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 8), layout="constrained")

x = np.linspace(3, 45, 1000)

for j in range(len(recon)):
    for k in range(len(sim)):
        for i in range(len(meas)):
            HS_std = np.std(1 - HS[j, k, 1, i], axis=1) / np.sqrt(n_sample)
            axs[k, i].errorbar(n_meas, np.average(HS[j, k, 1, i], axis=1), yerr= HS_std, c= c_meas[j], lw=l_w, ls= "", marker= markers[j], ms= m_s, label= recon[j])

            axs[k, i].set_title(sim[k] + ", " + meas[i])
            axs[k, i].set_xlim(2, 46)
            axs[k, i].set_xticks(n_meas[1::2])
            axs[k, i].legend(fontsize= 10, loc='upper right')
            axs[k, i].grid(visible= True)
            if i == 0: 
                axs[k, i].set_ylim(-0.01, 0.26)
                axs[k, i].set_ylabel(r'average risk (HS)', fontsize=f_s)
            else: axs[k, i].set_ylim(-0.03, 0.78)
            if k == 1:
                axs[k, i].set_xlabel(r'number of measurements $M$', fontsize=f_s)

plt.savefig("BDS_recon", dpi= 300)
plt.show()
