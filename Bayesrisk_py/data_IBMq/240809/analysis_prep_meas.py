import numpy as np
import matplotlib.pyplot as plt

title = ['BDS_dirichlet']
name = "out.npy"

sim = ['brisbane', 'ideal']
prep = ['exact', 'mixture']
meas = ['bell', 'pauli_BDS']
labels = [["exact_bell", "exact_pauli"], ["mixture_bell", "mixture_pauli"]]
c_meas = [['firebrick', 'goldenrod'], ['dodgerblue', 'olive']]
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

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 5), layout="constrained")

x = np.linspace(3, 45, 1000)

for j in range(len(recon)):
    for k in range(len(sim)):
        for l in range(len(prep)):
            for i in range(len(meas)):
                HS_std = np.std(1 - HS[j, k, l, i], axis=1) / np.sqrt(n_sample)
                axs[k, j].errorbar(n_meas, np.average(HS[j, k, l, i], axis=1), yerr= HS_std, c= c_meas[l][i], lw=l_w, ls= "", marker= markers[j], ms= m_s, label= labels[l][i])

        axs[k, j].set_title(recon[j] + ", " + sim[k])
        axs[k, j].set_xlim(2, 46)
        axs[k, j].set_xticks(n_meas[1::2])
        axs[k, j].legend(fontsize= 10, loc='upper right')
        axs[k, j].grid(visible= True)
        if j == 0: 
            axs[k, j].set_ylim(-0.01, 0.21)
            axs[k, j].set_ylabel(r'average risk (HS)', fontsize=f_s)
        elif j == 1: 
            axs[k, j].set_ylim(-0.02, 0.42)
        else: 
            axs[k, j].set_ylim(-0.04, 0.84)
        if k == 1:
            axs[k, j].set_xlabel(r'number of measurements $N$', fontsize=f_s)



plt.savefig("BDS_prep_meas", dpi= 300)
plt.show()
