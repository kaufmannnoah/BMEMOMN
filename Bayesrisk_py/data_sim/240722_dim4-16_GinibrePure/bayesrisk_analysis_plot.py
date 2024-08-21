import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

title = ['ginibre', 'pure']
name = "out_dim4-16_GP_2.npy"
meas = ['rand', 'rand_bipartite', 'rand_separable', 'pauli']
dim = [4, 8, 16]
n_meas = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
n_sample = 4000

c_meas = ['firebrick', 'goldenrod', 'dodgerblue', 'olive']
m_s = 6 #markersize
l_w = 3 #linewidth
f_s = 12 #fontsize

fid_data = np.load(name)[0, :, :] #[ensemble][dim][meas][nmeas][sample]
ess_data = np.load(name)[3, :, [0, 2]]
wma_data = np.load(name)[2, :, [0, 2]]
dur_data = np.load(name)[1, :, [0, 2]]

#fid_data = fid_data[:, [0, 2]]

fig, axs = plt.subplots(nrows=len(title), ncols=len(dim), figsize=(12, 5), layout="constrained")

coef = np.zeros((len(title), len(dim), len(meas), 2))
coef_std = np.zeros((len(title), len(dim), len(meas), 2))
#fit exponential
for j in range(len(title)):
    for k in range(len(dim)):
        for i in range(len(meas)):
            #opt = sp.optimize.curve_fit(lambda t, b, c: np.log(b) - c * t, n_meas, np.log(np.average(1 - fid_data[k, j, i, :, :], axis=1)), bounds= ((0, 0), (10, 10)))
            opt = sp.optimize.curve_fit(lambda t, b, c: b * np.exp(-c * t), n_meas, np.average(1 - fid_data[j, k, i, :, :], axis=1), bounds= ((0, 0), (1, 1)))
            coef[j, k, i] = opt[0]
            coef_std[j, k, i] = np.sqrt(np.diag(opt[1]))

            
x_t = np.linspace(1, n_meas[-1], 10000)

for j in range(len(title)):
    for k in range(len(dim)):
        #for i in range(len(meas)):
            #axs[j][k].plot(x_t, coef[k, j, i, 0] * np.exp(-coef[k, j, i, 1] * x_t), ls='-', c=c_meas[i], lw= l_w, alpha= 0.5, label= "coef= " + '{:.2E}'.format(coef[k, j, i, 1]) + r' $\pm$' + '{:.1E}'.format(coef_std[k, j, i, 1]), zorder= 0)
        for i in range(len(meas)):
            fid_std = np.std(1 - fid_data[j, k, i, :, :], axis=1) / np.sqrt(n_sample)
            axs[j][k].errorbar(n_meas, np.average(1 - fid_data[j, k, i, :, :], axis=1), yerr= fid_std, c= c_meas[i], lw=l_w, ls= "", marker='o', ms= m_s, label= meas[i], alpha=1, zorder= 1)

        axs[j][k].set_xscale('log', base=10)
        axs[j][k].set_xlim(0.8, 1200)
        if j == 0: axs[j][k].set_ylim(-0.02, 0.32)
        else: axs[j][k].set_ylim(-0.05, 1.05)
        axs[j][k].set_xticks([1, 10, 100, 1000])
        if k == 0: axs[j][k].set_ylabel(r'infidelity ($1-\bar{F}$)', fontsize=f_s)
        if j == 1: axs[j][k].set_xlabel(r'number of measurements $N$', fontsize=f_s)
        axs[j][k].set_title(title[j] + " dim= " + str(dim[k]))
        axs[j][k].grid()

axs[0][0].legend(fontsize= 10, loc='lower left')

plt.savefig("240722_bayesrisk.png", dpi= 300)

plt.show()
