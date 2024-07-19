import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

title = ['ginibre', 'pure']
name = ["240717_out_" + i + ".npy" for i in title]
meas = ['rand', 'rand_bipartite', 'rand_separable', 'pauli']
dim = [4, 16]
n_meas = np.array([1, 4, 16, 64, 256, 1024])
n_sample = 1000

c_meas = ['firebrick', 'goldenrod', 'dodgerblue', 'olive']
m_s = 8 #markersize
l_w = 3 #linewidth
f_s = 12 #fontsize

fid_data = np.array([np.load(i)[0, 0] for i in name]) #[ensemble][meas][nmeas][sample]
ess_data = np.array([np.load(i)[1, 0] for i in name])
wma_data = np.array([np.load(i)[2, 0] for i in name])
dur_data = np.array([np.load(i)[3, 0] for i in name])

fig, axs = plt.subplots(nrows=len(name), ncols=len(dim), figsize=(5 * len(dim), 5 * len(name)), layout="constrained")

coef = np.zeros((len(name), len(dim), len(meas), 2))
coef_std = np.zeros((len(name), len(dim), len(meas), 2))
#fit exponential
for j in range(len(name)):
    for k in range(len(dim)):
        for i in range(len(meas)):
            #opt = sp.optimize.curve_fit(lambda t, b, c: np.log(b) - c * t, n_meas, np.log(np.average(1 - fid_data[k, j, i, :, :], axis=1)), bounds= ((0, 0), (10, 10)))
            opt = sp.optimize.curve_fit(lambda t, b, c: b * np.exp(-c * t), n_meas, np.average(1 - fid_data[j, k, i, :, :], axis=1), bounds= ((0, 0), (1, 1)))
            coef[j, k, i] = opt[0]
            coef_std[j, k, i] = np.sqrt(np.diag(opt[1]))

            
x_t = np.linspace(1, n_meas[-1], 10000)

for j in range(len(name)):
    for k in range(len(dim)):
        #for i in range(len(meas)):
            #axs[j][k].plot(x_t, coef[k, j, i, 0] * np.exp(-coef[k, j, i, 1] * x_t), ls='-', c=c_meas[i], lw= l_w, alpha= 0.5, label= "coef= " + '{:.2E}'.format(coef[k, j, i, 1]) + r' $\pm$' + '{:.1E}'.format(coef_std[k, j, i, 1]), zorder= 0)
        for i in range(len(meas)):
            fid_std = np.std(1 - fid_data[k, j, i, :, :], axis=1) / np.sqrt(n_sample)
            axs[j][k].errorbar(n_meas, np.average(1 - fid_data[k, j, i, :, :], axis=1), yerr= fid_std, c= c_meas[i], lw=l_w, ls= "", marker='o', ms= m_s, label= meas[i], alpha=1, zorder= 1)

        axs[j][k].set_xscale('log', base=10)
        axs[j][k].set_xlim(0.8, 3000)
        axs[j][k].set_xticks([1, 10, 100, 1000])
        axs[j][k].set_ylabel(r'infidelity ($1-F$)', fontsize=f_s)
        #axs[j].set_yticks(np.linspace(0, 0.30, 7))
        #axs[j].set_ylim(-0.025, 0.325)
        axs[j][k].grid()
        axs[j][k].set_title(title[j] + " dim= " + str(dim[k]))
        axs[j][k].legend(fontsize= 10, loc='lower left')
axs[1][1].set_xlabel(r'number of measurements $M$', fontsize=f_s)
axs[1][0].set_xlabel(r'number of measurements $M$', fontsize=f_s)

plt.savefig("240717_bayesrisk.png", dpi= 300)

plt.show()
