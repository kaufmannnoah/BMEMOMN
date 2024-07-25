import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

title = ['BDS']
name = "out_dim4_B.npy"
meas = ['bell', 'MUB4', 'pauli_BDS', 'pauli']
dim = [4]
n_meas = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
n_sample = 4000

c_meas = ['firebrick', 'goldenrod', 'dodgerblue', 'olive']
m_s = 8 #markersize
l_w = 3 #linewidth
f_s = 12 #fontsize

fid_data = np.load(name)[0, 0, 0] #[ensemble][meas][nmeas][sample]
ess_data = np.load(name)[3, 0, 0]
wma_data = np.load(name)[2, 0, 0]
dur_data = np.load(name)[1, 0, 0]
fid_mle_data = np.load(name)[4, 0, 0] #[meas][nmeas][sample]

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 8), layout="constrained")
            
for i in range(len(meas)):
    fid_std = np.std(1 - fid_data[i], axis=1) / np.sqrt(n_sample)
    fid_std = np.std(1 - fid_mle_data[i], axis=1) / np.sqrt(n_sample)
    axs.errorbar(n_meas, np.average(1 - fid_data[i], axis=1), yerr= fid_std, c= c_meas[i], lw=l_w, ls= "", marker='o', ms= m_s, label= meas[i], alpha=1, zorder= 1)
    axs.errorbar(n_meas, np.average(1 - fid_data[i], axis=1), yerr= fid_std, c= c_meas[i], lw=1, ls= ":", marker='o', ms= 0)
    #axs.errorbar(n_meas, np.average(1 - fid_mle_data[i], axis=1), yerr= fid_std, c= c_meas[i], lw=l_w, ls= "", marker='x', ms= m_s, alpha=1, zorder= 1)

axs.set_xscale('log', base=10)
axs.set_xlim(0.8, 3000)
axs.set_xticks([1, 10, 100, 1000])
axs.legend(fontsize= 10, loc='lower left')
axs.grid()

axs.set_xlabel(r'number of measurements $M$', fontsize=f_s)
axs.set_ylabel(r'infidelity ($1-F$)', fontsize=f_s)

plt.savefig("240724_BDS_test.png", dpi= 300)
plt.show()