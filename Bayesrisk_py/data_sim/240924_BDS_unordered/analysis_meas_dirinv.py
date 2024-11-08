import numpy as np
import matplotlib.pyplot as plt
from mpmath import *

title = ['BDS_dirichlet']
name = "unordered.npy"

meas = ['Bell', 'Parity ordered','Parity random']
c_meas = ['#56B4E9', '#009E73', '#CC79A7', '#D55E00', '#E69F00', '#0072B2']
lims = [0, 0.65]
yticks = np.array([0, 0.25, 0.5])

dim = [4]
n_meas = [np.arange(1, 31, 1, dtype= int), np.arange(3, 31, 3, dtype= int), np.arange(1, 31, 1, dtype= int)]
n_sample = 400

recon =['Bayesian estimation', 'maximum likelihood estimaton', 'direct inversion']
metric =['fidelity', 'HS']

markers = ['o', 's', 'd']
m_s = 6 #markersize
l_w = 4 #linewidth
f_s = 12 #fontsize


HS = np.load(name)[[0, 2, 4], 0, 0] #[estimator][nmeas][meas][sample]
HS = np.load(name)[[1, 3, 5], 0, 0] #[estimator][nmeas][meas][sample]
HS = HS[:, :, :, :n_sample]

th_sq = 2 / 5
def fun(N):
    return (2/3)**N * ((1 - th_sq)*(1/2 * N * hyp3f2(1, 1, 1 - N, 2, 2, -1/2) - 1) + 3/4)

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), layout="constrained")
plt.style.use('tableau-colorblind10')
pos = [[0]]

for indj, j in enumerate([2]):
    for i in [0, 2, 1]:
        temp = HS[j][i][:len(n_meas[i])]
        HS_std = np.std(1 - temp, axis=1) / np.sqrt(n_sample)
        axs.errorbar(n_meas[i], np.average(temp, axis=1), yerr= HS_std, c= c_meas[i], lw=0, ls= "", marker= markers[i], ms= m_s, alpha=1, label= meas[i], zorder= 1)
        axs.errorbar(n_meas[i], np.average(temp, axis=1), yerr= HS_std, c= 'black', lw= 0.75, ls= "",  ms= 0, alpha=1, zorder= 2, capsize= 2)
        axs.errorbar(n_meas[i], np.average(temp, axis=1), yerr= HS_std, c= c_meas[i], lw=0, ls= "", marker= markers[i], ms= m_s, alpha= 0.5, zorder= 3)


    x = np.linspace(0.95, 31, 1000)
    axs.plot(x, 3/(5*x), c= c_meas[0], ls= ":", zorder= 0)
    x = np.linspace(0.7, 31, 1000)
    y= [fun(N) for N in x]
    axs.plot(x, y, c= c_meas[2], ls= ":", zorder= 0)
    x = np.linspace(2.85, 31, 1000)
    axs.plot(x, 9/(5*x), c= c_meas[1], ls= ":", zorder= 0)
 
    axs.set_xlim(0, 32)
    axs.set_ylim(*lims)
    axs.set_xticks(np.arange(0, 31, 12, dtype= int))
    axs.set_yticks(yticks)
    axs.legend(fontsize= f_s, loc='upper right')
    axs.grid()
    axs.set_xlabel(r'number of measurements $N$', fontsize=f_s)
    axs.set_ylabel(r'average risk (HS)', fontsize=f_s)

plt.savefig("BDS_mean_dirinv.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
plt.show()
