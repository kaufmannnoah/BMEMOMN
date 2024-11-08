import numpy as np
import matplotlib.pyplot as plt

title = ['BDS_dirichlet']
name = "MLE_HS.npy"

meas = 'XX_YY_ZZ (Pauli BDS)'
meas = 'Bell state measurements'
c_meas = ['purple', 'olive']
c_th = ['firebrick']*3


dim = [4]
n_meas = np.arange(3, 91, 3, dtype= int)
n_sample = 10000

recon =['bayesian', 'direct']
metric =['fidelity', 'HS']

markers = ['o', 'd']
m_s = 6 #markersize
l_w = 1 #linewidth
f_s = 12 #fontsize

fid = 1 - np.load(name)[[0, 4], 0, 0, 1] #[estimator][nmeas][meas][sample]
fid = 1 - np.load(name)[[0, 4], 0, 0, 0] #[estimator][nmeas][meas][sample]

def find_intersection(x, a):
    id0 = np.argwhere(np.diff(np.sign(a - x))).flatten()
    z = n_meas[id0] + (n_meas[id0+1] - n_meas[id0]) * (x - a[id0]) / (a[id0+1] - a[id0])
    return z[0]

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12, 5), layout="constrained")

th = [0.05, 0.03, 0.01]
x_int = np.zeros((len(th), len(recon)))

def find_intersection(x, a):
    id0 = np.argwhere(np.diff(np.sign(a - x))).flatten()
    z = n_meas[id0] + (n_meas[id0+1] - n_meas[id0]) * (x - a[id0]) / (a[id0+1] - a[id0])
    return z[0]

axs.hlines(0, xmin= 0, xmax= 93, colors= 'gray', lw = 2, alpha= 1)
for k in range(len(th)):
    for i in range(len(recon)):
        temp = np.average(fid[i][:len(n_meas)], axis= 1)
        x_int[k][i] = find_intersection(th[k], temp)
        axs.plot([x_int[k][i]]*2, [0, th[k]], c= c_meas[i], lw= 5, alpha= 0.3)
        axs.plot([x_int[k][i]]*2, [0, th[k]], c= c_meas[i], lw= 1, alpha= 1)
    axs.hlines(th[k], xmin= x_int[k][0], xmax= x_int[k][1], colors= c_th[k], lw = 5, alpha= 0.5)
    axs.hlines(th[k], xmin= 0, xmax= 93, colors= c_th[k], lw = 1, alpha= 1)
    axs.hlines(0, xmin= x_int[k][0], xmax= x_int[k][1], colors= c_th[k], lw = 5, alpha= 0.5)
    axs.hlines(0, xmin= x_int[k][0], xmax= x_int[k][1], colors= c_th[k], lw = 1, alpha= 1)
    axs.text((x_int[k][0]+x_int[k][1])/ 2, 0.005, str(np.round(x_int[k][1] - x_int[k][0], 2)), fontsize= f_s, ha= 'center')
    axs.text(1, th[k] + 0.005, str(th[k]), fontsize= f_s, c= c_th[k], ha= 'left')


for i in range(len(recon)):
    temp = fid[i][:len(n_meas)]
    temp_std = np.std(1 - temp, axis=1) / np.sqrt(n_sample)
    axs.errorbar(n_meas, np.average(temp, axis=1), yerr= temp_std, c= c_meas[i], lw=l_w, ls= "-", marker= markers[i], ms= m_s, label= recon[i], alpha=1, zorder= 1)

axs.set_title(meas)
axs.set_xlim(0, 93)
axs.set_ylim(-0.02, 0.27)
axs.set_xticks(n_meas[0::3])
axs.legend(fontsize= f_s, loc='upper right')
axs.grid()

axs.set_xlabel(r'number of measurements $N$', fontsize=f_s)
axs.set_ylabel(r'average risk (1-F)', fontsize=f_s)

#plt.savefig("BDS_XXYYZZ_th", dpi= 600)
plt.savefig("BDS_bell_th", dpi= 600)

plt.show()