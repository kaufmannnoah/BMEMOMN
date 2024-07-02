import numpy as np
import matplotlib.pyplot as plt

imp_data = np.load("output_bayesrisk_dimcomp_240626.npy")

# Load data 
fid_data = np.average(imp_data[:, 0, [1, 3, 0, 2], :, :, :], axis= 4)
fid_data = fid_data[[3, 2, 1, 0]]
# dim (1, 2, 3, 4) / type of measurement / number of measurements / ensemble


# Plot fidelities and duration
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(8, 10), layout="constrained")

c = ['firebrick', 'dodgerblue', 'goldenrod']
label = ['Random', 'Random_sep', 'Pauli']
fs = 12

for i in range(4):
    a = (1-fid_data[i, :-1, -1, :]).T
    mid = np.average(a)
    bins = np.linspace(mid - 0.02, mid + 0.02, 40)
    if i == 0: axs[i].set_ylim(-0.01, 0.81)
    else: axs[i].set_ylim(-0.01, 0.31)
    axs[i].hist(a, bins= bins, rwidth= 0.9, weights= np.ones(np.shape(a))/len(a), color= c, histtype='bar', label= label)
    axs[i].text(.02,.85, "d = " + str(2**(1+i)), horizontalalignment='left', transform=axs[i].transAxes, fontsize = fs)
    axs[i].set_ylabel('P.M.F.', fontsize= fs)

axs[0].legend()
axs[3].set_xlabel('Infidelity', fontsize= fs)

plt.show()