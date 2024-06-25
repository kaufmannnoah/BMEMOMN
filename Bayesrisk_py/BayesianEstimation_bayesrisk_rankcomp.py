import numpy as np
from joblib import Parallel, delayed

from functions_paulibasis import *
from functions_estimation import *

########################################################
#PARAMETERS ESTIMATION
n_q = np.array([5, 4, 3, 2, 1]) # number of Qubits - fixed in this implementation
dim = 2**n_q # dimension of Hilbert space
p = [create_pauli_basis(n_qi) for n_qi in n_q] # create Pauli basis
M = [1, 4, 16, 64, 256, 1024] # number of measurements
L = 1000 # number of sampling points

#AVERAGES FOR BAYES RISK ESTIMATION
R = L # number of different rho_0 in ensemble
rep_O = 1 # number of repetition per measurements

#PARAMETERS COMPUTATION
threshold = 1 / (L**2) # thrshold below which weights are cut off
cores = -1
n_active0 = np.arange(L)

########################################################
#ENSEMBLE
r = []
w0 =[]
for ind_d in range(len(dim)):
    r_temp, w0_temp = sample_ginibre_ensemble(L, p[ind_d], dim[ind_d], dim[ind_d]) #create ensemble
    r += [r_temp]
    w0 += [w0_temp]

########################################################
#ESTIMATION
def function(ind_r, n_m, ind_d, meas):
    start = time.time()
    rho = r[ind_d][ind_r]
    if meas == 'pauli': O = POVM_paulibasis(n_m, p[ind_d], dim[ind_d])
    if meas == 'rand': O = POVM_randbasis(n_m, p[ind_d], dim[ind_d])
    if meas == 'rand2': O = POVM_randbasis_2meas(n_m, p[ind_d], dim[ind_d])
    if meas == 'randsep': O = POVM_randbasis_seperable(n_m, p[ind_d], dim[ind_d])
    x = experiment(O, rho)
    w, _ = bayes_update(r[ind_d], w0[ind_d], x, O, n_active0, threshold)
    rho_est = pointestimate(r[ind_d], w)

    duration = np.round(time.time() - start, decimals= 3)
    fid = np.round(fidelity(rho, rho_est, p[ind_d]), decimals= 7)
    n_ess = np.round(1 / np.sum(w**2), decimals= 4)
    w_max = np.round(np.max(w), decimals= 4)

    return fid, n_ess, w_max, duration

out_ar = np.zeros((len(dim), 4, 4, len(M), rep_O, L)) # output array

for ind_d in range(len(dim)):
    np.save(str(ind_d), np.zeros(1)) # to track progress (no progress)
    for idm, n_m in enumerate(M):
        for idr in range(rep_O):
            out = []
            # Run parallelized estimaiton of bayes risk
            out += [Parallel(n_jobs=cores)(delayed(function)(ind_r, n_m, ind_d, 'pauli') for ind_r in np.arange(L))]
            out += [Parallel(n_jobs=cores)(delayed(function)(ind_r, n_m, ind_d, 'rand') for ind_r in np.arange(L))]
            out += [Parallel(n_jobs=cores)(delayed(function)(ind_r, n_m, ind_d, 'rand2') for ind_r in np.arange(L))]
            out += [Parallel(n_jobs=cores)(delayed(function)(ind_r, n_m, ind_d, 'randsep') for ind_r in np.arange(L))]

            # Save output
            for ido, oi in enumerate(out):
                for i in range(len(oi)):
                    for j in range(4):
                        out_ar[ind_d, j, ido, idm, idr, i] = oi[i][j]

np.save("output", out_ar)