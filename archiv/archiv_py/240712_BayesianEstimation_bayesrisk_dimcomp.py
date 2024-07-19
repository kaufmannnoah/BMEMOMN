import numpy as np
from joblib import Parallel, delayed
import time

from functions_paulibasis import *
from functions_estimation import *

########################################################
#PARAMETERS ESTIMATION
n_q = np.array([1, 2, 3, 4, 5]) # number of Qubits - fixed in this implementation
dim = 2**n_q # dimension of Hilbert space
p = [create_pauli_basis(n_qi) for n_qi in n_q] # create Pauli basis
M = [1, 10, 100] # number of measurements
L = 10000 # number of sampling points

#AVERAGES FOR BAYES RISK ESTIMATION
n_sample = 1000

#PARAMETERS COMPUTATION
threshold = 1 / (L**2) # thrshold below which weights are cut off
cores = -1
n_active0 = np.arange(L)

########################################################
#ENSEMBLE
rho_in_E = False # Flag if rho_0 is sampled from Ensemble or random
r = []
w0 =[]
for ind_d in range(len(dim)):
    #r_temp, w0_temp = sample_ginibre_ensemble(L, p[ind_d], dim[ind_d], dim[ind_d]) #create ensemble
    r_temp, w0_temp = sample_pure_ensemble(L, p[ind_d], dim[ind_d]) #create ensemble
    r += [r_temp]
    w0 += [w0_temp]

########################################################
#ESTIMATION
def function(n_m, ind_d, meas):
    start = time.time()
    if rho_in_E: rho = r[ind_d][np.random.randint(0, L)]
    else: rho = sample_pure_ensemble(1, p[ind_d], dim[ind_d])[0][0]
    #else: rho = sample_finibre_ensemble(1, p[ind_d], dim[ind_d], dim[ind_d])[0]

    if meas == 'pauli': O = POVM_paulibasis(n_m, p[ind_d], dim[ind_d])
    if meas == 'rand': O = POVM_randbasis(n_m, p[ind_d], dim[ind_d])
    if meas == 'rand2': O = POVM_randbasis_2outcome(n_m, p[ind_d], dim[ind_d])
    if meas == 'randsep': O = POVM_randbasis_separable(n_m, p[ind_d], dim[ind_d])

    x = experiment(O, rho)
    w = bayes_update(r[ind_d], w0[ind_d], x, O, n_active0, threshold)
    rho_est = pointestimate(r[ind_d], w)

    duration = np.round(time.time() - start, decimals= 3)
    fid = np.round(fidelity(rho, rho_est, p[ind_d]), decimals= 7)
    n_ess = np.round(1 / np.sum(w**2), decimals= 4)
    w_max = np.round(np.max(w), decimals= 4)

    return fid, n_ess, w_max, duration

for ind_d in range(len(dim)):
    out_ar = np.zeros((4, 4, len(M), n_sample)) # output array
    for idm, n_m in enumerate(M):
        out = []
        # Run parallelized estimaiton of bayes risk
        out += [Parallel(n_jobs=cores)(delayed(function)(n_m, ind_d, 'rand') for i in np.arange(n_sample))]
        out += [Parallel(n_jobs=cores)(delayed(function)(n_m, ind_d, 'randsep') for i in np.arange(n_sample))]
        out += [Parallel(n_jobs=cores)(delayed(function)(n_m, ind_d, 'pauli') for i in np.arange(n_sample))]
        out += [Parallel(n_jobs=cores)(delayed(function)(n_m, ind_d, 'rand_2') for i in np.arange(n_sample))]

        # Save output
        for ido, oi in enumerate(out):
            for i in range(len(oi)):
                for j in range(4):
                    out_ar[j, ido, idm, i] = oi[i][j]

    np.save("output_dim_pure_nInE"+str(dim[ind_d]), out_ar)