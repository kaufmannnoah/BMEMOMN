import numpy as np
from joblib import Parallel, delayed

from functions_paulibasis import *
from functions_estimation import *

########################################################
#PARAMETERS ESTIMATION
n_q = 1 # number of Qubits - fixed in this implementation
dim = 2**n_q # dimension of Hilbert space
p = create_pauli_basis(n_q) # create Pauli basis
#M = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] # number of measurements
M = [1]
L = 20000 # number of sampling points

#AVERAGES FOR BAYES RISK ESTIMATION
n_sample = 20000

#PARAMETERS COMPUTATION
threshold = 1 / (L**2) # thrshold below which weights are cut off
cores = -1
n_active0 = np.arange(L)

#ENSEMBLE
r, w0 = sample_ginibre_ensemble(L, p, dim, dim) #create mixed ensemble
#r, w0 = sample_pure_ensemble(L, p, dim) #create pure ensemble


########################################################
#ESTIMATION
def function(n_m, meas):
    start = time.time()
    rho = r[np.random.randint(0, L)]

    if meas == 'pauli': O = POVM_paulibasis(n_m, p, dim) # create M POVMs
    if meas == 'rand': O = POVM_randbasis(n_m, p, dim) # create M POVMs
    if meas == 'rand2': O = POVM_randbasis_2meas(n_m, p, dim) # create M POVMs
    if meas == 'randsep': O = POVM_randbasis_seperable(n_m, p, dim) # create M POVMs

    x = experiment(O, rho)
    w = bayes_update(r, w0, x, O, n_active0, threshold)
    rho_est = pointestimate(r, w)

    duration = np.round(time.time() - start, decimals= 3)
    fid = np.round(fidelity(rho, rho_est, p), decimals= 7)
    n_ess = np.round(1 / np.sum(w**2), decimals= 4)
    w_max = np.round(np.max(w), decimals= 4)

    return fid, n_ess, w_max, duration

out_ar = np.zeros((4, 4, len(M), n_sample)) # output array


for idm, n_m in enumerate(M):
    out = []
    # Run parallelized estimaiton of bayes risk
    out += [Parallel(n_jobs=cores)(delayed(function)(n_m, 'rand') for i in np.arange(n_sample))]
    out += [Parallel(n_jobs=cores)(delayed(function)(n_m, 'randsep') for i in np.arange(n_sample))]
    out += [Parallel(n_jobs=cores)(delayed(function)(n_m, 'pauli') for i in np.arange(n_sample))]
    out += [Parallel(n_jobs=cores)(delayed(function)(n_m, 'rand2') for i in np.arange(n_sample))]


    # Save output
    for ido, oi in enumerate(out):
        for i in range(len(oi)):
            for j in range(4):
                out_ar[j, ido, idm, i] = oi[i][j]

print(1 - np.average(out_ar[0, 0, 0]))
#np.save("output_bayesrisk", out_ar)