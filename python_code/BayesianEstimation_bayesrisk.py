import numpy as np
from joblib import Parallel, delayed

from functions_paulibasis import *
from functions_estimation import *

########################################################
#PARAMETERS ESTIMATION
n_q = 2 # number of Qubits - fixed in this implementation
dim = 2**n_q # dimension of Hilbert space
p = create_pauli_basis(n_q) # create Pauli basis
M = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] # number of measurements
L = 10000 # number of sampling points

#AVERAGES FOR BAYES RISK ESTIMATION
R = L # number of different rho_0 in ensemble
rep_O = 1 # number of repetition per measurements

#PARAMETERS COMPUTATION
threshold = 1 / (L**3) # thrshold below which weights are cut off
cores = -1
n_active0 = np.arange(L)

########################################################
#ENSEMBLE
r, w0 = sample_ginibre_ensemble(L, p, dim, dim) #create ensemble
np.save('r', r)

########################################################
#ESTIMATION

def function_pauli(ind_r, n_m):
    start = time.time()
    rho = r[ind_r]
    O = POVM_paulibasis(n_m, p, dim) # create M POMVs
    x = experiment(O, rho)
    w, _ = bayes_update(r, w0, x, O, n_active0, threshold)
    rho_est = pointestimate(r, w)

    duration = np.round(time.time() - start, decimals= 3)
    fid = np.round(fidelity(rho, rho_est, p), decimals= 7)
    n_ess = np.round(1 / np.sum(w**2), decimals= 4)
    w_max = np.round(np.max(w), decimals= 4)

    return fid, n_ess, w_max, duration

def function_rand(ind_r, n_m):
    start = time.time()
    rho = r[ind_r]
    O = POVM_randbasis(n_m, p, dim) # create M POMVs
    x = experiment(O, rho)
    w, _ = bayes_update(r, w0, x, O, n_active0, threshold)
    rho_est = pointestimate(r, w)

    duration = np.round(time.time() - start, decimals= 4)
    fid = np.round(fidelity(rho, rho_est, p), decimals= 4)
    n_ess = np.round(1 / np.sum(w**2), decimals= 4)
    w_max = np.round(np.max(w), decimals= 4)

    return fid, n_ess, w_max, duration

def function_rand2(ind_r, n_m):
    start = time.time()
    rho = r[ind_r]
    O = POVM_randbasis_2meas(n_m, p, dim) # create M POMVs
    x = experiment(O, rho)
    w, _ = bayes_update(r, w0, x, O, n_active0, threshold)
    rho_est = pointestimate(r, w)

    duration = np.round(time.time() - start, decimals= 3)
    fid = np.round(fidelity(rho, rho_est, p), decimals= 7)
    n_ess = np.round(1 / np.sum(w**2), decimals= 4)
    w_max = np.round(np.max(w), decimals= 4)

    return fid, n_ess, w_max, duration

out_ar = np.zeros((4, 3, len(M), rep_O, L))

for idm, n_m in enumerate(M):
    for idr in range(rep_O):
        out = []
        out += [Parallel(n_jobs=cores)(delayed(function_pauli)(ind_r, n_m) for ind_r in np.arange(L))]
        out += [Parallel(n_jobs=cores)(delayed(function_rand)(ind_r, n_m) for ind_r in np.arange(L))]
        out += [Parallel(n_jobs=cores)(delayed(function_rand2)(ind_r, n_m) for ind_r in np.arange(L))]

        for ido, oi in enumerate(out):
            for i in range(len(oi)):
                for j in range(4):
                    out_ar[j, ido, idm, idr, i] = oi[i][j]

np.save("output", out_ar)