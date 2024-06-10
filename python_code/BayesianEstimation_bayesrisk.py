import numpy as np
from joblib import Parallel, delayed

from functions_paulibasis import *
from functions_estimation import *

########################################################
#PARAMETERS ESTIMATION
n_q = 2 # number of Qubits - fixed in this implementation
dim = 2**n_q # dimension of Hilbert space
p = create_pauli_basis(n_q) # create Pauli basis
M = 10000 # number of measurements
L = 100 # number of sampling points

#AVERAGES FOR BAYES RISK ESTIMATION
R = L # number of different rho_0 in ensemble
rep_O = 1 # number of repetition per measurements

#PARAMETERS COMPUTATION
threshold = 1 / (L**2) # thrshold below which weights are cut off
cores = -2
n_active0 = np.arange(L)

########################################################
#ENSEMBLE
r, w0 = sample_ginibre_ensemble(L, p, dim, dim) #create ensemble
np.save('r', r)

########################################################
#ESTIMATION

def function_pauli(ind_r):
    start = time.time()
    rho = r[ind_r]
    O = POVM_paulibasis(M, p, dim) # create M POMVs
    x = experiment(O, rho)
    w, _ = bayes_update(r, w0, x, O, n_active0, threshold)
    rho_est = pointestimate(r, w)

    duration = np.round(time.time() - start, decimals= 3)
    fid = np.round(fidelity(rho, rho_est, p), decimals= 7)
    n_ess = np.round(1 / np.sum(w**2), decimals= 4)
    w_max = np.round(np.max(w), decimals= 4)

    return fid, n_ess, w_max, duration

def function_rand(ind_r):
    start = time.time()
    rho = r[ind_r]
    O = POVM_randbasis(M, p, dim) # create M POMVs
    x = experiment(O, rho)
    w, _ = bayes_update(r, w0, x, O, n_active0, threshold)
    rho_est = pointestimate(r, w)

    duration = np.round(time.time() - start, decimals= 3)
    fid = np.round(fidelity(rho, rho_est, p), decimals= 7)
    n_ess = np.round(1 / np.sum(w**2), decimals= 4)
    w_max = np.round(np.max(w), decimals= 4)

    return fid, n_ess, w_max, duration

def function_rand2(ind_r):
    start = time.time()
    rho = r[ind_r]
    O = POVM_randbasis_2meas(M, p, dim) # create M POMVs
    x = experiment(O, rho)
    w, _ = bayes_update(r, w0, x, O, n_active0, threshold)
    rho_est = pointestimate(r, w)

    duration = np.round(time.time() - start, decimals= 3)
    fid = np.round(fidelity(rho, rho_est, p), decimals= 7)
    n_ess = np.round(1 / np.sum(w**2), decimals= 4)
    w_max = np.round(np.max(w), decimals= 4)

    return fid, n_ess, w_max, duration

t_0 = time.time()
out = Parallel(n_jobs=cores)(delayed(function_rand)(ind_r) for ind_r in np.tile(np.arange(3), rep_O))

name = ['fid_p', 'n_ess_p', 'w_max_p', 'duration_p']
for i in range(4):
    np.save(name[i], out[i])
