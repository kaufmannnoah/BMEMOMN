import numpy as np
from joblib import Parallel, delayed
import time

from functions_paulibasis import *
from functions_estimation import *

########################################################
#PARAMETERS

#SYSTEM
n_q = np.array([2]) # number of Qubits - fixed in this implementation
dim = 2**n_q # dimension of Hilbert space
p = [create_pauli_basis(n_qi) for n_qi in n_q] # create Pauli basis

#ENSEMBLE
L_b = ['BDS_dirichlet'] # type of ensemble
L = 10000 # number of sampling points
rho_in_E = True # Flag whether the state to estimate is part of ensemble

#AVERAGES FOR BAYES RISK ESTIMATION
n_sample = 1000

#MEASUREMENTS
M_b = ['bell'] # type of measurement
M = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40] # number of measurements

#METRIC
out_m = ['fidelity', 'HS', 'fid_MLE', 'HS_MLE', 'fid_recon', 'HS_recon'] # fixed!

#OUTCOME
out = np.zeros((len(out_m), len(L_b), len(dim), len(M_b), len(M), n_sample))

#PARAMETERS COMPUTATION
cores = -2 # number of cores to Parallelize (-k := evey core expect k-cores)
threshold = 1 / (L**2) # threshold below which weights are cut off
n_active0 = np.arange(L)

#RANDOM SEED
seed = 20240724
rng = np.random.default_rng(seed)

########################################################
#ESTIMATION

def func(dim, p, m_basis, n_m, r, w0, rho_0, rng= None):    
    #Estimation
    O, b = create_POVM(n_m, p, dim, rng, type= m_basis, ret_basis= True)
    x = experiment(O, rho_0, rng)
    w = bayes_update(r, w0, x, O, n_active0, threshold)

    #MLE
    rho_mle = MLE_BDS(x, O)
    rho_recon = recon_from_bell(x)

    #Output
    rho_est = pointestimate(r, w)
    fid = np.round(fidelity(rho_0, rho_est, p), decimals= 7)
    HS = np.round(HS_dist(rho_0, rho_est, p), decimals= 7)
    fid_mle = np.round(fidelity(rho_0, rho_mle, p), decimals= 7)
    HS_mle = np.round(HS_dist(rho_0, rho_mle, p), decimals= 7)
    fid_recon = np.round(fidelity(rho_0, rho_recon, p), decimals= 7)
    HS_recon = np.round(HS_dist(rho_0, rho_recon, p), decimals= 7)

    return fid, HS, fid_mle, HS_mle, fid_recon, HS_recon

########################################################
#MAIN

#Ensemble Types
for in_lb, lb_i in enumerate(L_b):

    #Dimensions
    for in_d, d_i in enumerate(dim):
        r, w0 = create_ensemble(L, p[in_d], d_i, rng, type= lb_i)
        if rho_in_E: rho_0 = [r[rng.integers(L)] for i in range(n_sample)]
        else: rho_0 = [create_ensemble(1, p[in_d], d_i, rng, type= lb_i)[0] for i in range(n_sample)]
        
        #Measurement Basis
        for in_mb, mb_i in enumerate(M_b):

            #Number of Measurements
            for in_m, m_i in enumerate(M):
                np.save(str(m_i), np.ones(1))
                #Spawn Pseudo Random Number Generators for Paralelization
                child_rngs = rng.spawn(n_sample)
                out[:, in_lb, in_d, in_mb, in_m, :] = np.array(Parallel(n_jobs=cores)(delayed(func)(d_i, p[in_d], mb_i, m_i, r, w0, rho_0[k], child_rngs[k]) for k in range(n_sample))).T
                #for k in range(n_sample):
                    #out[:, in_lb, in_d, in_mb, in_m, k] = np.array(func(d_i, p[in_d], mb_i, m_i, r, w0, rho_0[k], rng))

np.save("MLE_HS", out)