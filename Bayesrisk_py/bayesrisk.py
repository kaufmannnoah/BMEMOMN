import numpy as np
from joblib import Parallel, delayed
import time

from functions_paulibasis import *
from functions_estimation import *

########################################################
#PARAMETERS

#SYSTEM
n_q = np.array([2, 4]) # number of Qubits - fixed in this implementation
dim = 2**n_q # dimension of Hilbert space
p = [create_pauli_basis(n_qi) for n_qi in n_q] # create Pauli basis

#ENSEMBLE
L_b = ['ginibre', 'pure'] # type of ensemble
L = 10000 # number of sampling points
rho_in_E = True # Flag whether state to estimate is part of ensemble

#AVERAGES FOR BAYES RISK ESTIMATION
n_sample = 4000

#MEASUREMENTS
M_b = ['rand', 'rand_bipartite', 'rand_separable', 'pauli'] # type of measurement
M = [1, 2, 4, 8, 16, 32, 64, 256, 512, 1024]  # number of measurements

#METRIC
out_m = ['fidelity', 'runtime', 'w_max', 'ESS'] # fixed!

#OUTCOME
out = np.zeros((len(out_m), len(L_b), len(dim), len(M_b), len(M), n_sample))

#PARAMETERS COMPUTATION
cores = -2 # number of cores to Parallelize (-k = evey core expect k-cores)
threshold = 1 / (L**2) # threshold below which weights are cut off
n_active0 = np.arange(L)

#RANDOM SEED
seed = 240719
np.random.seed(seed)

########################################################
#ESTIMATION

def func(dim, p, m_basis, n_m, r, w0, rho_0, seed= None):    
    start = time.time()

    #Estimation
    O = create_POVM(n_m, p, dim, type= m_basis, seed= seed)
    x = experiment(O, rho_0)
    w = bayes_update(r, w0, x, O, n_active0, threshold)
    
    #Output
    duration = np.round(time.time() - start, decimals= 3)
    rho_est = pointestimate(r, w)
    fid = np.round(fidelity(rho_0, rho_est, p), decimals= 7)
    n_ess = np.round(1 / np.sum(w**2), decimals= 4)
    w_max = np.round(np.max(w), decimals= 4)

    return fid, duration, w_max, n_ess

########################################################
#MAIN

#Ensemble Types
for in_lb, lb_i in enumerate(L_b):

    #Dimensions
    for in_d, d_i in enumerate(dim):
        r, w0 = create_ensemble(L, p[in_d], d_i, type= lb_i, seed= seed)
        if rho_in_E: rho_0 = [r[np.random.randint(L)] for i in range(n_sample)]
        else: rho_0 = [create_ensemble(1, p[in_d], d_i, type= lb_i)[0] for i in range(n_sample)]
        
        #Measurement Basis
        for in_mb, mb_i in enumerate(M_b):

            #Number of Measurements
            for in_m, m_i in enumerate(M):
                out[:, in_lb, in_d, in_mb, in_m, :] = np.array(Parallel(n_jobs=cores)(delayed(func)(d_i, p[in_d], mb_i, m_i, r, w0, rho_0[k]) for k in range(n_sample))).T
  
np.save("out", out)