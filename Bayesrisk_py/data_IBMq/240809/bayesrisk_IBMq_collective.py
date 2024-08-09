import numpy as np
from joblib import Parallel, delayed
from qiskit_ibm_runtime import QiskitRuntimeService
import qiskit_aer as qa

from functions_IBMq import *
from functions_estimation import *
from functions_paulibasis import *

########################################################
#PARAMETERS

#SIMULATOR
service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token='b5e45b71a67e93d98a42d7f03e9ac1039c749e024d112b9312b252c27ad6464bc9eb708ccb21ed1f403f924dc836c0306d9b5c64ac8815e656a59744b13d68ea'
)
backend = service.backend("ibm_brisbane")
aer_brisbane = qa.AerSimulator.from_backend(backend)
aer_ideal = qa.AerSimulator()
sim = [aer_brisbane, aer_ideal]


#SYSTEM (FIXED)
n_q = 2 # number of Qubits
dim = 4  # dimension of Hilbert space
p = create_pauli_basis(n_q) # create Pauli basis

#RANDOM SEED
seed = 20240722
rng = np.random.default_rng(seed)

#AVERAGES FOR BAYES RISK ESTIMATION
n_sample = 1000

#ENSEMBLE
L = 10000
rho_in_E = True
r, w0 = create_ensemble(L, p, dim, rng, type= 'BDS')
if rho_in_E: rho_0 = [r[rng.integers(L)] for _ in range(n_sample)]
else: rho_0 = [create_ensemble(1, p, dim, rng, type= 'BDS')[0] for _ in range(n_sample)]

#PREPARATION
prep = ['exact', 'mixed']

#MEASUREMENTS
M_b = ['bell', 'pauli_BDS'] # type of measurement

M = np.arange(3, 48, 3, dtype= int) # number of measurements

#METRIC
out_m = ['HS', 'HS_MLE', 'HS_recon']
out = np.zeros((len(out_m), len(sim), len(prep), len(M_b), len(M), n_sample))

#PARAMETERS COMPUTATION
cores = -2 # number of cores to Parallelize (-k := evey core expect k-cores)
threshold = 1 / (L**2) # threshold below which weights are cut off
n_active0 = np.arange(L)

########################################################
#ESTIMATION

def func(dim, p, M, r, w0, rho_0, aer, m_basis, prep, rng= None):    
    #Experiment
    O, b = create_POVM(M[-1], p, dim, rng, type= m_basis, ret_basis= True)
    x = experiment_IBMq(b, rho_0, aer, m_basis, prep, seed= None)
    
    out = np.zeros((len(M), 3))
    for in_m, m_i in enumerate(M):
        #Estimation
        w = bayes_update(r, w0, x[:m_i], O[:m_i], n_active0, threshold)
        rho_est = pointestimate(r, w)
        rho_mle = MLE_BDS(x[:m_i], O[:m_i])
        if m_basis == 'pauli_BDS': rho_recon = recon_from_paulibell(x[:m_i], b[:m_i])
        else: rho_recon = recon_from_bell(x[:m_i])

        #Output
        HS = np.round(HS_dist(rho_0, rho_est, p), decimals= 7)
        HS_mle = np.round(HS_dist(rho_0, rho_mle, p), decimals= 7)
        HS_recon = np.round(HS_dist(rho_0, rho_recon, p), decimals= 7)
        out[in_m] = np.array([HS, HS_mle, HS_recon])
    return out

########################################################
#MAIN

for in_sim, sim_i in enumerate(sim):
    for in_p, p_i in enumerate(prep):
        np.save(str(in_sim) + "_" + p_i, np.ones(1))
        for in_mb, mb_i in enumerate(M_b):
            child_rngs = rng.spawn(n_sample)
            out[:, in_sim, in_p, in_mb, :, :] = np.array(Parallel(n_jobs=cores)(delayed(func)(dim, p, M, r, w0, rho_0[k], sim_i, mb_i, p_i, child_rngs[k]) for k in range(n_sample))).T
            #for k in range(n_sample):
                #out[:, in_sim, in_p, in_mb, :, k] = func(dim, p, M, r, w0, rho_0[k], sim_i, mb_i, p_i, child_rngs[k]).T

np.save("out", out)