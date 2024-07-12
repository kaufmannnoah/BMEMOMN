import numpy as np

from functions_paulibasis import *
from functions_estimation import *

########################################################
#PARAMETERS
n_q = 1 # number of Qubits - fixed in this implementation
dim = 2**n_q # dimension of Hilbert space
p = create_pauli_basis(n_q) # create Pauli basis
M = 1 # number of measurements
L = 1000000 # number of sampling points
threshold = 1 / (L**2) # thrshold below which weights are cut off

################################################    ########
#ENSEMBLE AND POVMs
r, w0 = sample_ginibre_ensemble(L, p, dim, dim) #create ensemble
rho = r[0] # Ideal state
#O = POVM_paulibasis(M, p, dim) # create M POMVs
O = POVM_randbasis_seperable(M, p, dim)

########################################################
#ESTIMATION
x = experiment(O, rho)
n_active0 = np.arange(L)
w, dt = bayes_update(r, w0, x, O, n_active0, threshold)
rho_est = pointestimate(r, w)

########################################################
#RESULTS
fid = np.round(fidelity(rho, rho_est, p), decimals= 7)
n_ess = np.round(1 / np.sum(w**2), decimals= 4)
w_max = np.round(np.max(w), decimals= 4)

print(np.sqrt(np.sum(rho_est[1:]**2)))
print(fid)
print(n_ess)
print(w_max)
print(dt)