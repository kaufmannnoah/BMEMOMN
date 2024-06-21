from functions_estimation import *

n_q = 1 # number of Qubits - fixed in this implementation
dim = 2**n_q # dimension of Hilbert space
p = create_pauli_basis(n_q) # create Pauli basis
M = 1 # number of measurements
L = 1000 # number of sampling points

#AVERAGES FOR BAYES RISK ESTIMATION
rep_O = 100000 # number of repetition per measurements

#PARAMETERS COMPUTATION
threshold = 0 # thrshold below which weights are cut off
n_active0 = np.arange(L)

#ENSEMBLE
r, w0 = sample_ginibre_ensemble(L, p, dim, dim) #create ensemble
rho = sample_ginibre_ensemble(1, p, dim, dim)[0][0]

w_avg = np.zeros(w0.shape)

for i in range(rep_O):
    O = POVM_paulibasis(M, p, dim) # create M POMVs
    x = experiment(O, rho)
    w, _ = bayes_update(r, w0, x, O, n_active0, threshold)
    w_avg += w / rep_O

w_ana = np.zeros(w0.shape)
for i in range(L):
    w_ana[i] = 1 / (3 * L) * (3 + 4 * np.sum(rho[1:] * r[i][1:]))
print(np.sum(w_ana))

print(np.sum(abs(w0-w_avg)))

print(np.sum(abs(w_ana-w_avg)))

print(rho)