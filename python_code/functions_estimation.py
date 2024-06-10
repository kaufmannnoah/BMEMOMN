import numpy as np
import qutip as qt
import itertools
import time
from functions_paulibasis import *

def sample_ginibre_ensemble(n, p, dim_n, dim_k=None):
    # draw n states from the ginibre distribution (unbiased)
    # OUT: x_0: array of states sampled from Ginibre ensemble as Pauli vectors, w_0: uniform weights
    x_0 = np.zeros((n, dim_n**2))
    w_0 = np.ones(n)/n
    for i in range(n):
        dm = qt.rand_dm_ginibre(N=dim_n, rank=dim_k)
        x_0[i] = dm_to_bvector(dm, p, dim_n) # calculate pauli representation
    return x_0, w_0

def POVM_randbasis(M, p, dim):
    # returns a complete set of orthogonal states, sampled according to the haar measure
    o = np.zeros((M, dim, dim**2))
    for m in range(M):
        u = qt.rand_unitary_haar(dim)
        o[m] = np.array([ket_to_bvector(u[i], p, dim) for i in range(dim)])
    return o

def POVM_randbasis_2meas(M, p, dim):
    #
    o = np.zeros((M, 2, dim**2))
    for m in range(M):
        u = qt.rand_unitary_haar(dim)
        proj = ket_to_bvector(u[0], p, dim)
        remainder = dm_to_bvector(qt.qeye(dim)-bvector_to_dm(proj, p), p, dim)
        o[m] = np.array([proj, remainder])
    return o

def POVM_paulibasis(M, p, dim):
    # returns a complete set of orthogonal states, sampled according from the Pauli basis
    n_q = int(np.log2(dim)) #number of qubits
    o = np.zeros((M, dim, dim**2))
    b = np.random.randint(1, 4, size= (M, n_q)) # random Paulibasis (1=x, 2=y, 3=z)
    ind = np.array([4**i for i in range(n_q)][::-1]) # indices for Paulivector
    signs = np.array([list(i) for i in itertools.product([-1, 1], repeat= n_q)]) # (1 + sign_0 * P_0) x (1 + sign_1 * P_1) x ...
    comb = np.array([list(i) for i in itertools.product([0, 1], repeat= n_q)]) # create all combinations in the expression above
    for m in range(M):
        for ids, s in enumerate(signs):
            for c in comb:
                o[m][ids][np.sum(ind*c*b[m])] = (-1)**(len(np.where(c*s==-1)[0])%2)
    return o / dim

def prob_projectivemeas(oi, rho):
    # outcome probabilities of projective measurements specified in o, when measuring rho
    dim = np.sqrt(len(rho))
    prob = dim * np.array([np.sum(oo * rho) for oo in oi])
    if abs(np.sum(prob)-1) > 0.1: print(np.sum(prob))
    return prob
                                          
def experiment(o, rho):
    # measure rho in basis specified in POVM elements o
    x = np.zeros(len(o))
    for ido, oi in enumerate(o):
        prob = prob_projectivemeas(oi, rho)
        x[ido] = np.random.choice(np.arange(len(oi)), p= prob)
    return x
                                          
def likelihood(r, xi, oi):
    # calculate likelihood of measurement outcomes x for states in an array r 
    lh = np.array([np.sum(oi[int(xi)] * ri) for ri in r]) # proportional to probability (* dim is missing)
    return lh

def bayes_update(r, w, x, o, n_active, threshold):
    # update weights according to likelihood and normalize    
    start = time.time()
    w_temp = w
    for i in range(len(x)):
        w_new = np.zeros(len(w_temp)) # needed such that weights below the threshold are 0
        w_new[n_active] = w_temp[n_active] * likelihood(r[n_active], x[i], o[i])
        w_new[n_active] = np.divide(w_new[n_active], np.sum(w_new[n_active]))
        w_temp = w_new
        n_active = n_active[np.where(w_new[n_active] > threshold)]
        end = time.time()
    return w_new, end-start
                                  
def pointestimate(x, w):
    # return point estimate of rho
    return np.average(x, axis=0, weights= w)

def fidelity(a, b, p):
    # compute fidelity from density matrices in Pauli representation
    return qt.metrics.fidelity(bvector_to_dm(a, p), bvector_to_dm(b, p))**2