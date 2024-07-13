import numpy as np
import qutip as qt

from functions_paulibasis import *

########################################################
#ENSEMBLES

def create_ensemble(n, p, dim_n= 2, dim_k= None, type='ginibre'):
    match type:
        case 'ginibre': return sample_ginibre_ensemble(n, p, dim_n, dim_k)
        case 'pure': return sample_pure_ensemble(n, p, dim_n)
        case 'bell': return sample_belldiag_ensemble(n, p)

def sample_ginibre_ensemble(n, p, dim_n, dim_k=None):
    # draw n states from the ginibre distribution (unbiased)
    # OUT: x_0: array of states sampled from Ginibre ensemble as Pauli vectors, w_0: uniform weights
    x_0 = np.zeros((n, dim_n**2))
    w_0 = np.ones(n)/n
    for i in range(n):
        dm = qt.rand_dm(dim_n, distribution= 'ginibre', rank= dim_k) # if dim_k == None (return full rank)
        x_0[i] = dm_to_bvector(dm.full(), p, dim_n) # calculate pauli representation
    return x_0, w_0
    
def sample_pure_ensemble(n, p, dim_n):
    # draw n pure states (unbiased)
    # OUT: x_0: array of states sampled from Pure states as Pauli vectors, w_0: uniform weights
    x_0 = np.zeros((n, dim_n**2))
    w_0 = np.ones(n)/n
    for i in range(n):
        dm = qt.rand_dm(dim_n, distribution= 'pure')
        x_0[i] = dm_to_bvector(dm.full(), p, dim_n) # calculate pauli representation
    return x_0, w_0

def sample_belldiag_ensemble(n, p):
    # draw n states that are diagonal in the bell basis(unbiased)
    # OUT: x_0: array of states sampled from diagonal states in the Bell basis as Pauli vectors, w_0: uniform weights
    x_0 = np.zeros((n, 4**2))
    w_0 = np.ones(n)/n
    basis = [qt.ket2dm(qt.bell_state(b)).full() for b in ['00', '10', '01', '11']]
    for i in range(n):
        dm = qt.rand_dm(4, distribution= 'ginibre')
        for b in basis:
            x_0[i] = x_0[i] + dm_to_bvector(qt.expect(qt.Qobj(b), dm) * b, p ,4)
    return x_0, w_0

########################################################
#MEASUREMENT BASIS

def create_POVM(M, p, dim, type='rand'):
    match type:
        case 'rand': return POVM_randbasis(M, p, dim)
        case 'rand_bipartite': return POVM_randbasis_bipartite(M, p, dim)
        case 'rand_separable': return POVM_randbasis_separable(M, p, dim)
        case 'rand_2outcome': return POVM_randbasis_2outcome(M, p, dim)
        case 'pauli': return POVM_paulibasis(M, p, dim)

def POVM_randbasis(M, p, dim):
    # returns a complete set of orthogonal states, sampled according to the haar measure
    o = np.zeros((M, dim, dim**2))
    for m in range(M):
        u = qt.rand_unitary(dim, distribution= 'haar').full()
        o[m] = np.array([ket_to_bvector(u.T[i], p, dim) for i in range(dim)])
    return o

def POVM_randbasis_bipartite(M, p, dim):
    # returns a complete set of orthogonal bipartite states sampled from the haar measure, the systems are split as equal as possible
    o = np.zeros((M, dim, dim**2))
    nq = int(np.log2(dim))
    if nq == 1: return POVM_randbasis(M, p, dim)
    else:
        partition = [np.floor(nq / 2), np.ceil(nq / 2)]
        for m in range(M):
            u_i = [qt.rand_unitary(2**int(i), distribution= 'haar') for i in partition]
            u = qt.tensor(u_i).full()
            o[m] = np.array([ket_to_bvector(u.T[i], p, dim) for i in range(dim)])
        return o

def POVM_randbasis_separable(M, p, dim):
    # returns a complete set of orthogonal separable states sampled from the haar measure for each qubit
    o = np.zeros((M, dim, dim**2))
    nq = int(np.log2(dim))
    for m in range(M):
        u_i = [qt.rand_unitary(2, distribution= 'haar') for i in range(nq)]
        u = qt.tensor(u_i).full()
        o[m] = np.array([ket_to_bvector(u.T[i], p, dim) for i in range(dim)])
    return o

def POVM_paulibasis(M, p, dim):
    # returns a complete set of orthogonal separable states sampled from the haar measure for each qubit
    o = np.zeros((M, dim, dim**2))
    nq = int(np.log2(dim))
    u_p = [qt.Qobj([[1, 0], [0, 1]]), 1/np.sqrt(2) * qt.Qobj(np.array([[1, 1], [1, -1]])), 1/np.sqrt(2) * qt.Qobj([[1, 1], [1.j, -1.j]]), ] #I, H, SH
    for m in range(M):
        u_i = [u_p[np.random.randint(3)] for i in range(nq)]
        u = qt.tensor(u_i).full()
        o[m] = np.array([ket_to_bvector(u.T[i], p, dim) for i in range(dim)])
    return o

def POVM_randbasis_2outcome(M, p, dim):
    # teduces a random measurement with dim outcomes to 2 outcomes by averaging over the first dim/2 measurements and the last dim/2 meas
    temp = POVM_randbasis(M, p, dim)
    o = 2/dim * np.array([np.sum(temp[:, :int(dim / 2)], axis= 1), np.sum(temp[:, int(dim / 2):], axis= 1)])
    return o

########################################################
#ESTIMATION

def prob_projectivemeas(oi, rho):
    # outcome probabilities of projective measurements specified in o, when measuring rho
    dim = np.sqrt(len(rho))
    prob = dim * np.array([np.sum(oo * rho) for oo in oi])
    prob[np.where(abs(prob) < 10**(-12))] = 0 # get rid of numerical instabilities causing smmall negative probabilities
    prob = prob / np.sum(prob) # renormalizing probabilities
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
    w_temp = w
    for i in range(len(x)):
        w_new = np.zeros(len(w_temp)) # needed such that weights below the threshold are 0
        w_new[n_active] = w_temp[n_active] * likelihood(r[n_active], x[i], o[i])
        w_new[n_active] = np.divide(w_new[n_active], np.sum(w_new[n_active]))
        w_temp = w_new
        n_active = n_active[np.where(w_new[n_active] > threshold)]
    return w_new
                                  
def pointestimate(x, w):
    # return point estimate of rho
    return np.average(x, axis=0, weights= w)

def fidelity(a, b, p):
    # compute fidelity from density matrices in Pauli representation
    return qt.metrics.fidelity(bvector_to_dm(a, p), bvector_to_dm(b, p))**2