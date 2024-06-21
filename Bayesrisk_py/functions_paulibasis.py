import numpy as np
import qutip as qt

def create_pauli_basis(n):
    # return list with all n-qubit Paulis
    s = [qt.qeye(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
    p = s
    for i in range(n-1):
        p = [qt.tensor(pi, si) for pi in p for si in s]
    r = [np.array(pi.full()) for pi in p]
    return r

def dm_to_bvector(a, basis, dim):
    # convert density matrix to vector in pauli basis (if basis = pauli basis)
    return 1 / dim * np.real(np.array([np.trace(np.dot(np.array(a), bi)) for bi in basis]))

def ket_to_bvector(a, basis, dim):
    # convert ket to vector in pauli basis (if basis = pauli basis)
    aa = np.array(a).T
    return 1 / dim * np.real(np.array([np.dot(np.dot(np.conj(aa.T), bi), aa) for bi in basis]))

def bvector_to_dm(v, basis):
    # convert vector in pauli basis to density matrix (if basis = pauli basis)
    return qt.Qobj(np.sum(np.array([v[i] * basis[i] for i in range(len(v))]), axis= 0))