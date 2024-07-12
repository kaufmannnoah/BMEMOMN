import numpy as np
import qutip as qt

from functions_paulibasis import *
from functions_estimation import *

n_q = 2 # number of Qubits - fixed in this implementation
dim = 2**n_q # dimension of Hilbert space
p = create_pauli_basis(n_q) # create Pauli basis
#print(sample_belldiag_ensemble(5, p))

basis = [qt.ket2dm(qt.bell_state(b)).full() for b in ['00', '10', '01', '11']]

for b in basis:
    print(dm_to_bvector(b, p ,4))

