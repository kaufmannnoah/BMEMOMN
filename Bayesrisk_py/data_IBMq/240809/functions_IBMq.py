import numpy as np
import qiskit as qk
from qiskit.circuit import Parameter
from functions_paulibasis import *
  
def circuit_mixedstate(p):
    angles = get_angles(*p)
    G = qk.QuantumCircuit(4)
    G.ry(angles[0], 0)
    G.cx(0, 1)
    G.ry(angles[1], 0)
    G.ry(angles[2], 1)
    G.cx(0, 2)
    G.cx(1, 3)
    G.h(2)
    G.cx(2, 3)
    return G

def circuit_bellprep(b):
    G = qk.QuantumCircuit(2)
    G.h(0)
    G.cx(0, 1)
    if b // 2 == 1: G.x(0)
    if b % 2 == 1: G.z(0)
    return G
    
def circuit_bellmeasurement():
    G = qk.QuantumCircuit(2, 2)
    G.cx(0, 1)
    G.h(0)
    G.measure([0, 1], [0, 1])
    return G

def circuit_paulibell(a):
    G = qk.QuantumCircuit(2, 2)
    if a == 2: 
        G.s(0)
        G.s(1)
    if a > 0:
        G.h(0)
        G.h(1)
    G.measure([0, 1], [0, 1])
    return G

def vector_from_projector(P):
    for x in [np.array([1, 0]), np.array([0, 1])]:
        v = P @ x
        if not np.allclose(v, 0):
            return v / np.sqrt(v @ v)
    raise RuntimeError("Failed to find projection axis")

def get_angles(p00, p10, p01, p11):
    a_00, a_01, a_10, a_11 = np.sqrt([p00, p01, p10, p11])
    sin_α = np.clip(2 * (a_00 * a_11 - a_01 * a_10), -1, 1)
    cos_α = np.sqrt(1 - sin_α**2)
    α = np.arcsin(sin_α)
    if np.isclose(cos_α, 0, atol=1e-5):
        β = 2 * np.arctan2(np.sqrt(2) * a_10, np.sqrt(2) * a_00)
        γ = 0
    else:
        cα = np.cos(α / 2)
        sα = np.sin(α / 2)
        A = 1 / cos_α * np.array([
            [cα * a_00 - sα * a_11,  cα * a_01 + sα * a_10],
            [sα * a_01 + cα * a_10, -sα * a_00 + cα * a_11],
        ])
        b = vector_from_projector(A @ A.T)
        c = vector_from_projector(A.T @ A)
        # Fix sign of b
        if b[0] < 0 or (np.isclose(b[0], 0) and b[1] < 0):
            b = -b
        # Fix sign of c
        for _ in range(2):
            if np.allclose(np.outer(b, c), A):
                break
            else:
                c = -c
        else:
            raise RuntimeError("Could not fix sign for c")
        β = 2 * np.arctan2(b[1], b[0])
        γ = 2 * np.arctan2(c[1], c[0])
    return α, β, γ

def experiment_IBMq(b, rho_0, aer, meas= 'bell', prep= 'exact', seed= None):
    rng = np.random.default_rng(seed)
    x = np.ones(len(b))*4
    p = bvector_to_BDS(rho_0)
    
    if prep == 'exact': circuit_prep = circuit_mixedstate(p)
    elif prep == 'mixed':  bell_ind = rng.choice(np.array([0, 1, 2, 3]), p= p, size= (len(b), ))

    for i in range(4):
        if prep == 'mixed':
            ind_prep = np.where(bell_ind == i)[0]
            b_temp = b[ind_prep]
            ind_p_b = [ind_prep[np.where(b_temp == j)[0]] for j in range(3)]
        elif prep == 'exact':
            if i == 0: ind_p_b = [np.where(b == j)[0] for j in range(3)]
            else: ind_p_b = [[]]

        for ind_j, j in enumerate(ind_p_b):
            if len(j) != 0:
                C = qk.QuantumCircuit(4, 2)
                if prep == 'exact': C.append(circuit_prep, [0, 1, 2, 3])
                elif prep == 'mixed': C.append(circuit_bellprep(i), [2, 3])
                C.barrier()
                if meas == 'bell': C.append(circuit_bellmeasurement(), [2, 3], [0, 1])
                elif meas == 'pauli_BDS': C.append(circuit_paulibell(ind_j), [2, 3], [0, 1])
                result = aer.run(qk.transpile(C, aer), shots= len(j), seed= rng).result().get_counts(0).int_outcomes()
                temp = []
                for k in list(result):
                    temp += [k]*int(result.pop(k))
                temp = np.array(temp, dtype= int).flatten()
                rng.shuffle(temp)
                x[j] = temp

    return x.astype(int)