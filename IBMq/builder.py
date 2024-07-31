import numpy as np
import qiskit as qk
from qiskit.circuit import Parameter
from scipy.optimize import minimize

class BDS_Preparation:

    angle_names = ['a', 'b', 'c']

    def __init__(self, angles= None, probs= None):
        if probs is not None:
            self.probs = probs
            self.angles = self.get_angles(*probs)
        elif angles is not None:
            self.angles = angles
            self.probs = self.get_probs(*angles)
    
    def build(self):
        template = self.compact_circuit()
        params = [next(p for p in template.parameters if p.name == name)
                  for name in self.angle_names]
        return template.assign_parameters(
            {param: val for param, val in zip(params, self.angles)})
    
    @staticmethod
    def get_probs(alpha, beta, gamma):
        cα, sα = np.cos(alpha/2), np.sin(alpha/2)
        cβ, sβ = np.cos(beta/2), np.sin(beta/2)
        cγ, sγ = np.cos(gamma/2), np.sin(gamma/2)
        return np.array([
            cα * cβ * cγ + sα * sβ * sγ,
            cα * cβ * sγ - sα * sβ * cγ,
            cα * sβ * cγ - sα * cβ * sγ,
            cα * sβ * sγ + sα * cβ * cγ,
        ])**2

    @staticmethod
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
    
    @staticmethod
    def compact_circuit():
        G = qk.QuantumCircuit(4, name= "compact_g")
        G.ry(Parameter('a'), 0)
        G.cx(0, 1)
        G.ry(Parameter('b'), 0)
        G.ry(Parameter('c'), 1)
        G.cx(0, 2)
        G.cx(1, 3)
        G.h(2)
        G.cx(2, 3)
        return G
    
def circuit_bellmeasurement():
    G = qk.QuantumCircuit(2, 2, name= "bellmeas")
    G.cx(0, 1)
    G.h(0)
    G.measure([0, 1], [0, 1])
    return G

def circuit_paulibell(a):
    G = qk.QuantumCircuit(2, 2, name= "paulibellmeas")
    if a == 1: 
        G.s(0)
        G.s(1)
    if a < 2:
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
