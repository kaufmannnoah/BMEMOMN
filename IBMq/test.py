from qiskit_ibm_runtime import QiskitRuntimeService
from builder import *
import qiskit as qk
import qiskit_aer as qa
from functions_estimation import *

service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token='b5e45b71a67e93d98a42d7f03e9ac1039c749e024d112b9312b252c27ad6464bc9eb708ccb21ed1f403f924dc836c0306d9b5c64ac8815e656a59744b13d68ea'
)

def experiment_paulimeas(b, rho_0, rng= None):
    rng = np.random.default_rng(rng)
    x = np.zeros(len(b))
    backend = service.backend("ibm_brisbane")
    aer = qa.AerSimulator.from_backend(backend)
    aer = qa.AerSimulator()
    ind = [list(np.where(b == i)[0]) for i in range(3)]
    shots = [len(ind[j]) for j in range(3)]
    p = bvector_to_BDS(rho_0)
    circuit_prep = BDS_Preparation(probs=p).build()
    for i in range(3):
        C = qk.QuantumCircuit(4, 2, name= "cicuit")
        C.append(circuit_prep, [0, 1, 2, 3])
        C.barrier()
        C.append(circuit_paulibell(i), [2, 3], [0, 1])
        result = aer.run(qk.transpile(C, aer), shots= shots[i], seed= rng).result().get_counts(0).int_outcomes()
        temp = []
        for j in list(result):
            temp += [j]*int(result.pop(j))
        temp = np.array(temp, dtype= int).flatten()
        rng.shuffle(temp)
        x[ind[i]] = temp
    return x.astype(int)

#RANDOM SEED
seed = 20240722
rng = np.random.default_rng(seed)

b = rng.integers(3, size=40)
print(b)
r = np.array([1, 0, 0, 0])
rho_0 = BDS_to_bvector(r)

a = experiment_paulimeas(b, rho_0, rng= rng)
print(a)
print(a[np.where(b == 0)])
print(a[np.where(b == 1)])
print(a[np.where(b == 2)])