
import qiskit
from qiskit import QuantumCircuit, Aer, execute

def generate_quantum_key():
    backend = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)  # Hadamard gate
    circuit.measure(0, 0)
    
    job = execute(circuit, backend, shots=1)
    result = job.result()
    key = result.get_counts(circuit)
    return list(key.keys())[0]
