import numpy as np
import scipy as sp
import qiskit as qk
from equations.base_equation import Equation
from qiskit.circuit import ParameterVector

def ULA(n_qubits: int, depth: int) -> qk.QuantumCircuit:
    """
    Build a Parametric Unitary Layered Ansatz (ULA) quantum circuit.

    Args:
        n_qubits (int): Number of qubits in the circuit.
        depth (int): Number of layers in the ULA.

    Returns:
        qk.QuantumCircuit: The constructed ULA quantum circuit.
    """

    circuit = qk.QuantumCircuit(n_qubits)

    params = ParameterVector("lambda", 3 * depth * n_qubits)

    param_index = 0
    for layer in range(depth):
        for qubit in range(n_qubits):
            circuit.rx(params[param_index], qubit)
            param_index += 1
            circuit.ry(params[param_index], qubit)
            param_index += 1
            circuit.rz(params[param_index], qubit)
            param_index += 1

        # Add entangling CNOTs in a ring topology
        if n_qubits > 1:
            for qubit in range(n_qubits):
                target = (qubit + 1) % n_qubits
                circuit.cx(qubit, target)

    return circuit, params




# test plotting the circuit with Cytokine system
if __name__ == "__main__":
    from equations.cytokine import Cytokine
    
    n_qubits = 2
    depth = 1
    param_vals = np.random.rand(depth, n_qubits, 3) * 2 * np.pi

    ula_circuit, params = ULA(n_qubits, depth)
    param_map = {params[i]: param_vals.flatten()[i] for i in range(len(params))}
    ula_circuit_assigned = ula_circuit.assign_parameters(param_map, inplace=False)

    ula_circuit.draw('mpl', style={'fontsize': 8}).savefig("ULA.png")
    print(qk.quantum_info.Operator(ula_circuit_assigned).data.round(2))
    
    # Test with Cytokine system
    cytokine = Cytokine()
    print(f"\nCytokine system:")
    print(f"  Name: {cytokine.name}")
    print(f"  Dimension: {cytokine.dim}")
    print(f"  Variables: {cytokine.var_names}")
    
    quantum_terms = cytokine.gate_equiv()
    print(f"  Number of quantum terms: {len(quantum_terms)}")
    for i, term in enumerate(quantum_terms):
        print(f"    Term {i}: type={term.gate_type}, n_qubits={term.circuit.num_qubits}")