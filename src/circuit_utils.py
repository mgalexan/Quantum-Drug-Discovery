import numpy as np
import qiskit as qk
import qiskit_aer as qka

import numpy as np
import qiskit as qk
from qiskit.circuit import ParameterVector

def make_parameterized_diagonal_circuit(n_qubits: int):
    """
    Create a parameterized diagonal-loading circuit with one ancilla.
    Returns (circuit, parameters).
    """
    n_states = 2 ** n_qubits
    ancilla = n_qubits
    circuit = qk.QuantumCircuit(n_qubits + 1)

    # One angle parameter per computational basis state
    thetas = ParameterVector("theta", n_states)

    for i in range(n_states):
        binary_index = format(i, f"0{n_qubits}b")[::-1]  # Qiskit bit order

        for j, bit in enumerate(binary_index):
            if bit == "0":
                circuit.x(j)

        circuit.mcry(thetas[i], list(range(n_qubits)), ancilla, None)

        for j, bit in enumerate(binary_index):
            if bit == "0":
                circuit.x(j)

    return circuit, thetas

def find_diagonal_parameters(
    state: np.ndarray,
    thetas,
):
    """
    Bind parameters so the circuit prepares the given state amplitudes.
    """
    coeff = np.linalg.norm(state)
    if coeff == 0:
        # Handle zero vector case
        state = np.ones_like(state) / np.sqrt(len(state))
        coeff = 1.0
    else:
        state = state / coeff
    
    # Clamp state values to [-1, 1] to avoid arccos domain errors
    state = np.clip(state, -1.0, 1.0)
    angles = [2 * np.arccos(a) for a in state]
    param_map = dict(zip(thetas, angles))
    return param_map, coeff

# test the Diagonal Circuit
if __name__ == "__main__":
    state = np.array([0.0, 1.0, 0.0, 0.0])
    diag_circuit, thetas = make_parameterized_diagonal_circuit(n_qubits=2)
    param_map, coeff = find_diagonal_parameters(state, thetas)
    # Bind the parameters to the circuit
    diag_circuit.assign_parameters(param_map, inplace=True)
    diag_circuit.draw('mpl', style={'fontsize': 8}).savefig("diagonal_circuit.png")
    print(qk.quantum_info.Operator(diag_circuit).data.round(2)[0:4, 0:4])