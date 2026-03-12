import numpy as np
import qiskit as qk
import qiskit_aer as qka

import numpy as np
import qiskit as qk
from qiskit.circuit import ParameterVector

from omegaconf import DictConfig

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

# Import ansatz circuit for the next circuit
from ansatz import ULA

def make_ansatz_diagonal_circuit(n_qubits: int, ansatz_params: DictConfig = None):
    """
    Create a parameterized diagonal-loading circuit with an ansatz.
    
    Returns (full_circuit, placeholder_gate).
    """
    
    ansatz_name = list(ansatz_params.keys())[0]
    ansatz_cfg = ansatz_params.get(ansatz_name, {})
    ansatz_circuit, ansatz_params = globals()[ansatz_name](**ansatz_cfg)

    
    # Make the full circuit with 2*n qubits
    full_circuit = qk.QuantumCircuit(2 * n_qubits)
    
    # Append placeholder gate onto second set of qubits (qubits n to 2n-1)
    full_circuit.compose(ansatz_circuit, qubits=range(n_qubits, 2 * n_qubits), inplace=True)

    # Control the circuit on the first n_qubits
    for i in range(n_qubits):
        full_circuit.cx(i, i + n_qubits)
    
    return full_circuit, ansatz_params


# test the Diagonal Circuit
if __name__ == "__main__":
    state = np.array([0.0, 1.0, 0.0, 0.0])
    diag_circuit, thetas = make_parameterized_diagonal_circuit(n_qubits=2)
    param_map, coeff = find_diagonal_parameters(state, thetas)
    # Bind the parameters to the circuit
    diag_circuit.assign_parameters(param_map, inplace=True)
    diag_circuit.draw('mpl', style={'fontsize': 8}).savefig("diagonal_circuit.png")
    print(qk.quantum_info.Operator(diag_circuit).data.round(2)[0:4, 0:4])

# test the Ansatz Diagonal Circuit
if __name__ == "__main__":
    n_qubits = 2
    ansatz_cfg = {
        "ULA": {
            "n_qubits": n_qubits,
            "depth": 2
        }
    }
    diag_circuit, ansatz_params = make_ansatz_diagonal_circuit(n_qubits=n_qubits, ansatz_params=ansatz_cfg)
    lambdas = np.array([np.float64(1.267741308642421), np.float64(0.8105808813256852), np.float64(1.3646934053593776), np.float64(-1.9920335006412864), np.float64(0.7071939013669387), np.float64(-2.112373068381937), np.float64(0.37409877739319103), np.float64(1.0055697552364036), np.float64(0.5805341711698945), np.float64(0.11054015961818844), np.float64(1.485493858983698), np.float64(-0.10179399891359979)])
    param_map = {ansatz_params[i]: lambdas[i] for i in range(len(lambdas))}
    # Add X gates before and after the diagonal circuit to flip the first qubit
    full_circuit = qk.QuantumCircuit(4)
    full_circuit.x(0)
    full_circuit.x(1)
    full_circuit.compose(diag_circuit, inplace=True)
    full_circuit.x(1)
    full_circuit.x(0)

    full_circuit.assign_parameters(param_map, inplace=True)
    full_circuit.draw('mpl', style={'fontsize': 8}).savefig("ansatz_diagonal_circuit.png")
    print(qk.quantum_info.Operator(full_circuit).data.round(2)[0:4, 0:4])
    ansatz, ansatz_params = ULA(2, 2)
    param_map = {ansatz_params[i]: lambdas[i] for i in range(len(lambdas))}
    ansatz = ansatz.assign_parameters(param_map, inplace=False)
    print(qk.quantum_info.Operator(ansatz).data.round(2)[0:4, 0:4])