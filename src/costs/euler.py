from costs.base_cost import BaseCost
from equations.base_equation import Equation, QuantumTerm
import qiskit as qk
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
import numpy as np
from omegaconf import DictConfig

class EulerCost(BaseCost):
    """Cost function based on the forward Euler method for ODE systems."""
    
    def __init__(self, equation: Equation, cfg: DictConfig):
        """
        Initialize the Euler cost function with an equation and time step.

        Args:
            equation (Equation): The equation object containing the ODE system and gate equivalents.
            tau (float): The time step for the Euler method.
        """
        super().__init__(equation, cfg)
        self.tau = cfg.tau
        if cfg.backend_type == "aer_simulator":
            self.backend = AerSimulator()
        elif cfg.backend_type == "exact":
            self.backend = None


    def compile_with_ansatz(self, ansatz: qk.QuantumCircuit, ansatz_params) -> None:
        """
        Compile the cost function with a given ansatz circuit.

        Args:
            ansatz (qk.QuantumCircuit): The ansatz quantum circuit to be used in the cost function.
            ansatz_params: The parameters object of the ansatz circuit.
        """
        self.ansatz = ansatz
        self.ansatz_params = ansatz_params
        # We also need a copy of the ansatz for the previous timestep 
        self.ansatz_prev = ansatz.copy()
        self.ansatz_prev_params = ParameterVector('lambda_prev', len(ansatz_params))
        self.ansatz_prev = self.ansatz_prev.assign_parameters(self.ansatz_prev_params, inplace=False)

        self.quantum_terms = self.equation.gate_equiv()

        # For forwards Euler, we combine the Ansatz with each quantum term on the right
        self.combined_circuits = []
        self.hadamard_circuits = []

        # number of system qubits represented by the equation dim
        n_system_qubits = int(np.log2(self.equation.dim))

        for term in self.quantum_terms:
            term_nq = term.circuit.num_qubits
            # total qubits in the composed circuit: ensure we have space for system qubits and any ancillae
            total_qubits = max(n_system_qubits, term_nq)

            # Create an empty combined circuit with the required number of qubits
            combined = qk.QuantumCircuit(total_qubits)

            # Compose the ansatz onto the first n_system_qubits qubits
            ansatz_targets = list(range(self.ansatz.num_qubits))
            combined = combined.compose(self.ansatz, qubits=ansatz_targets, inplace=False)

            # Compose the term circuit so that its first n_system_qubits map to the system qubits (0..n_system_qubits-1)
            # and any ancilla qubits occupy subsequent indices.
            term_targets = list(range(term_nq))
            combined = combined.compose(term.circuit, qubits=term_targets, inplace=False)

            # Compose the adjoint ansatz onto the first n_system_qubits qubits
            combined = combined.compose(self.ansatz_prev.inverse(), qubits=ansatz_targets, inplace=False)
            self.combined_circuits.append(QuantumTerm(combined, term.find_parameters))

            hadamard_test = qk.QuantumCircuit(total_qubits + 1, 1)
            # ancilla at qubit 0
            hadamard_test.h(0)
            # Build a controlled version of the combined unitary (control = ancilla)
            controlled = combined.to_gate().control(1)
            # Append controlled gate with ancilla as control and system qubits shifted by +1
            hadamard_test.append(controlled, [0] + [i + 1 for i in range(total_qubits)])
            hadamard_test.h(0)
            hadamard_test.measure(0, 0)

            # Transpile before running
            hadamard_test = qk.transpile(hadamard_test, self.backend)

            self.hadamard_circuits.append(QuantumTerm(hadamard_test, term.find_parameters))
        
        # We also need the pure inner product of current and previous ansatz states
        self.inner_product_circuit_hadamard = qk.QuantumCircuit(n_system_qubits + 1, 1)
        self.inner_product_circuit_hadamard.h(0)
        
        self.inner_product_circuit = self.ansatz.compose(self.ansatz_prev.inverse(), qubits=ansatz_targets, inplace=False)
        self.inner_product_circuit_hadamard.compose(self.inner_product_circuit.to_gate().control(1), inplace=True, qubits=[0] + [i + 1 for i in range(n_system_qubits)])
        self.inner_product_circuit_hadamard.h(0)
        self.inner_product_circuit_hadamard.measure(0, 0)

        # Transpile before running
        self.inner_product_circuit_hadamard = qk.transpile(self.inner_product_circuit_hadamard, self.backend)

    def compute_cost(self, lambdas, lambdas_prev, t, u_prev) -> float:
        """
        Compute the cost value for given ansatz parameters.

        Args:
            lambdas: The specific values for the ansatz parameters.
            lambdas_prev: The previous ansatz parameters.
            t: The current time.
            u_prev: The previous state values.
        Returns:
            float: The computed cost value.
        """
        # Bind parameters to each combined circuit
        lambda_0 = lambdas[0]
        ansatz_param_dict = {self.ansatz_params[i]: lambdas[i+1] for i in range(len(lambdas)-1)}
        lambda_prev_0 = lambdas_prev[0]
        ansatz_prev_param_dict = {self.ansatz_prev_params[i]: lambdas_prev[i+1] for i in range(len(lambdas_prev)-1)} 
        cost = 0.0
        if self.backend is not None:
            for term in self.hadamard_circuits:
                # find_parameters is expected to return (param_map, coeff)
                param_map, coeff = term.find_parameters(t, u_prev)

                # bind ansatz parameters and term-specific parameters (leave any other parameters alone)
                bound_circuit = term.circuit.assign_parameters(param_map, inplace=False)
                bound_circuit = bound_circuit.assign_parameters(ansatz_param_dict, inplace=False)
                bound_circuit = bound_circuit.assign_parameters(ansatz_prev_param_dict, inplace=False)

                job = self.backend.run(bound_circuit, shots=8192)
                result = job.result()
                counts = result.get_counts()
                # Estimate the expectation value from measurement results
                n_shots = sum(counts.values())
                
                # Count the number of times the ancilla qubit is measured in the |0> state
                n_zero = 0
                for bitstring, count in counts.items():
                    if bitstring[-1] == '0':  # Ancilla is the first qubit measured
                        n_zero += count
                
                expectation = (n_zero - (n_shots - n_zero)) / n_shots
                cost += self.tau * coeff * expectation

            # Inner product circuit
            bound_inner = self.inner_product_circuit_hadamard.assign_parameters(ansatz_param_dict, inplace=False)
            bound_inner = bound_inner.assign_parameters(ansatz_prev_param_dict, inplace=False)
            job = self.backend.run(bound_inner, shots=8192)
            result = job.result()
            counts = result.get_counts()
            # Estimate the expectation value from measurement results
            n_shots = sum(counts.values())
            
            # Count the number of times the ancilla qubit is measured in the |0> state
            n_zero = 0
            for bitstring, count in counts.items():
                if bitstring[-1] == '0':  # Ancilla is the first qubit measured
                    n_zero += count
            
            expectation = (n_zero - (n_shots - n_zero)) / n_shots
            cost += expectation

        elif self.backend is None:
            for term in self.combined_circuits:
                # find_parameters is expected to return (param_map, coeff)
                param_map, coeff = term.find_parameters(t, u_prev)
                # bind ansatz parameters and term-specific parameters (leave any other parameters alone)
                bound_circuit = term.circuit.assign_parameters(param_map, inplace=False)
                bound_circuit = bound_circuit.assign_parameters(ansatz_param_dict, inplace=False)
                bound_circuit = bound_circuit.assign_parameters(ansatz_prev_param_dict, inplace=False)

                # Get the unitary matrix
                unitary = qk.quantum_info.Operator(bound_circuit).data

                # The expectation value is the real part of the top-left element
                expectation = np.real(unitary[0, 0])
                cost += self.tau * coeff * expectation

            # Inner product circuit
            bound_inner = self.inner_product_circuit.assign_parameters(ansatz_param_dict, inplace=False)
            bound_inner = bound_inner.assign_parameters(ansatz_prev_param_dict, inplace=False)
            unitary = qk.quantum_info.Operator(bound_inner).data
            expectation = np.real(unitary[0, 0])
            cost += expectation
        cost = lambda_0**2 - 2 * lambda_0 * lambda_prev_0 * cost
        return cost

# Test the EulerCost class with Lotka-Volterra equation
if __name__ == "__main__":
    from equations.lotka_volterra import LotkaVolterra
    from ansatz import ULA

    config = DictConfig({
        "tau": 0.1,
        "backend_type": "exact"
    })
    # Define the equation and cost
    lv_equation = LotkaVolterra()
    euler_cost = EulerCost(lv_equation, config)
    # Build an ansatz
    n_qubits = 1
    depth = 1
    ula_circuit, ula_params = ULA(n_qubits, depth)

    # Compile the cost with the ansatz
    euler_cost.compile_with_ansatz(ula_circuit, ula_params)

    # Plot term circuits
    lambdas = [5] + [np.random.rand() * np.pi / 2 for _ in range(len(ula_params))]
    lambdas_prev = [5] + [np.random.rand() * np.pi / 2 for _ in range(len(ula_params))]
    for i, term in enumerate(euler_cost.combined_circuits):
        circuit = term.circuit
        params, coeff = term.find_parameters(0.0, np.array([5.0, 2.0]))
        param_map = {ula_params[j]: lambdas[j + 1] for j in range(len(ula_params))}
        param_map_prev = {euler_cost.ansatz_prev_params[j]: lambdas_prev[j + 1] for j in range(len(ula_params))}
        circuit = circuit.assign_parameters(param_map, inplace=False)
        circuit = circuit.assign_parameters(params, inplace=False)
        circuit = circuit.assign_parameters(param_map_prev, inplace=False)
        circuit.draw('mpl', style={'fontsize': 8}).savefig(f"euler_cost_term_{i}.png")
        circuit_h = euler_cost.hadamard_circuits[i].circuit
        circuit_h = circuit_h.assign_parameters(param_map, inplace=False)
        circuit_h = circuit_h.assign_parameters(params, inplace=False)
        circuit_h = circuit_h.assign_parameters(param_map_prev, inplace=False)
        circuit_h.draw('mpl', style={'fontsize': 8}).savefig(f"euler_cost_hadamard_term_{i}.png")

    # Define test parameters and previous state
    
    t = 0.0
    # Compute the cost
    cost_value = euler_cost.compute_cost(lambdas, lambdas_prev, t, np.array([5.0, 2.0]))
    print(f"Computed Euler cost: {cost_value}")