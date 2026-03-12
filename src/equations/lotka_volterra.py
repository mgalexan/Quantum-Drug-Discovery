from equations.base_equation import Equation, QuantumTerm
from circuit_utils import make_parameterized_diagonal_circuit, find_diagonal_parameters, make_ansatz_diagonal_circuit
import numpy as np
import qiskit as qk
from qiskit.circuit import ParameterVector


class LotkaVolterra(Equation):
    """Lotka-Volterra predator-prey model."""
    
    def __init__(self, alpha=1.0, beta=1.0, delta=1.0, gamma=1.0):
        """
        Initialize the Lotka-Volterra model with parameters.

        Args:
            alpha (float): Prey growth rate.
            beta (float): Predation rate.
            delta (float): Predator reproduction rate.
            gamma (float): Predator death rate.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.name = "Lotka-Volterra"
        self.dim = 2
        self.ode_system = self.lotka_volterra_ode
        self.gate_equiv = self.lotka_volterra_gate_equiv
        self.var_names = ['Prey', 'Predator']

    def lotka_volterra_ode(self, t, u):
        """
        Lotka-Volterra ODE system in a scipy format.
        """
        x, y = u
        dudt = [
            #self.alpha * x,
            #- self.gamma * y
            #- self.beta * x * y,
            #self.delta * x * y
            - self.beta * x * y + self.alpha * x,
            self.delta * x * y - self.gamma * y 
                ]
        return dudt
    
    def lotka_volterra_gate_equiv(self, ansatz_cfg=None):
        """
        Get the quantum gate equivalent for the Lotka-Volterra system.
        """
        # Quantum term for the nonlinear parts
        diag_circuit_1, thetas_1 = make_parameterized_diagonal_circuit(n_qubits=1)
        
        # Add a Pauli-X
        diag_circuit_1.x(0)

        # Quantum term for the linear parts
        diag_circuit_2, thetas_2 = make_parameterized_diagonal_circuit(n_qubits=1)

        return [
            QuantumTerm(diag_circuit_1, lambda t, u: find_diagonal_parameters(u * np.array([-self.beta, self.delta]), thetas_1)),
            QuantumTerm(diag_circuit_2, lambda t, u: find_diagonal_parameters(np.array([self.alpha, -self.gamma]), thetas_2))
            ]
    
class LotkaVolterraConstant(LotkaVolterra):
    """Lotka-Volterra with a Constant for testing"""
    def lotka_volterra_ode(self, t, u):
        """
        Lotka-Volterra ODE system in a scipy format.
        """
        x, y = u
        dudt = [self.alpha * x + self.alpha - self.beta * x * y,
                self.delta * x * y - self.gamma * y]
        return dudt
    
    def lotka_volterra_gate_equiv(self, ansatz_cfg=None):
        """
        Get the quantum gate equivalent for the Lotka-Volterra system.
        """
        # Quantum term for the nonlinear parts
        diag_circuit_1, thetas_1 = make_parameterized_diagonal_circuit(n_qubits=1)
        
        # Add a Pauli-X
        diag_circuit_1.x(0)

        # Quantum term for the linear parts
        diag_circuit_2, thetas_2 = make_parameterized_diagonal_circuit(n_qubits=1)

        diag_circuit_3, thetas_3 = make_parameterized_diagonal_circuit(n_qubits=1)

        return [
            QuantumTerm(diag_circuit_1, lambda t, u: find_diagonal_parameters(u * np.array([-self.beta, self.delta]), thetas_1)),
            QuantumTerm(diag_circuit_2, lambda t, u: find_diagonal_parameters(np.array([self.alpha, -self.gamma]), thetas_2)),
            QuantumTerm(diag_circuit_3, lambda t, u: find_diagonal_parameters(np.array([self.alpha / u[0], 0]), thetas_3), "linear")
            ]

class LotkaVolterraFullQuantum(LotkaVolterra):
    """Lotka-Volterra fully quantum version for testing"""
    def lotka_volterra_gate_equiv(self, ansatz_cfg=None):
        """
        Get the quantum gate equivalent for the Lotka-Volterra system.
        """
        # Quantum term for the nonlinear parts
        diag_circuit_1, thetas_1 = make_parameterized_diagonal_circuit(n_qubits=1)
        # Find and pre-bind parameters:
        param_map_1, coeff_1 = find_diagonal_parameters(np.array([-self.beta, self.delta]), thetas_1)
        diag_circuit_1.assign_parameters(param_map_1, inplace=True)

        # Now we will add the nonlienarity
        ansatz_diag, ansatz_params = make_ansatz_diagonal_circuit(n_qubits=1, ansatz_params=ansatz_cfg)

        # Now we have 2 ancillae, so we need a bigger circuit
        full_circuit = qk.QuantumCircuit(3)
        full_circuit.compose(diag_circuit_1, qubits=[0,1], inplace=True)
        full_circuit.compose(ansatz_diag, qubits=[0,2], inplace=True)
        ansatz_params = ParameterVector('theta', len(ansatz_params))
        full_circuit.assign_parameters(ansatz_params, inplace=True)

        
        # Add a Pauli-X
        full_circuit.x(0)

        # Quantum term for the linear parts
        diag_circuit_2, thetas_2 = make_parameterized_diagonal_circuit(n_qubits=1)
        # Find and pre-bind parameters:
        param_map_2, coeff_2 = find_diagonal_parameters(np.array([self.alpha, -self.gamma]), thetas_2)
        diag_circuit_2.assign_parameters(param_map_2, inplace=True)


        return [
            QuantumTerm(full_circuit, lambda t, u: ({}, coeff_1), "ansatz", 1),
            QuantumTerm(diag_circuit_2, lambda t, u: ({}, coeff_2)),
        ]

# test the Lotka-Volterra equation
if __name__ == "__main__":
    lv = LotkaVolterraFullQuantum()
    t = 0.0
    u = np.array([2.0, 5.0])
    ansatz_cfg = {
        "ULA": {
            "n_qubits": 1,
            "depth": 1
        }
    }
    terms = lv.lotka_volterra_gate_equiv(ansatz_cfg=ansatz_cfg)
    lambda_vals = [np.float64(3.3462001851456884e-07), np.float64(1.854590545248467), np.float64(3.6709375186758804e-07)]
    for i, term in enumerate(terms):
        param_map, coeff = term.find_parameters(t, u)
        print(term.circuit.parameters)

        circuit = term.circuit.assign_parameters(param_map, inplace=False)
        if term.circuit.num_qubits == 3:
            circuit.assign_parameters(lambda_vals, inplace=True)
        circuit.draw('mpl', style={'fontsize': 8}).savefig(f"lotka_volterra_term_{i}.png")
        print(coeff)
        print(qk.quantum_info.Operator(circuit).data.round(2)[0:2,0:2])