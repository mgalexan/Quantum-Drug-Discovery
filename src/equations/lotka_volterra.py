from equations.base_equation import Equation, QuantumTerm
from circuit_utils import make_parameterized_diagonal_circuit, find_diagonal_parameters
import numpy as np
import qiskit as qk

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
        dudt = [self.alpha * x - self.beta * x * y,
                self.delta * x * y - self.gamma * y]
        return dudt
    
    def lotka_volterra_gate_equiv(self):
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
    
    def lotka_volterra_gate_equiv(self):
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
            #QuantumTerm(diag_circuit_3, lambda t, u: find_diagonal_parameters(np.array([self.alpha, 0]), thetas_3), "constant")

            ]


# test the Lotka-Volterra equation
if __name__ == "__main__":
    lv = LotkaVolterra()
    t = 0.0
    u = np.array([2.0, 5.0])
    terms = lv.lotka_volterra_gate_equiv()
    for i, term in enumerate(terms):
        param_map, coeff = term.find_parameters(t, u)
        circuit = term.circuit.assign_parameters(param_map, inplace=False)
        circuit.draw('mpl', style={'fontsize': 8}).savefig(f"lotka_volterra_term_{i}.png")
        print(coeff)
        print(qk.quantum_info.Operator(circuit).data.round(2))