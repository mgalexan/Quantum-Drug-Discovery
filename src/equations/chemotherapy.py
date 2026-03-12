from equations.base_equation import Equation, QuantumTerm
from circuit_utils import make_parameterized_diagonal_circuit, find_diagonal_parameters
import numpy as np
import qiskit as qk
from scipy.optimize import least_squares


class Chemotherapy(Equation):
    """Tumor growth inhibition model"""
    def __init__(self, k_e = 0.4, k_e0 = 0.7, lambda_T = 0.05, K_T = 1.0, k_max = 0.3, EC50 = 0.3):
        """
        Initialize the Chemotherapy model with parameters.

        Args:
            k_e (float): Elimination rate constant of the drug.
            k_e0 (float): Effect-site equilibration rate.
            lambda_T (float): Tumor growth rate.
            K_T (float): Carrying capacity of the tumor.
            k_max (float): Maximum drug-induced kill rate.
            EC50 (float): Effect-site concentration for half-maximal effect.
        """
        super().__init__()
        self.k_e = k_e
        self.k_e0 = k_e0
        self.lambda_T = lambda_T
        self.K_T = K_T
        self.k_max = k_max
        self.EC50 = EC50
        self.name = "Chemotherapy"
        self.dim = 4
        self.ode_system = self.chemotherapy_ode
        self.gate_equiv = self.chemotherapy_gate_equiv
        self.var_names = ['C', 'E', 'T', None]
    
    def chemotherapy_ode(self, t, u):
        """
        Chemotherapy ODE system in a scipy format.
        """
        C, E, T, _ = u

        dCdt = -self.k_e * C
        dEdt = self.k_e0 * (C - E)
        dTdt = self.lambda_T * T * (1 - T / self.K_T) - (self.k_max * E / (E + self.EC50)) * T
        dDdt = 0.0  # Dummy variable for compatibility
        dudt = [dCdt, dEdt, dTdt, dDdt]
        return dudt

    def chemotherapy_gate_equiv(self, ansatz_cfg=None):
        """
        Get the quantum gate equivalent for the Chemotherapy system.
        """
        # Quantum term for the nonlinear parts
        diag_circuit_1, thetas_1 = make_parameterized_diagonal_circuit(n_qubits=2)

        # Add a Pauli-X in the 0th qubit
        diag_circuit_1.x(0)

        # Quantum term for the linear parts
        diag_circuit_2, thetas_2 = make_parameterized_diagonal_circuit(n_qubits=2)

        def find_parameters_term1(t, u):
            return find_diagonal_parameters(np.array([0.0, self.k_e0, 0.0, 0.0]), thetas_1)
        def find_parameters_term2(t, u):
            _, E, T, _ = u
            val1 = -self.k_max * E / (E + self.EC50)
            val2 = self.lambda_T * (1 - T / self.K_T)
            return find_diagonal_parameters(np.array([-self.k_e, -self.k_e0, val1 + val2, 0.0]), thetas_2)
        
        return [
            QuantumTerm(diag_circuit_1, find_parameters_term1),
            QuantumTerm(diag_circuit_2, find_parameters_term2)
            ]

class AltChemotherapy(Chemotherapy):
    """Alternate decomposition given by the Taylor Series"""
    def _fit_taylor_static(self):
        def cost(a):
            d = np.linspace(0, 0.5, 1000)
            cost = np.linalg.norm(a[0]*d + a[1]*d**2 + a[2]*d**3 - d / (d + self.EC50))
            
            return cost
        optimizer = least_squares(cost, [0, 0, 0])

        result = optimizer.x
        
        
        return result[0], result[1], result[2]
    
    def chemotherapy_gate_equiv(self, ansatz_cfg=None):

        a_1, a_2, a_3 = self._fit_taylor_static()
        # Quantum term for the nonlinear parts
        diag_circuit_1, thetas_1 = make_parameterized_diagonal_circuit(n_qubits=2)

        # Add a Pauli-X in the 0th qubit
        diag_circuit_1.x(0)

        # Quantum term for the linear parts
        diag_circuit_2, thetas_2 = make_parameterized_diagonal_circuit(n_qubits=2)

        # Set of three Taylor Terms
        diag_circuit_3, thetas_3 = make_parameterized_diagonal_circuit(n_qubits=2)
        diag_circuit_4, thetas_4 = make_parameterized_diagonal_circuit(n_qubits=2)
        diag_circuit_5, thetas_5 = make_parameterized_diagonal_circuit(n_qubits=2)


        def find_parameters_term1(t, u):
            return find_diagonal_parameters(np.array([0.0, self.k_e0, 0.0, 0.0]), thetas_1)
        def find_parameters_term2(t, u):
            _, E, T, _ = u
            val = self.lambda_T * (1 - T / self.K_T)
            return find_diagonal_parameters(np.array([-self.k_e, -self.k_e0, val, 0.0]), thetas_2)
        
        def find_parameters_term3(t, u):
            _, E, _, _ = u
            return find_diagonal_parameters(np.array([0.0, 0.0, a_1 * E, 0.0]), thetas_3)
        
        def find_parameters_term4(t, u):
            _, E, _, _ = u
            return find_diagonal_parameters(np.array([0.0, 0.0, a_2 * E**2, 0.0]), thetas_4)
        
        def find_parameters_term5(t, u):
            _, E, _, _ = u
            return find_diagonal_parameters(np.array([0.0, 0.0, a_3 * E**3, 0.0]), thetas_5)

        return [
            QuantumTerm(diag_circuit_1, find_parameters_term1),
            QuantumTerm(diag_circuit_2, find_parameters_term2),
            QuantumTerm(diag_circuit_3, find_parameters_term3),
            QuantumTerm(diag_circuit_4, find_parameters_term4),
            QuantumTerm(diag_circuit_5, find_parameters_term5),
            ]



        


# test the Chemotherapy equation
if __name__ == "__main__":
    chemo = AltChemotherapy()
    t = 0.0
    u = np.array([2.0, 5.0, 0.5, 0.0])
    terms = chemo.chemotherapy_gate_equiv()
    for i, term in enumerate(terms):
        params, coeff = term.find_parameters(t, u)
        circuit = term.circuit.assign_parameters(params, inplace=False)
        circuit.draw('mpl', style={'fontsize': 8}).savefig(f"chemotherapy_term_{i}.png")
        print(coeff)
        print(qk.quantum_info.Operator(circuit).data.round(2)[0:4, 0:4])


 
