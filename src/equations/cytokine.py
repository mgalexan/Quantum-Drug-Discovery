from equations.base_equation import Equation, QuantumTerm
from circuit_utils import make_parameterized_diagonal_circuit, find_diagonal_parameters
import numpy as np
import qiskit as qk

class Cytokine(Equation):
    """Cytokine-Mediated Tumor-Drug Interaction Model"""

    def __init__(self, alpha_T=0.05, K_T=1.0, alpha_H=0.10, H_max=1.0, beta_T=0.1, gamma_T=1.0, gamma_H=0.3, IC50T=0.3, IC50H=0.8, k_a=1.0, k_e=0.3, k_in=1.0, k_prod=0.2, k_deg=0.5, eta_T=0.5, delta_CD=0.2, dose=0.0):
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
        self.alpha_T = alpha_T
        self.alpha_H = alpha_H
        self.K_T = K_T
        self.H_max = H_max
        self.beta_T = beta_T
        self.gamma_T = gamma_T
        self.gamma_H = gamma_H
        self.IC50T = IC50T
        self.IC50H = IC50H
        self.k_a = k_a
        self.k_e = k_e
        self.k_in = k_in
        self.k_prod = k_prod
        self.k_deg = k_deg
        self.eta_T = eta_T
        self.delta_CD = delta_CD
        self.dose = dose

        self.name = "Cytokine"
        self.dim = 4
        self.ode_system = self.cytokine_ode
        self.gate_equiv = self.cytokine_gate_equiv
        self.var_names = ['T', 'H', 'D', 'C']

    def _dose(self, t):
        if t > 0.5:
            return 0.0
        else:
            return self.dose
    def cytokine_ode(self, t, u):
        """
        Cytokine ODE system in a scipy format.
        """
        T, H, D, C = u

        dTdt = self.alpha_T * T * (1 - T / self.K_T) - self.beta_T * C * T - self.gamma_T * D / (D + self.IC50T) * T

        dHdt = self.alpha_H * H * (1 - H / self.H_max) - self.gamma_H * D / (D + self.IC50H) * H

        dDdt =  - (self.k_a + self.k_e) * D + self.k_in * self._dose(t)

        dCdt = - self.k_deg * C - self.delta_CD * C * D + self.eta_T * T / (self.K_T + T) + self.k_prod

        dudt = [dTdt, dHdt, dDdt, dCdt]

        return dudt
    
    def cytokine_gate_equiv(self):
        """
        Get the quantum gate equivalent for the Cytokine system.
        """
        diag_circuit_1, thetas_1 = make_parameterized_diagonal_circuit(n_qubits=2)

        diag_circuit_2, thetas_2 = make_parameterized_diagonal_circuit(n_qubits=2)

        diag_circuit_3, thetas_3 = make_parameterized_diagonal_circuit(n_qubits=2)

        diag_circuit_4, thetas_4 = make_parameterized_diagonal_circuit(n_qubits=2)

        # Linear term in u
        def find_parameters_term1(t, u):
            return find_diagonal_parameters(np.array([self.alpha_T, self.alpha_H, -(self.k_a + self.k_e), -self.k_deg]), thetas_1)
        
        # Quadratic terms in u
        def find_parameters_term2(t, u):
            T, H, D, C = u
            
            return find_diagonal_parameters(np.array([
                -self.alpha_T / self.K_T * T - self.beta_T * C, 
                -self.alpha_H / self.H_max * H, 
                0.0, 
                -self.delta_CD * D
                ]), thetas_2)
        
        # Special nonlinear terms in u
        def find_parameters_term3(t, u):
            T, H, D, C = u
            
            return find_diagonal_parameters(np.array([
                -self.gamma_T * D / (self.IC50T + D), 
                -self.gamma_H * D / (self.IC50H + D), 
                0.0, 
                0.0
                ]), thetas_3)
        
        # Constant terms at each step
        def find_parameters_term4(t, u):
            T, H, D, C = u
            if t > 0.5:
                dose = 0.0
            else:
                dose = self.dose
            return find_diagonal_parameters(np.array([
                0.0, 
                0.0, 
                dose * self.k_in, 
                self.k_prod + self.eta_T * T / (self.K_T + T)
                ]), thetas_4)
        
        return [
            QuantumTerm(diag_circuit_1, find_parameters_term1),
            QuantumTerm(diag_circuit_2, find_parameters_term2),
            QuantumTerm(diag_circuit_3, find_parameters_term3),
            QuantumTerm(diag_circuit_4, find_parameters_term4, "constant")
        ]