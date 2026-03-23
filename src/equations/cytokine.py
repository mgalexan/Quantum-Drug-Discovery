from equations.base_equation import Equation, QuantumTerm
from circuit_utils import make_parameterized_diagonal_circuit, find_diagonal_parameters, make_ansatz_diagonal_circuit
import numpy as np
import qiskit as qk
from qiskit.circuit import ParameterVector
from collections import defaultdict
from ansatz import ULA

class Cytokine(Equation):
    """Cytokine-Mediated Tumor-Drug Interaction Model"""

    def __init__(self, alpha_T=0.05, K_T=1.0, alpha_H=0.10, H_max=1.0, beta_T=0.1, gamma_T=1.0, gamma_H=0.3, IC50T=0.3, IC50H=0.8, k_a=1.0, k_e=0.3, k_in=1.0, k_prod=0.2, k_deg=0.5, eta_T=0.5, delta_CD=0.2, dose=1.0):
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

        dTdt = self.alpha_T * T * (1 - T / self.K_T) - self.gamma_T * D * T - self.beta_T * C * T

        dHdt = self.alpha_H * H * (1 - H / self.H_max) - self.gamma_H * D * H

        dDdt =  - (self.k_a + self.k_e) * D + self.k_in * self._dose(t)

        dCdt = - self.k_deg * C - self.delta_CD * C * D + self.eta_T * T / (self.K_T + T) + self.k_prod

        dudt = [dTdt, dHdt, dDdt, dCdt]

        return dudt
    
    def cytokine_gate_equiv(self, ansatz_cfg=None):
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

class CytokineFullQuantum(Cytokine):
    """Cytokine-Mediated Tumor-Drug Interaction Model with implicit quantum terms"""

    def cytokine_gate_equiv(self, ansatz_cfg=None):
        """
        Get the quantum gate equivalent for the Cytokine system.
        """
        # Linear term in u
        diag_circuit_1, thetas_1 = make_parameterized_diagonal_circuit(n_qubits=2)
        param_map_1, coeff1 = find_diagonal_parameters(np.array([self.alpha_T, self.alpha_H,-(self.k_a + self.k_e),-self.k_deg]), thetas_1)
        diag_circuit_1.assign_parameters(param_map_1, inplace=True)


        # Pure Quadratic term in u
        diag_circuit_2, thetas_2 = make_parameterized_diagonal_circuit(n_qubits=2)
        param_map_2, coeff2 = find_diagonal_parameters(np.array([-self.alpha_T / self.K_T, -self.alpha_H / self.H_max, 0, -self.delta_CD]), thetas_2)
        diag_circuit_2.assign_parameters(param_map_2, inplace=True)
        ansatz_diag, ansatz_params = make_ansatz_diagonal_circuit(n_qubits=2, ansatz_params=ansatz_cfg)

        ansatz_params = ParameterVector("theta1", len(ansatz_params))
        ansatz_diag = ansatz_diag.assign_parameters(ansatz_params)

        full_circuit_2 = qk.QuantumCircuit(5)
        full_circuit_2.compose(diag_circuit_2, qubits=[0,1,4], inplace=True)
        full_circuit_2.compose(ansatz_diag, qubits=[0,1,2,3], inplace=True)
        full_circuit_2.cx(1,0)

        # Constant terms for u
        diag_circuit_3, thetas_3 = make_parameterized_diagonal_circuit(n_qubits=2)

        def find_parameters_term3(t, u):
            if t > 0.5:
                dose = 0.0
            else:
                dose = self.dose
            return find_diagonal_parameters(np.array([
                0.0, 
                0.0, 
                dose * self.k_in, 
                self.k_prod
                ]), thetas_3)
        
        # We need to add in the Tumor-Cytokine interaction
        diag_circuit_4, thetas_4 = make_parameterized_diagonal_circuit(n_qubits=2)
        param_map_4, coeff4 = find_diagonal_parameters(np.array([-self.beta_T, 0, 0, 0]), thetas_4)
        diag_circuit_4.assign_parameters(param_map_4, inplace=True)
        full_circuit_3 = qk.QuantumCircuit(5)
        full_circuit_3.compose(diag_circuit_4, qubits=[0,1,4], inplace=True)
        full_circuit_3.compose(ansatz_diag, qubits=[0,1,2,3], inplace=True)
        full_circuit_3.x(0)
        full_circuit_3.x(1)

        # Now we handle the Hill functions with a fitted polynomial approximation
        vals = np.linspace(0.01, 1.0, 100)
        #D_vals = vals / (vals + self.IC50T)
        #H1 = np.polyfit(vals, D_vals, 2)

        #D_vals = vals / (vals + self.IC50H)
        #H2 = np.polyfit(vals, D_vals, 2)

        T_vals = vals / (vals + self.K_T)
        H3 = np.polyfit(vals, T_vals, 3)

        # Formulation of the first Hill function term with ansatz diagonals
        # Constant term for H1
        #H1_diag1, thetas_H1_1 = make_parameterized_diagonal_circuit(n_qubits=2)
        #H1_param_map_1, H1_coeff1 = find_diagonal_parameters(np.array([-self.gamma_T * H1[0], 0, 0, 0]), thetas_H1_1)
        #H1_diag1.assign_parameters(H1_param_map_1, inplace=True)

        # Linear term for H1 with ansatz
        H1_diag2, thetas_H1_2 = make_parameterized_diagonal_circuit(n_qubits=2)
        H1_param_map_2, H1_coeff2 = find_diagonal_parameters(np.array([-self.gamma_T, 0, 0, 0]), thetas_H1_2)
        H1_diag2.assign_parameters(H1_param_map_2, inplace=True)
        ansatz_diag_H1_2, ansatz_params_H1_2 = make_ansatz_diagonal_circuit(n_qubits=2, ansatz_params=ansatz_cfg)
        ansatz_params_H1_2 = ParameterVector("theta1", len(ansatz_params_H1_2))
        ansatz_diag_H1_2 = ansatz_diag_H1_2.assign_parameters(ansatz_params_H1_2)
        full_circuit_H1_1 = qk.QuantumCircuit(5)
        full_circuit_H1_1.compose(H1_diag2, qubits=[0,1,4], inplace=True)
        full_circuit_H1_1.x(1)
        full_circuit_H1_1.compose(ansatz_diag_H1_2, qubits=[0,1,2,3], inplace=True)
        full_circuit_H1_1.x(1)

        '''
        # Quadratic term for H1 with ansatz
        H1_diag3, thetas_H1_3 = make_parameterized_diagonal_circuit(n_qubits=2)
        H1_param_map_3, H1_coeff3 = find_diagonal_parameters(np.array([-self.gamma_T * H1[2], 0, 0, 0]), thetas_H1_3)
        H1_diag3.assign_parameters(H1_param_map_3, inplace=True)
        ansatz_diag_H1_3, ansatz_params_H1_3 = make_ansatz_diagonal_circuit(n_qubits=2, ansatz_params=ansatz_cfg)
        ansatz_params_H1_3 = ParameterVector("theta1", len(ansatz_params_H1_3))
        ansatz_diag_H1_3 = ansatz_diag_H1_3.assign_parameters(ansatz_params_H1_3)
        ansatz_diag_H1_4, ansatz_params_H1_4 = make_ansatz_diagonal_circuit(n_qubits=2, ansatz_params=ansatz_cfg)
        ansatz_params_H1_4 = ParameterVector("theta2", len(ansatz_params_H1_4))
        ansatz_diag_H1_4 = ansatz_diag_H1_4.assign_parameters(ansatz_params_H1_4)
        full_circuit_H1_2 = qk.QuantumCircuit(7)
        full_circuit_H1_2.compose(H1_diag3, qubits=[0,1,6], inplace=True)
        full_circuit_H1_2.x(1)
        full_circuit_H1_2.compose(ansatz_diag_H1_3, qubits=[0,1,2,3], inplace=True)
        full_circuit_H1_2.compose(ansatz_diag_H1_4, qubits=[0,1,4,5], inplace=True)
        full_circuit_H1_2.x(1)
        
        
        # Cubic term for H1 with ansatz
        H1_diag4, thetas_H1_4 = make_parameterized_diagonal_circuit(n_qubits=2)
        H1_param_map_4, H1_coeff4 = find_diagonal_parameters(np.array([-self.gamma_T * H1[3], 0, 0, 0]), thetas_H1_4)
        H1_diag4.assign_parameters(H1_param_map_4, inplace=True)
        ansatz_diag_H1_5, ansatz_params_H1_5 = make_ansatz_diagonal_circuit(n_qubits=2, ansatz_params=ansatz_cfg)
        ansatz_params_H1_5 = ParameterVector("theta1", len(ansatz_params_H1_5))
        ansatz_diag_H1_5 = ansatz_diag_H1_5.assign_parameters(ansatz_params_H1_5)
        ansatz_diag_H1_6, ansatz_params_H1_6 = make_ansatz_diagonal_circuit(n_qubits=2, ansatz_params=ansatz_cfg)
        ansatz_params_H1_6 = ParameterVector("theta2", len(ansatz_params_H1_6))
        ansatz_diag_H1_6 = ansatz_diag_H1_6.assign_parameters(ansatz_params_H1_6)
        ansatz_diag_H1_7, ansatz_params_H1_7 = make_ansatz_diagonal_circuit(n_qubits=2, ansatz_params=ansatz_cfg)
        ansatz_params_H1_7 = ParameterVector("theta3", len(ansatz_params_H1_7))
        ansatz_diag_H1_7 = ansatz_diag_H1_7.assign_parameters(ansatz_params_H1_7)
        full_circuit_H1_3 = qk.QuantumCircuit(9)
        full_circuit_H1_3.compose(H1_diag4, qubits=[0,1,8], inplace=True)
        full_circuit_H1_3.x(1)
        full_circuit_H1_3.compose(ansatz_diag_H1_5, qubits=[0,1,2,3], inplace=True)
        full_circuit_H1_3.compose(ansatz_diag_H1_6, qubits=[0,1,4,5], inplace=True)
        full_circuit_H1_3.compose(ansatz_diag_H1_7, qubits=[0,1,6,7], inplace=True)
        full_circuit_H1_3.x(1)
        '''
        # Now for the Hill term in T
        # Constant term for H2
        #H2_diag1, thetas_H2_1 = make_parameterized_diagonal_circuit(n_qubits=2)
        #H2_param_map_1, H2_coeff1 = find_diagonal_parameters(np.array([0, -self.gamma_H * H2[0], 0, 0]), thetas_H2_1)
        #H2_diag1.assign_parameters(H2_param_map_1, inplace=True)

        # Linear term for H2 with ansatz
        H2_diag2, thetas_H2_2 = make_parameterized_diagonal_circuit(n_qubits=2)
        H2_param_map_2, H2_coeff2 = find_diagonal_parameters(np.array([0, -self.gamma_H, 0, 0]), thetas_H2_2)
        H2_diag2.assign_parameters(H2_param_map_2, inplace=True)
        ansatz_diag_H2_2, ansatz_params_H2_2 = make_ansatz_diagonal_circuit(n_qubits=2, ansatz_params=ansatz_cfg)
        ansatz_params_H2_2 = ParameterVector("theta1", len(ansatz_params_H2_2))
        ansatz_diag_H2_2 = ansatz_diag_H2_2.assign_parameters(ansatz_params_H2_2)
        full_circuit_H2_1 = qk.QuantumCircuit(5)
        full_circuit_H2_1.compose(H2_diag2, qubits=[0,1,4], inplace=True)
        full_circuit_H2_1.x(0)
        full_circuit_H2_1.x(1)
        full_circuit_H2_1.compose(ansatz_diag_H2_2, qubits=[0,1,2,3], inplace=True)
        full_circuit_H2_1.x(0)
        full_circuit_H2_1.x(1)

        '''
        # Quadratic term for H2 with ansatz
        H2_diag3, thetas_H2_3 = make_parameterized_diagonal_circuit(n_qubits=2)
        H2_param_map_3, H2_coeff3 = find_diagonal_parameters(np.array([0, -self.gamma_H * H2[2], 0, 0]), thetas_H2_3)
        H2_diag3.assign_parameters(H2_param_map_3, inplace=True)
        ansatz_diag_H2_3, ansatz_params_H2_3 = make_ansatz_diagonal_circuit(n_qubits=2, ansatz_params=ansatz_cfg)
        ansatz_params_H2_3 = ParameterVector("theta1", len(ansatz_params_H2_3))
        ansatz_diag_H2_3 = ansatz_diag_H2_3.assign_parameters(ansatz_params_H2_3)
        ansatz_diag_H2_4, ansatz_params_H2_4 = make_ansatz_diagonal_circuit(n_qubits=2, ansatz_params=ansatz_cfg)
        ansatz_params_H2_4 = ParameterVector("theta2", len(ansatz_params_H2_4))
        ansatz_diag_H2_4 = ansatz_diag_H2_4.assign_parameters(ansatz_params_H2_4)
        full_circuit_H2_2 = qk.QuantumCircuit(7)
        full_circuit_H2_2.compose(H2_diag3, qubits=[0,1,6], inplace=True)
        full_circuit_H2_2.x(0)
        full_circuit_H2_2.x(1)
        full_circuit_H2_2.compose(ansatz_diag_H2_3, qubits=[0,1,2,3], inplace=True)
        full_circuit_H2_2.compose(ansatz_diag_H2_4, qubits=[0,1,4,5], inplace=True)
        full_circuit_H2_2.x(0)
        full_circuit_H2_2.x(1)
        
        
        # Qubic term for H2 with ansatz
        H2_diag4, thetas_H2_4 = make_parameterized_diagonal_circuit(n_qubits=2)
        H2_param_map_4, H2_coeff4 = find_diagonal_parameters(np.array([0, -self.gamma_H * H2[3],0, 0]), thetas_H2_4)
        H2_diag4.assign_parameters(H2_param_map_4, inplace=True)
        ansatz_diag_H2_5, ansatz_params_H2_5 = make_ansatz_diagonal_circuit(n_qubits=2, ansatz_params=ansatz_cfg)
        ansatz_params_H2_5 = ParameterVector("theta1", len(ansatz_params_H2_5))
        ansatz_diag_H2_5 = ansatz_diag_H2_5.assign_parameters(ansatz_params_H2_5)
        ansatz_diag_H2_6, ansatz_params_H2_6 = make_ansatz_diagonal_circuit(n_qubits=2, ansatz_params=ansatz_cfg)
        ansatz_params_H2_6 = ParameterVector("theta2", len(ansatz_params_H2_6))
        ansatz_diag_H2_6 = ansatz_diag_H2_6.assign_parameters(ansatz_params_H2_6)
        ansatz_diag_H2_7, ansatz_params_H2_7 = make_ansatz_diagonal_circuit(n_qubits=2, ansatz_params=ansatz_cfg)
        ansatz_params_H2_7 = ParameterVector("theta3", len(ansatz_params_H2_7))
        ansatz_diag_H2_7 = ansatz_diag_H2_7.assign_parameters(ansatz_params_H2_7)
        full_circuit_H2_3 = qk.QuantumCircuit(9)
        full_circuit_H2_3.compose(H2_diag4, qubits=[0,1,8], inplace=True)
        full_circuit_H2_3.x(0)
        full_circuit_H2_3.x(1)
        full_circuit_H2_3.compose(ansatz_diag_H2_5, qubits=[0,1,2,3], inplace=True)
        full_circuit_H2_3.compose(ansatz_diag_H2_6, qubits=[0,1,4,5], inplace=True)
        full_circuit_H2_3.compose(ansatz_diag_H2_7, qubits=[0,1,6,7], inplace=True)
        full_circuit_H2_3.x(0)
        full_circuit_H2_3.x(1)
        '''
        # Now for the Tumor related Hill function in C, H3
        # This one is a little different because it isn't multiplied by D, so we are one degree lower.
        # Constant term for H3
        H3_diag1, thetas_H3_1 = make_parameterized_diagonal_circuit(n_qubits=2)
        H3_param_map_1, H3_coeff1 = find_diagonal_parameters(np.array([0, 0, 0, self.eta_T * H3[0]]), thetas_H3_1)
        H3_diag1.assign_parameters(H3_param_map_1, inplace=True)

        # Linear term for H3 with ansatz
        H3_diag2, thetas_H3_2 = make_parameterized_diagonal_circuit(n_qubits=2)
        H3_param_map_2, H3_coeff2 = find_diagonal_parameters(np.array([0, 0, 0, self.eta_T * H3[1]]), thetas_H3_2)
        H3_diag2.assign_parameters(H3_param_map_2, inplace=True)
        H3_diag2.x(1)
        H3_diag2.x(0)

        # Quadratic term for H3 with ansatz
        H3_diag3, thetas_H3_3 = make_parameterized_diagonal_circuit(n_qubits=2)
        H3_param_map_3, H3_coeff3 = find_diagonal_parameters(np.array([0, 0, 0, self.eta_T * H3[2]]), thetas_H3_3)
        H3_diag3.assign_parameters(H3_param_map_3, inplace=True)
        ansatz_diag_H3_3, ansatz_params_H3_3 = make_ansatz_diagonal_circuit(n_qubits=2, ansatz_params=ansatz_cfg)
        ansatz_params_H3_3 = ParameterVector("theta1", len(ansatz_params_H3_3))
        ansatz_diag_H3_3 = ansatz_diag_H3_3.assign_parameters(ansatz_params_H3_3)
        full_circuit_H3_1 = qk.QuantumCircuit(5)
        full_circuit_H3_1.compose(H3_diag3, qubits=[0,1,4], inplace=True)
        full_circuit_H3_1.compose(ansatz_diag_H3_3, qubits=[0,1,2,3], inplace=True)
        full_circuit_H3_1.x(1)
        full_circuit_H3_1.x(0)

        # Cubic term for H3 with ansatz
        H3_diag4, thetas_H3_4 = make_parameterized_diagonal_circuit(n_qubits=2)
        H3_param_map_4, H3_coeff4 = find_diagonal_parameters(np.array([0, 0, 0, self.eta_T * H3[3]]), thetas_H3_4)
        H3_diag4.assign_parameters(H3_param_map_4, inplace=True)
        ansatz_diag_H3_4, ansatz_params_H3_4 = make_ansatz_diagonal_circuit(n_qubits=2, ansatz_params=ansatz_cfg)
        ansatz_params_H3_4 = ParameterVector("theta1", len(ansatz_params_H3_4))
        ansatz_diag_H3_4 = ansatz_diag_H3_4.assign_parameters(ansatz_params_H3_4)
        ansatz_diag_H3_5, ansatz_params_H3_5 = make_ansatz_diagonal_circuit(n_qubits=2, ansatz_params=ansatz_cfg)
        ansatz_params_H3_5 = ParameterVector("theta2", len(ansatz_params_H3_5))
        ansatz_diag_H3_5 = ansatz_diag_H3_5.assign_parameters(ansatz_params_H3_5)
        full_circuit_H3_2 = qk.QuantumCircuit(7)
        full_circuit_H3_2.compose(H3_diag4, qubits=[0,1,6], inplace=True)
        full_circuit_H3_2.compose(ansatz_diag_H3_4, qubits=[0,1,2,3], inplace=True)
        full_circuit_H3_2.compose(ansatz_diag_H3_5, qubits=[0,1,4,5], inplace=True)
        full_circuit_H3_2.x(1)
        full_circuit_H3_2.x(0)


        return [
            QuantumTerm(diag_circuit_1, lambda t, u: ({}, coeff1)),
            QuantumTerm(full_circuit_2, lambda t, u: ({}, coeff2), "ansatz", 1),
            QuantumTerm(diag_circuit_3, find_parameters_term3, "constant"),
            QuantumTerm(full_circuit_3, lambda t, u: ({}, coeff4), "ansatz", 1),

            #QuantumTerm(H1_diag1, lambda t, u: ({}, H1_coeff1)),
            QuantumTerm(full_circuit_H1_1, lambda t, u: ({}, H1_coeff2), "ansatz", 1),
            #QuantumTerm(full_circuit_H1_2, lambda t, u: ({}, H1_coeff3), "ansatz", 2),
            #QuantumTerm(full_circuit_H1_3, lambda t, u: ({}, H1_coeff4), "ansatz", 3),

            #QuantumTerm(H2_diag1, lambda t, u: ({}, H2_coeff1)),
            QuantumTerm(full_circuit_H2_1, lambda t, u: ({}, H2_coeff2), "ansatz", 1),
            #QuantumTerm(full_circuit_H2_2, lambda t, u: ({}, H2_coeff3), "ansatz", 2),
            #QuantumTerm(full_circuit_H2_3, lambda t, u: ({}, H2_coeff4), "ansatz", 3),

            QuantumTerm(H3_diag1, lambda t, u: ({}, H3_coeff1), "constant"),
            QuantumTerm(H3_diag2, lambda t, u: ({}, H3_coeff2)),
            QuantumTerm(full_circuit_H3_1, lambda t, u: ({}, H3_coeff3), "ansatz", 1),
            QuantumTerm(full_circuit_H3_2, lambda t, u: ({}, H3_coeff4), "ansatz", 2)
        ]
    

if __name__ == "__main__":
    ck = CytokineFullQuantum()
    t = 0.0
    u = np.array([0.1, 1.0, 1.0, 0.1])
    ansatz_cfg = {
        "ULA": {
            "n_qubits": 2,
            "depth": 2
        }
    }
    terms = ck.cytokine_gate_equiv(ansatz_cfg=ansatz_cfg)
    lambda_vals = [np.float64(1.267741308642421), np.float64(0.8105808813256852), np.float64(1.3646934053593776), np.float64(-1.9920335006412864), np.float64(0.7071939013669387), np.float64(-2.112373068381937), np.float64(0.37409877739319103), np.float64(1.0055697552364036), np.float64(0.5805341711698945), np.float64(0.11054015961818844), np.float64(1.485493858983698), np.float64(-0.10179399891359979)]
    for i, term in enumerate(terms):
        param_map, coeff = term.find_parameters(t, u)
        circuit = term.circuit.assign_parameters(param_map, inplace=False)
        if term.gate_type == "ansatz":
            by_name = defaultdict(list)
            for p in circuit.parameters:
                vec_name = str(p).split('[')[0]
                by_name[vec_name].append(p)
            
            param_dict = {}
            for vec_name in sorted(by_name.keys()):
                for param_idx, p in enumerate(sorted(by_name[vec_name], key=lambda x: int(str(x).split('[')[1].split(']')[0]))):
                    param_dict[p] = lambda_vals[param_idx]
            
            circuit = circuit.assign_parameters(param_dict, inplace=False)
        
        circuit.draw('mpl', style={'fontsize': 8}).savefig(f"cytokine_term_{i}.png")
        print(coeff)
        op = qk.quantum_info.Operator(circuit).data.round(4)[0:4,0:4]
        print(op)
        print(u @ op)
    ula, ansatz_params = ULA(2, 2)
    ula.assign_parameters(lambda_vals, inplace=True)
    print(qk.quantum_info.Operator(ula).data.round(4)[0:4,0:4])


        