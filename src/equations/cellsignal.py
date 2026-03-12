from equations.base_equation import Equation, QuantumTerm
from circuit_utils import make_parameterized_diagonal_circuit, find_diagonal_parameters, make_ansatz_diagonal_circuit
import numpy as np
large_width = 400
np.set_printoptions(linewidth=large_width)
import qiskit as qk
from scipy.optimize import least_squares
from collections import defaultdict
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import UnitaryGate

class CellSignal(Equation):
    """Oscillatory cell signaling model"""
    def __init__(self, k1=1.4, k2=0.9, k3=2.5, k4=1.5, k5=0.6, k6=0.8, k7=2.0, k8=1.3, k9=0.7, k10=1.0, k11=0.3, k12=3.1, k13=1.8, k14=1.5):
        """Initialize the CellSignal model with parameters."""
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.k5 = k5
        self.k6 = k6
        self.k7 = k7
        self.k8 = k8
        self.k9 = k9
        self.k10 = k10
        self.k11 = k11
        self.k12 = k12
        self.k13 = k13
        self.k14 = k14
        self.name = "CellSignal"
        self.dim = 8
        self.ode_system = self.cell_signal_ode
        self.gate_equiv = self.cell_signal_gate_equiv
        self.var_names = ["ACA", "PKA", "ERK2", "REGA", "icAMP", "ecAMP", "CAR1",  None]
    
    def cell_signal_ode(self, t, u):
        """Cell signaling ODE system in a scipy format."""
        ACA, PKA, ERK2, REGA, icAMP, ecAMP, CAR1, _ = u

        dACAdt = self.k1 * ERK2 - self.k2 * ACA
        dPKAdt = self.k3 * icAMP - self.k4 * PKA
        dERK2dt = self.k5 * CAR1 - self.k6 * ERK2 * PKA
        dREGAdt = self.k7 - self.k8 * REGA * ERK2
        dicAMPdt = self.k9 * ACA - self.k10 * icAMP * REGA
        decAMPdt = self.k11 * ACA - self.k12 * ecAMP
        dCAR1dt = self.k13 * ecAMP - self.k14 * PKA
        dDdt = 0.0  # Dummy variable for compatibility
        dudt = [dACAdt, dPKAdt, dERK2dt, dREGAdt, dicAMPdt, decAMPdt, dCAR1dt, dDdt]
        return dudt

    def cell_signal_gate_equiv(self, ansatz_cfg=None):
        """Get the quantum gate equivalent for the CellSignal system."""
        # Quantum term for the nonlinear parts

        diag_circuit_1, thetas_1 = make_parameterized_diagonal_circuit(n_qubits=3)
        param_map_1, coeff1 = find_diagonal_parameters(np.array([0, -self.k6, -self.k8, -self.k10, 0, 0, 0, 0]), thetas_1)
        diag_circuit_1.assign_parameters(param_map_1, inplace=True)
        ansatz_diag, ansatz_params_1 = make_ansatz_diagonal_circuit(n_qubits=3, ansatz_params=ansatz_cfg)
        ansatz_params_1 = ParameterVector("theta1", len(ansatz_params_1))
        ansatz_diag = ansatz_diag.assign_parameters(ansatz_params_1)
        full_circuit_1 = qk.QuantumCircuit(7)
        
        full_circuit_1.x(0)
        full_circuit_1.cx(0,1)
        full_circuit_1.ccx(0,1,2)
        full_circuit_1.compose(diag_circuit_1, qubits=[0,1,2,6], inplace=True)
        full_circuit_1.compose(ansatz_diag, qubits=[0,1,2,3,4,5], inplace=True)
        full_circuit_1.ccx(0,1,2)
        full_circuit_1.cx(0,1)
        full_circuit_1.x(0)

        diag_circuit_2, thetas_2 = make_parameterized_diagonal_circuit(n_qubits=3)
        param_map_2, coeff2 = find_diagonal_parameters(np.array([0, -self.k14, 0, 0, 0, 0, 0, 0]), thetas_2)
        diag_circuit_2.assign_parameters(param_map_2, inplace=True)
        full_circuit_2 = qk.QuantumCircuit(7)
        full_circuit_2.x(0)
        full_circuit_2.x(1)
        full_circuit_2.x(2)
        full_circuit_2.ccx(0,1,2)
        full_circuit_2.compose(diag_circuit_2, qubits=[0,1,2,6], inplace=True)
        full_circuit_2.compose(ansatz_diag, qubits=[0,1,2,3,4,5], inplace=True)
        full_circuit_2.ccx(0,1,2)
        full_circuit_2.x(2)
        full_circuit_2.x(1)
        full_circuit_2.x(0)

        # Quantum terms for the linear parts
        # We need 3 here because the linear term for ACA appears 3 times in the ODE system
        diag_circuit_3, thetas_3 = make_parameterized_diagonal_circuit(n_qubits=3)
        
        param_map_3, coeff3 = find_diagonal_parameters(np.array([self.k1, self.k3, self.k5, 0, self.k9, -self.k12, 0, 0]), thetas_3)
        diag_circuit_3.assign_parameters(param_map_3, inplace=True) 

        perm = [2, 4, 6, 3, 0, 5, 1, 7]
        size = 2 ** 3
        perm_mat = np.zeros((size, size), dtype=complex)
        for j in range(size):
            perm_mat[perm[j], j] = 1
        perm_gate = UnitaryGate(perm_mat, label="row_perm_1")
        perm_circ = qk.QuantumCircuit(3)
        perm_circ.append(perm_gate, [0, 1, 2])
        diag_circuit_3 = diag_circuit_3.compose(perm_circ, qubits=[0, 1, 2], inplace=False)

        diag_circuit_4, thetas_4 = make_parameterized_diagonal_circuit(n_qubits=3)
        param_map_4, coeff4 = find_diagonal_parameters(np.array([-self.k2, -self.k4, 0, 0, 0, 0, self.k13, 0]), thetas_4)
        diag_circuit_4.assign_parameters(param_map_4, inplace=True)
        perm = [0, 1, 2, 3, 4, 6, 5, 7]
        size = 2 ** 3
        perm_mat = np.zeros((size, size), dtype=complex)
        for j in range(size):
            perm_mat[perm[j], j] = 1
        perm_gate = UnitaryGate(perm_mat, label="row_perm_2")
        perm_circ = qk.QuantumCircuit(3)
        perm_circ.append(perm_gate, [0, 1, 2])
        diag_circuit_4 = diag_circuit_4.compose(perm_circ, qubits=[0, 1, 2], inplace=False)

        diag_circuit_5, thetas_5 = make_parameterized_diagonal_circuit(n_qubits=3)
        param_map_5, coeff5 = find_diagonal_parameters(np.array([0, 0, 0, 0, 0, self.k11, 0, 0]), thetas_5)
        diag_circuit_5.assign_parameters(param_map_5, inplace=True)
        perm = [5, 1, 2, 3, 4, 0, 6, 7]
        size = 2 ** 3
        perm_mat = np.zeros((size, size), dtype=complex)
        for j in range(size):
            perm_mat[perm[j], j] = 1
        perm_gate = UnitaryGate(perm_mat, label="row_perm_3")
        perm_circ = qk.QuantumCircuit(3)
        perm_circ.append(perm_gate, [0, 1, 2])
        diag_circuit_5 = diag_circuit_5.compose(perm_circ, qubits=[0, 1, 2], inplace=False)

        # Constant term for the linear part of REGA
        diag_circuit_6, thetas_6 = make_parameterized_diagonal_circuit(n_qubits=3)
        param_map_6, coeff6 = find_diagonal_parameters(np.array([0, 0, 0, self.k7, 0, 0, 0, 0]), thetas_6)
        diag_circuit_6.assign_parameters(param_map_6, inplace=True)


        
        return [QuantumTerm(circuit=full_circuit_1, find_parameters= lambda t, u: ({}, coeff1), gate_type="ansatz", u_deg=1),
                QuantumTerm(circuit=full_circuit_2, find_parameters= lambda t, u: ({}, coeff2), gate_type="ansatz", u_deg=1),
                QuantumTerm(circuit=diag_circuit_3, find_parameters= lambda t, u: ({}, coeff3), gate_type="linear", u_deg=0),
                QuantumTerm(circuit=diag_circuit_4, find_parameters= lambda t, u: ({}, coeff4), gate_type="linear", u_deg=0),
                QuantumTerm(circuit=diag_circuit_5, find_parameters= lambda t, u: ({}, coeff5), gate_type="linear", u_deg=0),
                QuantumTerm(circuit=diag_circuit_6, find_parameters= lambda t, u: ({}, coeff6), gate_type="constant", u_deg=0)
                ]



"""Test the CellSignal system and plot the circuit"""
if __name__ == "__main__":
    cell_signal = CellSignal()
    ansatz_cfg = {
        "ULA": {
            "n_qubits": 3,
            "depth": 3
        }
    }
    t= 0.5
    u = [0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.7, 0.0]
    terms = cell_signal.cell_signal_gate_equiv(ansatz_cfg=ansatz_cfg)
    lambda_vals = [np.float64(-0.09472189356095911), np.float64(-1.1304792701301078), np.float64(1.260911074950096), np.float64(0.386545107887517), np.float64(1.209889404843403), np.float64(0.8517559515029183), np.float64(0.3616661745713242), np.float64(0.269639665265803), np.float64(-1.5102772372462776), np.float64(-0.8671382173106668), np.float64(-0.7903402578600256), np.float64(-0.34135127051223657), np.float64(-0.3081980579119404), np.float64(0.7344366353272939), np.float64(-0.8762864221723178), np.float64(0.24694795517712304), np.float64(0.2113486672078288), np.float64(0.10808371091792326), np.float64(0.14129564542660827), np.float64(1.4633099481222798), np.float64(0.1415848503816715), np.float64(-1.0902064864013155), np.float64(2.2457753516791277), np.float64(-0.6189726308412415), np.float64(-0.25569530718292316), np.float64(0.4631991348333657), np.float64(0.7790601510105634)]
    from ansatz import ULA
    ula, ansatz_params = ULA(3, 3)
    ula.assign_parameters(lambda_vals, inplace=True)
    #print(qk.quantum_info.Operator(ula).data.round(4)[0:8,0:8])
    vec = qk.quantum_info.Operator(ula).data[:, 0]
    print(vec.round(4))
    
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
        
        circuit.draw('mpl', style={'fontsize': 8}).savefig(f"cellsignal_term_{i}.png")
        print(coeff)
        print(qk.quantum_info.Operator(circuit).data.round(4)[0:8,0:8])
        print(vec[0:8].round(4) @ qk.quantum_info.Operator(circuit).data.round(4)[0:8,0:8])
    