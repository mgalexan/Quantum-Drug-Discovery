import numpy as np
import scipy as sp
import qiskit as qk

class Equation():
    """A class to store ODE systems and their quantum gate equivalents."""
    
    def __init__(self):
        """
        Initialize the equation with a name, ODE system, and gate equivalent.

        Args:
            ode_system (callable): Function representing the ODE system.
            gate_equiv (callable): Function for the list of QuantumTerms appearing in the cost function.
        """
        self.ode_system = None
        self.gate_equiv = None
        self.name = "Generic Equation"
        self.dim = 0
        self.var_names = []
    
    def plot_results(self, t, u, name=None, u_num=None):
        """
        Plot the results of the ODE system.

        Args:
            t (array-like): Time points.
            u (array-like): Solution array.
        """
        import matplotlib.pyplot as plt

        for i in range(self.dim):
            if self.var_names and self.var_names[i]:
                label = f'Quantum {self.var_names[i]}'
                vals = [u_step[i] for u_step in u]
                plt.plot(t, vals, label=label)
                if u_num is not None:
                    label = f'Numerical {self.var_names[i]}'
                    plt.plot(t, u_num[i], label=label)   
        plt.xlabel('Time')
        plt.ylabel('Variables')
        plt.title(f'Solution of {self.name} ODE System')
        plt.legend()
        plt.savefig(name if name else f'{self.name}_solution.png')

class QuantumTerm():
    """A class to represent a quantum term in the cost function with its circuit and normalization coefficient."""
    
    def __init__(self, circuit: qk.QuantumCircuit, find_parameters: callable, gate_type: str = "linear", u_deg: int= 0):
        """
        Initialize the quantum term.

        Args:
            circuit (qk.QuantumCircuit): The quantum circuit representing the term.
            find_parameters (callable): A function to determine parameters to the circuit and find the normalization coefficient.

        """
        self.circuit = circuit
        self.find_parameters = find_parameters
        self.gate_type = gate_type
        self.u_deg = u_deg


        
        