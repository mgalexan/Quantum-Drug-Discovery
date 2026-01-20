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

class QuantumTerm():
    """A class to represent a quantum term in the cost function with its circuit and normalization coefficient."""
    
    def __init__(self, circuit: qk.QuantumCircuit, find_parameters: callable):
        """
        Initialize the quantum term.

        Args:
            circuit (qk.QuantumCircuit): The quantum circuit representing the term.
            find_parameters (callable): A function to determine parameters to the circuit and find the normalization coefficient.
        """
        self.circuit = circuit
        self.find_parameters = find_parameters


        
        