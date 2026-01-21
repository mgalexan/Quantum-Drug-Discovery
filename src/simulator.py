from equations.base_equation import Equation, QuantumTerm
from costs.base_cost import BaseCost
import qiskit as qk
from omegaconf import DictConfig
import numpy as np
import nevergrad as ng
from functools import partial


# Import all specific instances
from costs.euler import EulerCost

from ansatz import ULA

from equations.lotka_volterra import LotkaVolterra



class QuantumForward():
    """A class to perform quantum forward evolution for ODE systems."""
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize the quantum forward evolution with an equation.

        Args:
            equation (Equation): The equation object containing the ODE system and gate equivalents.
        """
        equation_cfg = cfg.equation
        equation_name = list(equation_cfg.keys())[0]
        equation_params = equation_cfg[equation_name]
        self.equation: Equation = globals()[equation_name](**equation_params)

        ansatz_cfg = cfg.ansatz
        ansatz_name = list(ansatz_cfg.keys())[0]
        ansatz_params = ansatz_cfg[ansatz_name]
        self.ansatz, self.ansatz_params = globals()[ansatz_name](**ansatz_params)

        cost_cfg = cfg.cost
        cost_name = list(cost_cfg.keys())[0]
        cost_params = cost_cfg[cost_name]
        self.cost: BaseCost = globals()[cost_name](self.equation, cost_params)
        self.tau = cost_params.tau

        self.initial_conditions = np.array(cfg.initial_conditions)

        self.workers = cfg.get("workers", 1)

        self._compile()
    
    def _compile(self) -> None:
        """
        Compile the quantum forward evolution components.
        """
        # Compile the cost function with the ansatz
        self.cost.compile_with_ansatz(self.ansatz, self.ansatz_params)

        # Find initial conditions parameters
        lambda_0 = np.linalg.norm(self.initial_conditions)
        u0_normalized = self.initial_conditions / lambda_0

        
        def _ic_cost(lambdas) -> float:
            """
            Compute the initial condition cost.

            Args:
                lambdas: The specific values for the ansatz parameters.
            
            Returns:
                float: The computed initial condition cost.
            """
            param_map = {self.ansatz_params[i]: lambdas[i] for i in range(len(lambdas))}
            circuit_ic = self.ansatz.assign_parameters(param_map, inplace=False)
            op = qk.quantum_info.Operator(circuit_ic)
            statevector = op.data[:, 0]
            cost = np.linalg.norm(statevector - u0_normalized) ** 2
            return cost
        optimizer = ng.optimizers.NGOpt(parametrization=len(self.ansatz_params), budget=1000)
        ic_result = optimizer.minimize(_ic_cost)
        self.ic_lambdas = [lambda_0] + list(ic_result.value)
        self.current_lambdas = self.ic_lambdas

        # Compute the first state
        param_map = {self.ansatz_params[i]: self.ic_lambdas[i+1] for i in range(len(self.ansatz_params))}
        ansatz_bound = self.ansatz.assign_parameters(param_map, inplace=False)
        op = qk.quantum_info.Operator(ansatz_bound)
        statevector = op.data[:, 0] * self.ic_lambdas[0]
        self.current_state = np.real(statevector)
        self.current_time = 0.0
        
    
    def step(self) -> np.ndarray:
        
        # Cost function at current step
        cost_step = partial(self.cost.compute_cost, lambdas_prev= self.current_lambdas, u_prev=self.current_state, t=self.current_time)
        # Optimizer
        optimizer = ng.optimizers.NGOpt(parametrization=len(self.current_lambdas), budget=1000, num_workers=self.workers)

        # Add initial guess to previous step
        optimizer.value = self.current_lambdas
        result = optimizer.minimize(cost_step)
        
        self.current_lambdas = result.value
        param_map = {self.ansatz_params[i]: self.current_lambdas[i+1] for i in range(len(self.ansatz_params))}
        ansatz_bound = self.ansatz.assign_parameters(param_map, inplace=False)
        op = qk.quantum_info.Operator(ansatz_bound)
        statevector = op.data[:, 0] * self.current_lambdas[0]
        self.current_state = np.real(statevector)
        self.current_time += self.tau
                
    
# Test the simulation class with config
if __name__ == "__main__":
    with open("config/plan.yaml", 'r') as f:
        import yaml
        config_dict = yaml.safe_load(f)
    
    config = DictConfig(config_dict)

    qforward = QuantumForward(config)
    print(qforward.equation.name)
    print("Initial condition lambdas:", qforward.ic_lambdas)
    print("Statevector from ansatz at t=0:", qforward.current_state)
    qforward.step()
    print(qforward.current_state)