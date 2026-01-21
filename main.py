from simulator import QuantumForward

import hydra
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

@hydra.main(version_base=None, config_path="config", config_name="plan")
def main(cfg):
    qforward = QuantumForward(cfg)
    n_steps = cfg.n_steps
    t = [i * cfg.cost.EulerCost.tau for i in range(n_steps + 1)]
    # numerical solution
    scipy_dt = qforward.equation.ode_system
    from scipy.integrate import solve_ivp
    sol = solve_ivp(scipy_dt, [0, n_steps * cfg.cost.EulerCost.tau], cfg.initial_conditions, t_eval=t)

    states = [qforward.current_state]
    for _ in tqdm((range(n_steps))):
        qforward.step()
        states.append(qforward.current_state)

    qforward.equation.plot_results(t, states, name=f"{cfg.name}.png", u_num=sol.y)



if __name__ == "__main__":
    main()
