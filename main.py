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

    qforward.equation.plot_results(t, states, name=f"plots/{cfg.name}.png", u_num=sol.y)

    # save the results from the numerical solution
    np.savez(f"results/{cfg.name}_numerical.npz", t=t, x=sol.y)

    # save the results from the quantum forward simulation
    states = np.array(states).T
    np.savez(f"results/{cfg.name}_quantum.npz", t=t, x=states)


if __name__ == "__main__":
    main()
