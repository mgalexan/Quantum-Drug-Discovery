from simulator import QuantumForward

import hydra
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

@hydra.main(version_base=None, config_path="config", config_name="plan")
def main(cfg):
    qforward = QuantumForward(cfg)
    n_steps = 7500
    t = [i * cfg.cost.EulerCost.tau for i in range(n_steps)]
    # numerical solution
    scipy_dt = qforward.equation.ode_system
    from scipy.integrate import solve_ivp
    sol = solve_ivp(scipy_dt, [0, n_steps * cfg.cost.EulerCost.tau], cfg.initial_conditions, t_eval=t)
    x_num = sol.y[0]
    y_num = sol.y[1]
    states = []
    for _ in tqdm((range(n_steps))):
        qforward.step()
        states.append(qforward.current_state)

    x = [s[0] for s in states]
    y = [s[1] for s in states]

    np.savez("test.npz", x=x, y=y, x_num=x_num, y_num=y_num)

    plt.plot(t, x, label="Quantum x")
    plt.plot(t, y, label="Quantum y")
    plt.plot(t, x_num, label="Numerical x")
    plt.plot(t, y_num, label="Numerical y")
    plt.legend()
    plt.title("Quantum vs Numerical Solution of the Lotka-Volterra Equations")
    plt.savefig("test.png")
    plt.clf()
    plt.plot(x, y, label="Quantum Phase Space")
    plt.plot(x_num, y_num, label="Numerical Phase Space")
    plt.legend()
    plt.title("Phase Space: Quantum vs Numerical Solution of the Lotka-Volterra Equations")
    plt.savefig("phase_space.png")



if __name__ == "__main__":
    main()
