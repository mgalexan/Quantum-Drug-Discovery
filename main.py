from simulator import QuantumForward

import hydra
from tqdm import tqdm
import matplotlib.pyplot as plt

@hydra.main(version_base=None, config_path="config", config_name="plan")
def main(cfg):
    qforward = QuantumForward(cfg)
    states = []
    for _ in tqdm((range(20))):
        qforward.step()
        states.append(qforward.current_state)

    x = [s[0] for s in states]
    y = [s[1] for s in states]
    plt.plot(x)
    plt.plot(y)
    plt.savefig("test.png")

if __name__ == "__main__":
    main()
