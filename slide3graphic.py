import matplotlib.pyplot as plt
import numpy as np

def linear_beta(steps: int, beta_start: float = 0.1, beta_end: float = 3.0) -> np.ndarray:
    return np.linspace(beta_start, beta_end, steps, dtype=np.float32)

def geometric_beta(steps: int, beta_start: float = 0.1, beta_end: float = 3.0) -> np.ndarray:
    return np.geomspace(beta_start, beta_end, steps, dtype=np.float32)

def constant_beta(steps: int, beta: float = 1.0) -> np.ndarray:
    return np.full(steps, beta, dtype=np.float32)

N_STEPS = 4000
beta = linear_beta(N_STEPS)   # actual schedule used
T = 1.0 / beta                # temperature

fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=200)

ax.plot(np.arange(N_STEPS), T, color="#0b2994", linewidth=2.2)

ax.set_title("Annealing Schedule", fontsize=16, pad=10)
ax.set_xlabel("Steps", fontsize=16)
ax.set_ylabel("Temperature", fontsize=16)

ax.set_xlim(0, N_STEPS)
ax.set_ylim(0, T.max() * 1.05)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=8)
ax.grid(False)

fig.tight_layout()
fig.savefig("fig_schedule_temperature.png", transparent=True)
print("saved, T range:", T.min(), T.max())

plt.show()