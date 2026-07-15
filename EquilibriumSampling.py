import numpy as np
from lattice import simulate_monolithic, simulate_partitioned,PartitionSpec, GhostBoundary, random_bimodal_instance, random_spin_state
import matplotlib.pyplot as plt

NX = NY = NZ = 6
BETA = 1.0
STEPS = 1000
SEED = 42

rng = np.random.default_rng(SEED)


instance = random_bimodal_instance(NX, NY, NZ, rng)
initial_state = random_spin_state(NX, NY, NZ, rng)

equilibrium_data = np.load("equilibrium_dataset.npz")
equilibrium_states = equilibrium_data["states"]


#this is the function that we will use in simulate_partitioned
def equilibrium_update(step, ghost_age, ghosts, state, instance, partitions, beta, rng):
    nx, ny, nz = state.shape
    age = min(ghost_age, len(equilibrium_states) - 1)

    past_states = equilibrium_states[:-age]
    future_states = equilibrium_states[age:]
    updated = {}

    for p in partitions:
        x0, x1 = p.x_start, p.x_end
        y0, y1 = p.y_start, p.y_end
        z0, z1 = p.z_start, p.z_end
        g = ghosts[p.partition_id]

        current = state[x0:x1, y0:y1, z0:z1]
        stored = past_states[:, x0:x1, y0:y1, z0:z1]

        distances = np.sum(stored != current, axis=(1, 2, 3))

        k = min(20, len(past_states))
        nearest = np.argpartition(distances, k - 1)[:k]
        sample = future_states[rng.choice(nearest)]

        updated[p.partition_id] = GhostBoundary(
            x_lo=None if g.x_lo is None else sample[(x0 - 1) % nx, y0:y1, z0:z1].astype(np.float32),
            x_hi=None if g.x_hi is None else sample[x1 % nx, y0:y1, z0:z1].astype(np.float32),
            y_lo=None if g.y_lo is None else sample[x0:x1, (y0 - 1) % ny, z0:z1].astype(np.float32),
            y_hi=None if g.y_hi is None else sample[x0:x1, y1 % ny, z0:z1].astype(np.float32),
            z_lo=None if g.z_lo is None else sample[x0:x1, y0:y1, (z0 - 1) % nz].astype(np.float32),
            z_hi=None if g.z_hi is None else sample[x0:x1, y0:y1, z1 % nz].astype(np.float32),
        )

    return updated


def frozen(step, ghost_age, ghosts, state, instance, partitions, beta, rng):
           return ghosts



frozenresult = simulate_partitioned(instance=instance, beta=BETA, steps=STEPS, seed=SEED, partition_spec=PartitionSpec(2, 2, 2), communication_interval=10, initial_state=initial_state, ghost_update_fn=frozen)
print("Frozen Best energy:", frozenresult.best_energy)
print("Frozen Final energy:", frozenresult.energies[-1])

equilibriumresult = simulate_partitioned(instance=instance, beta=BETA, steps=STEPS, seed=SEED, partition_spec=PartitionSpec(2, 2, 2), communication_interval=10, initial_state=initial_state, ghost_update_fn=equilibrium_update)
print()
print("Equilibrium Best energy:", equilibriumresult.best_energy)
print("Equilibrium Final energy:", equilibriumresult.energies[-1])


monolithic_result = simulate_monolithic(instance=instance, beta=BETA, steps=STEPS, seed=SEED, initial_state=initial_state)

print()
print("Best energy:", monolithic_result.best_energy)
print("Final energy:", monolithic_result.energies[-1])

plt.plot(frozenresult.energies, label="Frozen Partitioned")
plt.plot(monolithic_result.energies, label="Monolithic")
plt.plot(equilibriumresult.energies, label="Equilibirum Sampling Partitioned")
plt.xlabel("Steps")
plt.ylabel("Energy (Lower is better)")
plt.legend()
plt.show()
