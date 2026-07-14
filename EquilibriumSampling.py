import numpy as np
from lattice import simulate_monolithic, simulate_partitioned,PartitionSpec, GhostBoundary, random_bimodal_instance, random_spin_state
import matplotlib.pyplot as plt

NX = NY = NZ = 12 
BETA = 1.0
STEPS = 100000
SEED = 42

rng = np.random.default_rng(SEED)

instance = random_bimodal_instance(NX, NY, NZ, rng)
initial_state = random_spin_state(NX, NY, NZ, rng)

equilibrium_data = np.load("equilibrium_dataset.npz")
equilibrium_states = equilibrium_data["states"]


#this is the function that we will use in simulate_partitioned
def equilibrium_update(step, ghost_age, ghosts, state, instance, partitions, beta, rng):
    #right now we are randomly selecting an equilibrium sample
    #later we should change this to a context conditioned sampling technique
    sample = equilibrium_states[rng.integers(len(equilibrium_states))]

    #sample = rng.choice([-1, 1], size=state.shape)

    nx, ny, nz = state.shape
    updated = {}

    def update(old, proposed, local, bond):
        delta_energy = -bond * local * (proposed - old)
        accepted = (delta_energy <= 0) | (rng.random(old.shape) < np.exp(-beta * delta_energy))
        return np.where(accepted, proposed, old).astype(np.float32)

    for p in partitions:
        x0, x1 = p.x_start, p.x_end
        y0, y1 = p.y_start, p.y_end
        z0, z1 = p.z_start, p.z_end
        g = ghosts[p.partition_id]

        updated[p.partition_id] = GhostBoundary(
            x_lo=update(g.x_lo, sample[(x0-1)%nx, y0:y1, z0:z1], state[x0, y0:y1, z0:z1], instance.bond_x[(x0-1)%nx, y0:y1, z0:z1]),
            x_hi=update(g.x_hi, sample[x1%nx, y0:y1, z0:z1], state[x1-1, y0:y1, z0:z1], instance.bond_x[x1-1, y0:y1, z0:z1]),
            y_lo=update(g.y_lo, sample[x0:x1, (y0-1)%ny, z0:z1], state[x0:x1, y0, z0:z1], instance.bond_y[x0:x1, (y0-1)%ny, z0:z1]),
            y_hi=update(g.y_hi, sample[x0:x1, y1%ny, z0:z1], state[x0:x1, y1-1, z0:z1], instance.bond_y[x0:x1, y1-1, z0:z1]),
            z_lo=update(g.z_lo, sample[x0:x1, y0:y1, (z0-1)%nz], state[x0:x1, y0:y1, z0], instance.bond_z[x0:x1, y0:y1, (z0-1)%nz]),
            z_hi=update(g.z_hi, sample[x0:x1, y0:y1, z1%nz], state[x0:x1, y0:y1, z1-1], instance.bond_z[x0:x1, y0:y1, z1-1]),
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