import numpy as np

from lattice import random_bimodal_instance, random_spin_state, simulate_monolithic


NX = NY = NZ = 12
BETA = 1.0
BURN_IN_SWEEPS = 5000
NUM_SAMPLES = 2000
SEED = 42

rng = np.random.default_rng(SEED)
instance = random_bimodal_instance(NX, NY, NZ, rng)
initial_state = random_spin_state(NX, NY, NZ, rng)

equilibrium_states = []


def collect_sample(step, state, energy):
    completed_sweeps = step + 1
    sweeps_after_burn_in = completed_sweeps - BURN_IN_SWEEPS

    if (sweeps_after_burn_in > 0):
        equilibrium_states.append(state.copy())


total_sweeps = BURN_IN_SWEEPS + NUM_SAMPLES

simulate_monolithic(
    instance=instance,
    beta=BETA,
    steps=total_sweeps,
    seed=SEED,
    initial_state=initial_state,
    record_history=False,
    on_step=collect_sample,
)

equilibrium_states = np.stack(equilibrium_states).astype(np.int8)
np.savez_compressed("equilibrium_dataset.npz", states=equilibrium_states, bond_x=instance.bond_x, bond_y=instance.bond_y, bond_z=instance.bond_z, fields=instance.fields, beta=BETA)
