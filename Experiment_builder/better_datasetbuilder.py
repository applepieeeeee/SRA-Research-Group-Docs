import numpy as np
from lattice import random_bimodal_instance, random_spin_state, simulate_monolithic

NX = NY = NZ = 6
SEED = 42

BURN_IN_SWEEPS = 5000
NUM_SAMPLES = 500        # kept samples per bucket
THIN = 50                # spacing between kept samples

# our different betas for annealing
BETA_GRID = np.linspace(0.1, 3.0, 12).astype(np.float32)

rng = np.random.default_rng(SEED)
instance = random_bimodal_instance(NX, NY, NZ, rng)

buckets = []
for k, beta in enumerate(BETA_GRID):
    equilibrium_states = []

    def collect_sample(step, state, energy):
        s = (step + 1) - BURN_IN_SWEEPS
        if s > 0 and s % THIN == 0:
            equilibrium_states.append(state.copy())

    initial_state = random_spin_state(NX, NY, NZ, rng)
    simulate_monolithic(
        instance=instance,
        beta=float(beta),                       # equilibrium at this temperature
        steps=BURN_IN_SWEEPS + NUM_SAMPLES * THIN,
        seed=SEED + k,                          # decorrelate buckets
        initial_state=initial_state,
        record_history=False,
        on_step=collect_sample,
    )
    buckets.append(np.stack(equilibrium_states).astype(np.int8))

states = np.stack(buckets)   # 

np.savez_compressed(
    "equilibrium_library_seed42.npz",
    beta_grid=BETA_GRID,
    states=states,
    bond_x=instance.bond_x,
    bond_y=instance.bond_y,
    bond_z=instance.bond_z,
    fields=instance.fields,
)
