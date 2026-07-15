import numpy as np
import matplotlib.pyplot as plt
from lattice import (simulate_monolithic, simulate_partitioned, PartitionSpec,
                     GhostBoundary, random_bimodal_instance, random_spin_state)
from schedules import linear_beta

NX = NY = NZ = 6
STEPS       = 10000
COMM        = 1000
BETA_FREEZE = 1.5
SEED        = 42

BETA = linear_beta(STEPS)

rng0 = np.random.default_rng(SEED)
instance = random_bimodal_instance(NX, NY, NZ, rng0)
init = random_spin_state(NX, NY, NZ, rng0)

lib = np.load("equilibrium_library_seed42.npz")
beta_grid = lib["beta_grid"]; library_states = lib["states"]

def equilibrium_update(step, ghost_age, ghosts, state, instance, partitions, beta, rng):
    k = int(np.argmin(np.abs(beta_grid - beta)))
    bucket = library_states[k]; sample = bucket[rng.integers(len(bucket))]
    nx, ny, nz = state.shape; updated = {}
    def upd(old, proposed, local, bond):
        de = -bond * local * (proposed - old)
        acc = (de <= 0) | (rng.random(old.shape) < np.exp(-beta * np.clip(de, None, 50)))
        return np.where(acc, proposed, old).astype(np.float32)
    for p in partitions:
        x0,x1=p.x_start,p.x_end; y0,y1=p.y_start,p.y_end; z0,z1=p.z_start,p.z_end
        g = ghosts[p.partition_id]
        updated[p.partition_id] = GhostBoundary(
            x_lo=upd(g.x_lo, sample[(x0-1)%nx,y0:y1,z0:z1], state[x0,y0:y1,z0:z1], instance.bond_x[(x0-1)%nx,y0:y1,z0:z1]),
            x_hi=upd(g.x_hi, sample[x1%nx,y0:y1,z0:z1], state[x1-1,y0:y1,z0:z1], instance.bond_x[x1-1,y0:y1,z0:z1]),
            y_lo=upd(g.y_lo, sample[x0:x1,(y0-1)%ny,z0:z1], state[x0:x1,y0,z0:z1], instance.bond_y[x0:x1,(y0-1)%ny,z0:z1]),
            y_hi=upd(g.y_hi, sample[x0:x1,y1%ny,z0:z1], state[x0:x1,y1-1,z0:z1], instance.bond_y[x0:x1,y1-1,z0:z1]),
            z_lo=upd(g.z_lo, sample[x0:x1,y0:y1,(z0-1)%nz], state[x0:x1,y0:y1,z0], instance.bond_z[x0:x1,y0:y1,(z0-1)%nz]),
            z_hi=upd(g.z_hi, sample[x0:x1,y0:y1,z1%nz], state[x0:x1,y0:y1,z1-1], instance.bond_z[x0:x1,y0:y1,z1-1]))
    return updated

def frozen(step, ghost_age, ghosts, state, instance, partitions, beta, rng):
    return ghosts

def hybrid(step, ghost_age, ghosts, state, instance, partitions, beta, rng):
    if beta >= BETA_FREEZE:
        return ghosts
    return equilibrium_update(step, ghost_age, ghosts, state, instance, partitions, beta, rng)

def part(fn):
    return simulate_partitioned(instance=instance, beta=BETA, steps=STEPS, seed=SEED,
                                partition_spec=PartitionSpec(2,2,2), communication_interval=COMM,
                                initial_state=init, ghost_update_fn=fn)

froz = np.minimum.accumulate(part(frozen).energies)
equi = np.minimum.accumulate(part(hybrid).energies)
mono = np.minimum.accumulate(simulate_monolithic(instance=instance, beta=BETA, steps=STEPS,
                                                 seed=SEED, initial_state=init).energies)
E0 = float(min(froz.min(), equi.min(), mono.min()))
steps = np.arange(STEPS)

# 1. annealing schedule
plt.figure()
plt.plot(steps, BETA)
plt.xlabel("Steps")
plt.ylabel("Beta (1/T)")
plt.title("Annealing Schedule")
plt.savefig("fig1_schedule.png", dpi=150, bbox_inches="tight")

# 2. energy
plt.figure()
plt.plot(froz, label="Frozen Partitioned")
plt.plot(mono, label="Monolithic")
plt.plot(equi, label="Equilibrium Sampling Partitioned")
plt.xlabel("Steps")
plt.ylabel("Energy (Lower is better)")
plt.legend()
plt.savefig("fig2_energy.png", dpi=150, bbox_inches="tight")

# 3. residual energy
plt.figure()
plt.plot(froz - E0, label="Frozen Partitioned")
plt.plot(mono - E0, label="Monolithic")
plt.plot(equi - E0, label="Equilibrium Sampling Partitioned")
plt.yscale("linear")
plt.xlabel("Steps")
plt.ylabel("Residual Energy (E - E0)")
plt.legend()
plt.savefig("fig3_residual.png", dpi=150, bbox_inches="tight")

# 4. success probability TBD possibly
