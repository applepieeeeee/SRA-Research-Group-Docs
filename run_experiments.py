"""
run_experiments.py  --  the SLOW half (simulations only).

Runs every instance/method/S combination exactly as the old figures.py did,
then writes all raw per-instance arrays to figure_data.npz.
No plotting, no CSVs. Run this only when the physics changes.

    python run_experiments.py      ->  writes figure_data.npz
    python make_figures.py         ->  reads it, draws figures + CSVs
"""
import numpy as np
from lattice import (simulate_monolithic, simulate_partitioned, PartitionSpec,
                     GhostBoundary, random_bimodal_instance, random_spin_state)
from schedules import linear_beta

# ----------------------------- configuration -----------------------------
NX = NY = NZ = 6
STEPS          = 4000
N_RUNS         = 8                        # runs per instance per point
INSTANCE_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
S_LIST         = [10, 50, 200, 1000]
FIG1_S         = 200                     # S used for the per-step figures (must be in S_LIST)
BETA_FREEZE    = 1.5
TOL            = 0.0
BETA = linear_beta(STEPS)

CACHE = "figure_data.npz"

# these globals are reassigned for each instance before its runs
instance = None
beta_grid = None
library_states = None

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

GHOST = {"frozen": frozen, "hybrid": hybrid}

def bestsofar(kind, S, run_seed):
    init = random_spin_state(NX, NY, NZ, np.random.default_rng(run_seed))
    if kind == "mono":
        res = simulate_monolithic(instance=instance, beta=BETA, steps=STEPS,
                                  seed=run_seed, initial_state=init)
    else:
        res = simulate_partitioned(instance=instance, beta=BETA, steps=STEPS, seed=run_seed,
                                   partition_spec=PartitionSpec(2,2,2), communication_interval=S,
                                   initial_state=init, ghost_update_fn=GHOST[kind])
    return np.minimum.accumulate(res.energies)

# --------------- collect, one instance at a time ---------------
steps = np.arange(STEPS)

resid_steps = {"frozen": [], "mono": [], "hybrid": []}   # per-instance residual-vs-steps at FIG1_S
succ_steps  = {"frozen": [], "mono": [], "hybrid": []}   # per-instance success-vs-steps at FIG1_S
finalS      = {"frozen": {S: [] for S in S_LIST},
               "hybrid": {S: [] for S in S_LIST}}
mono_final  = []

for iseed in INSTANCE_SEEDS:
    instance = random_bimodal_instance(NX, NY, NZ, np.random.default_rng(iseed))
    lib = np.load(f"equilibrium_library_seed{iseed}.npz")
    beta_grid = lib["beta_grid"]; library_states = lib["states"]

    data = {}
    for method in ("frozen", "hybrid"):
        for S in S_LIST:
            arr = np.empty((N_RUNS, STEPS), dtype=np.float32)
            for r in range(N_RUNS):
                arr[r] = bestsofar(method, S, iseed * 10000 + r)
            data[(method, S)] = arr
    monoarr = np.empty((N_RUNS, STEPS), dtype=np.float32)
    for r in range(N_RUNS):
        monoarr[r] = bestsofar("mono", None, iseed * 10000 + r)

    E0 = float(min(monoarr.min(), *(a.min() for a in data.values())))

    # residual (E - E0) averaged over runs, at FIG1_S  -- comparable across instances
    resid_steps["frozen"].append((data[("frozen", FIG1_S)] - E0).mean(0))
    resid_steps["hybrid"].append((data[("hybrid", FIG1_S)] - E0).mean(0))
    resid_steps["mono"].append((monoarr - E0).mean(0))
    # success (reached ground) averaged over runs, at FIG1_S
    succ_steps["frozen"].append((data[("frozen", FIG1_S)] <= E0 + TOL).mean(0))
    succ_steps["hybrid"].append((data[("hybrid", FIG1_S)] <= E0 + TOL).mean(0))
    succ_steps["mono"].append((monoarr <= E0 + TOL).mean(0))
    # final success vs S
    for S in S_LIST:
        finalS["frozen"][S].append((data[("frozen", S)][:, -1] <= E0 + TOL).mean())
        finalS["hybrid"][S].append((data[("hybrid", S)][:, -1] <= E0 + TOL).mean())
    mono_final.append((monoarr[:, -1] <= E0 + TOL).mean())
    print(f"instance {iseed}: E0={E0}")

# --------------- save everything make_figures.py needs ---------------
# scalars/config are saved too, so the figure script reads them from here
# and the two files can never disagree about S_LIST / STEPS / FIG1_S / seeds.
np.savez_compressed(
    CACHE,
    steps=steps, S_list=np.array(S_LIST), instance_seeds=np.array(INSTANCE_SEEDS),
    FIG1_S=FIG1_S, STEPS=STEPS, TOL=TOL, beta=BETA,
    resid_frozen=np.stack(resid_steps["frozen"]),
    resid_mono=np.stack(resid_steps["mono"]),
    resid_hybrid=np.stack(resid_steps["hybrid"]),
    succ_frozen=np.stack(succ_steps["frozen"]),
    succ_mono=np.stack(succ_steps["mono"]),
    succ_hybrid=np.stack(succ_steps["hybrid"]),
    finalS_frozen=np.array([finalS["frozen"][S] for S in S_LIST]).T,   # (n_instances, len(S_list))
    finalS_hybrid=np.array([finalS["hybrid"][S] for S in S_LIST]).T,
    mono_final=np.array(mono_final),
)
print(f"saved raw per-instance data to {CACHE}")