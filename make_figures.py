"""
make_figures.py  --  the FAST half (plotting + export only).

Reads figure_data.npz (written by run_experiments.py), draws Figures 0-3,
and exports the plotted curves + final values to CSV. Imports nothing
expensive, so it starts instantly. Re-run this freely while restyling graphs.

    python run_experiments.py      ->  writes figure_data.npz   (slow, rare)
    python make_figures.py         ->  reads it, draws + exports (fast, often)
"""
import numpy as np
import matplotlib.pyplot as plt

# ---- global font sizes: bump titles and axis labels ----
plt.rcParams.update({
    "axes.titlesize": 20,
    "axes.labelsize": 18,
})

CACHE = "figure_data.npz"
d = np.load(CACHE)

# ---- config comes from the data file, so it can't drift from the simulation ----
steps          = d["steps"]
S_LIST         = list(d["S_list"])
INSTANCE_SEEDS = list(d["instance_seeds"])
FIG1_S         = int(d["FIG1_S"])
STEPS          = int(d["STEPS"])
BETA           = d["beta"]

# reassemble the same dicts the old figures.py had
resid_steps = {"frozen": list(d["resid_frozen"]),
               "mono":   list(d["resid_mono"]),
               "hybrid": list(d["resid_hybrid"])}
succ_steps  = {"frozen": list(d["succ_frozen"]),
               "mono":   list(d["succ_mono"]),
               "hybrid": list(d["succ_hybrid"])}
finalS      = {"frozen": {S: list(d["finalS_frozen"][:, j]) for j, S in enumerate(S_LIST)},
               "hybrid": {S: list(d["finalS_hybrid"][:, j]) for j, S in enumerate(S_LIST)}}
mono_final  = list(d["mono_final"])

def bootstrap_ci(list_of_curves, n_boot=2000, ci=95, seed=0):
    # mean across instances, plus a 95% CI by resampling instances with replacement
    A = np.stack(list_of_curves)                 # (n_instances, ...)
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))   # resampled instance indices
    boot_means = A[idx].mean(axis=1)             # (n_boot, ...)
    lo = np.percentile(boot_means, (100 - ci) / 2, axis=0)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2, axis=0)
    return A.mean(0), lo, hi

# =================== FIGURE 0: annealing schedule (temperature) ===================
plt.figure()
plt.plot(steps, 1.0 / BETA)
plt.xlabel("Steps"); plt.ylabel("Temperature (T = 1/beta)")
plt.title("Annealing Schedule", pad=15)
plt.savefig("fig_schedule_temperature.png", dpi=150, bbox_inches="tight")

# =================== FIGURE 1: residual energy vs steps ===================
plt.figure()
for key, lab in [("frozen","Frozen Partitioned"), ("mono","Monolithic"),
                 ("hybrid","Equilibrium Sampling Partitioned")]:
    m, lo, hi = bootstrap_ci(resid_steps[key])
    plt.plot(steps, m, label=lab)
    plt.fill_between(steps, lo, hi, alpha=0.15)
plt.xlabel("Steps"); plt.ylabel("Mean Residual Energy (E - E0)")
plt.xlim(right=2500)
plt.title(f"Residual Energy, averaged over {len(INSTANCE_SEEDS)} instances", pad=15)
plt.legend()
plt.savefig("fig_residual_avg.png", dpi=150, bbox_inches="tight")

# =================== FIGURE 2: success probability vs steps ===================
plt.figure()
for key, lab in [("frozen","Frozen Partitioned"), ("mono","Monolithic"),
                 ("hybrid","Equilibrium Sampling Partitioned")]:
    m, lo, hi = bootstrap_ci(succ_steps[key])
    plt.plot(steps, m, label=lab)
    plt.fill_between(steps, np.clip(lo, 0, 1), np.clip(hi, 0, 1), alpha=0.15)
plt.xlabel("Steps"); plt.ylabel("Success Probability")
plt.title(f"Success Probability, averaged over {len(INSTANCE_SEEDS)} instances", pad=15)
plt.ylim(-0.02, 1.02); plt.legend()
plt.savefig("fig_success_steps_avg.png", dpi=150, bbox_inches="tight")

# =================== FIGURE 3: success probability vs S ===================
plt.figure()
for method, lab in [("frozen","Frozen Partitioned"),
                    ("hybrid","Equilibrium Sampling Partitioned")]:
    ms, los, his = [], [], []
    for S in S_LIST:
        m, lo, hi = bootstrap_ci(finalS[method][S])
        ms.append(m); los.append(lo); his.append(hi)
    ms = np.array(ms); los = np.array(los); his = np.array(his)
    yerr = np.vstack([ms - los, his - ms])   # asymmetric: [lower, upper] distances from mean
    plt.errorbar(S_LIST, ms, yerr=yerr, marker="o", capsize=3, label=lab)
mono_m, mono_lo, mono_hi = bootstrap_ci(mono_final)
plt.axhline(mono_m, ls="--", color="gray", label="Monolithic (reference)")
plt.fill_between([min(S_LIST), max(S_LIST)], mono_lo, mono_hi, color="gray", alpha=0.10)
plt.xscale("log")
plt.xlabel("Communication interval S (log scale)"); plt.ylabel("Final Success Probability")
plt.title(f"Success vs Communication Sparsity, {len(INSTANCE_SEEDS)} instances", pad=15)
plt.ylim(-0.02, 1.02); plt.legend()
plt.savefig("fig_success_vs_S_avg.png", dpi=150, bbox_inches="tight")

# =================== EXPORT: CSVs for re-plotting + final values ===================
def _save_steps_csv(fname, keys_labels, source):
    cols, header, finals = [steps.astype(float)], ["step"], {}
    for key, _ in keys_labels:
        m, lo, hi = bootstrap_ci(source[key])
        cols += [m, lo, hi]
        header += [f"{key}_mean", f"{key}_lo", f"{key}_hi"]
        finals[key] = (m[-1], lo[-1], hi[-1])
    np.savetxt(fname, np.column_stack(cols), delimiter=",",
               header=",".join(header), comments="")
    return finals

keys = [("frozen", "Frozen"), ("mono", "Monolithic"), ("hybrid", "Equilibrium")]
resid_final = _save_steps_csv("data_residual_vs_steps.csv", keys, resid_steps)
succ_final  = _save_steps_csv("data_success_vs_steps.csv",  keys, succ_steps)

with open("data_success_vs_S.csv", "w") as f:
    f.write("S,frozen_mean,frozen_lo,frozen_hi,hybrid_mean,hybrid_lo,hybrid_hi\n")
    for S in S_LIST:
        fm, flo, fhi = bootstrap_ci(finalS["frozen"][S])
        hm, hlo, hhi = bootstrap_ci(finalS["hybrid"][S])
        f.write(f"{S},{fm},{flo},{fhi},{hm},{hlo},{hhi}\n")

# ---- final values the curves approach at the last step ----
print(f"\n=== FINAL VALUES at step {STEPS-1}  (mean [95% CI]) ===")
print("Residual energy (E - E0):")
for key, lab in keys:
    m, lo, hi = resid_final[key]
    print(f"  {lab:12s}: {m:.3f}  [{lo:.3f}, {hi:.3f}]")
print(f"Success probability at S={FIG1_S}:")
for key, lab in keys:
    m, lo, hi = succ_final[key]
    print(f"  {lab:12s}: {m:.3f}  [{lo:.3f}, {hi:.3f}]")
print("Final success probability vs S:")
for S in S_LIST:
    fm, flo, fhi = bootstrap_ci(finalS["frozen"][S])
    hm, hlo, hhi = bootstrap_ci(finalS["hybrid"][S])
    print(f"  S={S:5d}  frozen={fm:.3f} [{flo:.3f},{fhi:.3f}]   hybrid={hm:.3f} [{hlo:.3f},{hhi:.3f}]")
print(f"  monolithic reference={mono_m:.3f} [{mono_lo:.3f},{mono_hi:.3f}]")

with open("data_final_values.csv", "w") as f:
    f.write("figure,method,x,mean,lo,hi\n")
    for key, _ in keys:
        m, lo, hi = resid_final[key]; f.write(f"residual_vs_steps,{key},{STEPS-1},{m},{lo},{hi}\n")
    for key, _ in keys:
        m, lo, hi = succ_final[key];  f.write(f"success_vs_steps,{key},{STEPS-1},{m},{lo},{hi}\n")
    for S in S_LIST:
        fm, flo, fhi = bootstrap_ci(finalS["frozen"][S])
        hm, hlo, hhi = bootstrap_ci(finalS["hybrid"][S])
        f.write(f"success_vs_S,frozen,{S},{fm},{flo},{fhi}\n")
        f.write(f"success_vs_S,hybrid,{S},{hm},{hlo},{hhi}\n")
    f.write(f"success_vs_S,mono,NA,{mono_m},{mono_lo},{mono_hi}\n")

print("done")