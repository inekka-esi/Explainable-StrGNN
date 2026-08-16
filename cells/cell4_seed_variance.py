# =============================================================================
# CELL 4 / 4  —  Seed variance
# Run after CELL 3. Repeats the full explainer training and evaluation over
# several seeds and reports mean +/- sd for every metric.
#
# The detector is not retrained: the frozen StrGNN checkpoint is held fixed, so
# the dispersion measured here is explainer variance alone, which is the
# quantity a comparison between explainers actually depends on. Single-seed
# explainability tables are not interpretable when between-method gaps are of
# the same order as between-seed spread, and on this benchmark they are.
#
# No hyperparameter is changed between seeds. Anything tuned after seeing this
# output is tuned on the test set.
# =============================================================================

import copy
import numpy as np

SEEDS = [1, 2, 3]
RUNS = {}

for sd in SEEDS:
    print("\n" + "#" * 74)
    print(f"# SEED {sd}")
    print("#" * 74)
    cfg = copy.deepcopy(XCFG)
    cfg["seed"] = sd

    e_pg = train_explainer(TRAIN_S, mode="pg_sufficiency", cfg=cfg, verbose=False)
    e_x = train_explainer(TRAIN_S, mode="xstrgnn", cfg=cfg, verbose=False)
    print("  explainers trained")

    methods = [
        ("Random",               attr_random),
        ("Gradient x input",     attr_gradient),
        ("GNNExplainer (inst.)", lambda b: attr_gnnexplainer(b, cfg)),
        ("PGExplainer (amort.)", lambda b: attr_amortised(e_pg, b)),
        ("X-StrGNN (ours)",      lambda b: attr_amortised(e_x, b)),
    ]
    for nm, fn in methods:
        r = evaluate_explainer(nm, fn, TEST_S, cfg=cfg)
        r["temporal"] = (r["tf_top"] / r["tf_rand"]
                         if abs(r["tf_rand"]) > 1e-9 else float("nan"))
        RUNS.setdefault(nm, []).append(r)
        print(f"  {nm:<22} Charact {r['charact']:.4f} | temporal {r['temporal']:.2f}")
    RUNS.setdefault("_sanity", []).append(
        sanity_model_randomisation(e_x, TEST_S, cfg=cfg))

# -----------------------------------------------------------------------------
FIELDS = [("nFid+", "fid_plus_n"), ("nFid-", "fid_minus_n"),
          ("Charact", "charact"), ("Temporal", "temporal"),
          ("Stab", "stability"), ("ms/exp", "ms_per_expl")]
ORDER = ["Random", "Gradient x input", "GNNExplainer (inst.)",
         "PGExplainer (amort.)", "X-StrGNN (ours)"]

print("\n" + "=" * 96)
print(f"RESULTS OVER {len(SEEDS)} SEEDS  (mean +/- sd; detector frozen throughout)")
print("=" * 96)
hdr = f"{'explainer':<22}" + "".join(f"{n:>15}" for n, _ in FIELDS)
print(hdr); print("-" * len(hdr))
for nm in ORDER:
    row = f"{nm:<22}"
    for _, key in FIELDS:
        v = np.array([r[key] for r in RUNS[nm]], float)
        row += f"{np.nanmean(v):>9.3f}+-{np.nanstd(v):<4.3f}"
    print(row)

x = np.array([r["charact"] for r in RUNS["X-StrGNN (ours)"]])
p = np.array([r["charact"] for r in RUNS["PGExplainer (amort.)"]])
g = np.array([r["charact"] for r in RUNS["GNNExplainer (inst.)"]])
xt = np.array([r["temporal"] for r in RUNS["X-StrGNN (ours)"]])
pt = np.array([r["temporal"] for r in RUNS["PGExplainer (amort.)"]])
sa = np.array(RUNS["_sanity"], float)

print("\n" + "=" * 96)
print("CONTRIBUTION ANALYSIS")
print("=" * 96)
print(f"Charact  X-StrGNN vs control  : {np.mean(x-p):+.4f} +- {np.std(x-p):.4f}")
print(f"Charact  X-StrGNN vs per-inst.: {np.mean(x-g):+.4f} +- {np.std(x-g):.4f}")
print(f"Temporal X-StrGNN vs control  : {np.mean(xt-pt):+.4f} +- {np.std(xt-pt):.4f}")
print(f"sanity rho (model randomisation): {np.nanmean(sa):.3f} +- {np.nanstd(sa):.3f}")
print()
print("A difference smaller than roughly twice its own sd is not separable at")
print("this sample size and should be reported as a tie, not as a win.")

with open(os.path.join(WORK, "seed_variance.pkl"), "wb") as f:
    pickle.dump(dict(runs=RUNS, seeds=SEEDS), f)
print(f"\nsaved -> {WORK}/seed_variance.pkl")
