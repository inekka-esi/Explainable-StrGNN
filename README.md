# X-StrGNN: Amortised Spatio-Temporal Explanation for Dynamic-Graph Anomaly Detection

A post-hoc explanation layer for **StrGNN** (Cai et al., CIKM 2021), together with a
controlled evaluation of mask-based explainers on a dynamic-graph anomaly detector.

The headline result of this repository concerns **evaluation practice** as much as it
concerns the method. Under a single protocol with three seeds:

- a **parameter-free gradient baseline** matches or beats every trained explainer on
  both fidelity axes, at 1.8 ms per explanation;
- **per-instance GNNExplainer**, the most expensive method in the pool at 178 ms,
  scores *below the random floor* on temporal fidelity;
- **learned explainers show seed-to-seed dispersion larger than every between-method
  gap in the table**, while the non-learned baselines are deterministic to three
  decimals.

Single-seed explainability tables are therefore not interpretable for this class of
detector. This repository releases the protocol, the runners, and the seed variance so
that the claim can be checked rather than taken on trust.

---

## 1. What X-StrGNN is

StrGNN scores a target edge from `w` enclosing subgraphs, so an explanation must name
both a **structure** and a **time**. X-StrGNN learns two masks over the **frozen**
detector:

| mask | acts on | parameterisation |
|---|---|---|
| structural `m_e` | messages of every enclosing-subgraph edge | PGExplainer-style shared MLP over `[z_i; z_j; z_u; z_v; pos(t)]`, binary-concrete relaxation |
| temporal `g_t` | the `w` inputs of the GRU | second head over the snapshot representation, with a smoothness penalty |

Both masks are multiplicative and identically 1 in the unexplained pass, so X-StrGNN is
an **exact pass-through** of StrGNN. This is verified numerically, not asserted:

```
metric              StrGNN      X-StrGNN           Delta
AUC-ROC       0.8749834853  0.8749834853       0.000e+00
AP            0.5163393989  0.5163393989       0.000e+00
P@100         0.6600000000  0.6600000000       0.000e+00
```

The objective is a sufficiency term (the retained subgraph reproduces the prediction,
which is what Fid− measures) plus separate counterfactual terms for structure and for
time (discarding the explanation destroys the prediction, which is what Fid+ and the
temporal ratio measure).

---

## 2. Results

**Detector.** StrGNN on UCI-Messages, 59,835 interactions / 1,899 nodes, 80/20 temporal
split, 10% uniform injection into the test partition. Best epoch 18: **AUC 0.8750,
AP 0.5163, P@100 0.660**.

**Explainers.** Three seeds, frozen detector, 20% explanation budget, mean ± sd.

| explainer | nFid+ ↑ | nFid− ↓ | Charact ↑ | Temporal ↑ | Stability ↑ | ms/expl |
|---|---|---|---|---|---|---|
| Random | 0.228 ± 0.003 | 0.820 ± 0.002 | 0.201 ± 0.001 | 0.973 ± 0.050 | 0.002 ± 0.004 | 0.00 |
| Gradient × input | 0.336 ± 0.001 | 0.656 ± 0.000 | 0.340 ± 0.000 | **1.842 ± 0.034** | 0.405 ± 0.002 | 1.79 |
| GNNExplainer (per-instance) | 0.337 ± 0.001 | **0.556 ± 0.002** | **0.383 ± 0.000** | 0.836 ± 0.032 | 0.538 ± 0.002 | 177.97 |
| PGExplainer (amortised) | 0.221 ± 0.005 | 0.800 ± 0.026 | 0.209 ± 0.016 | 1.081 ± 0.427 | 0.863 ± 0.005 | 0.63 |
| **X-StrGNN (ours)** | 0.317 ± 0.068 | 0.739 ± 0.084 | 0.286 ± 0.078 | 1.601 ± 0.364 | **0.913 ± 0.010** | 0.66 |

`Temporal` is the drop from ablating the snapshot the explainer ranks first, divided by
the drop from ablating a uniformly chosen snapshot. A ratio of 1.0 is the random floor.
`nFid±` are normalised by the detector's own ablation range (0.326) so they do not
depend on how confident this particular detector happens to be.

### Contribution analysis

A difference smaller than roughly twice its own standard deviation is reported as a tie.

| comparison | mean ± sd | ratio | verdict |
|---|---|---|---|
| Charact, X-StrGNN vs its own control | +0.0766 ± 0.0633 | 1.21 | **tie** |
| Charact, X-StrGNN vs per-instance | −0.0977 ± 0.0782 | 1.25 | not separable |
| Temporal, X-StrGNN vs its own control | +0.5193 ± 0.2263 | **2.29** | **separable** |

The control is X-StrGNN's own explainer trained with the published sufficiency-only
PGExplainer objective, so it isolates what the counterfactual and temporal terms add
over amortisation alone.

**What this supports.** X-StrGNN's temporal attribution is separably better than its
sufficiency-only control, and its stability and cost advantages are unambiguous. Its
structural fidelity is **not** separable from per-instance optimisation, and the
direction of that difference is negative. We report it as such.

### Findings

**E1 — a parameter-free baseline is competitive with trained explainers.**
Gradient × input reaches characterisation 0.340 ± 0.000 against per-instance
GNNExplainer's 0.383 ± 0.000, at 1.8 ms versus 178 ms, and leads every method on
temporal fidelity (1.842). Explainability papers on dynamic-graph detectors should
report Δ against a gradient baseline; an absolute fidelity number is not interpretable
without one.

**E2 — the most expensive explainer is worse than chance on the temporal axis.**
Per-instance GNNExplainer scores 0.836 ± 0.032 temporal fidelity against a random floor
of 0.973 ± 0.050. A static explainer applied to a windowed detector does not identify
which snapshot carries the decision. Explainers for temporal detectors must be
evaluated on a temporal probe, not only on a structural one.

**E3 — learned explainers are less reproducible than the gaps being reported.**
X-StrGNN's characterisation varies ±0.078 across seeds and its temporal ratio ±0.364,
both larger than every between-method gap in the table. The non-learned baselines vary
by ±0.001. Run-to-run variation on identical seeds is also non-negligible, because
`index_add_` is non-deterministic on CUDA. Single-seed explainability tables for this
class of detector are not interpretable, and this repository reports mean ± sd over
three seeds for that reason.

**E4 — amortisation buys stability and cost, not fidelity.**
X-StrGNN reaches 0.913 ± 0.010 attribution stability at 0.66 ms, against 0.538 and
178 ms for per-instance optimisation: 271× cheaper and substantially more stable, which
is what makes explaining a full alarm list feasible rather than a hand-picked handful.
It does not reach per-instance fidelity.

**E5 — the model-randomisation sanity check behaves differently for amortised methods.**
`rho(trained, re-initialised) = 0.272 ± 0.061`. An amortised explainer retains its own
trained MLP when the detector is randomised, so it still emits a structured ranking;
per-instance methods optimise against the randomised model directly and score near
zero. This is a property of amortisation and should be interpreted accordingly rather
than as evidence that the explanation ignores the detector.

### Negative result

We tested whether StrGNN's explanation mass concentrates on the degree signature that a
parameter-free degree heuristic exploits on injected benchmarks. Across five runs the
rank correlation between explanation mass and endpoint degree was
+0.107, +0.096, +0.088, +0.073, +0.103 — no usable signal. We record this because the
hypothesis is natural and the absence of support for it is informative.

---

## 3. Reproducing

Four cells, run in order in one Colab session (T4 or CPU; CPU is roughly 3–6× slower).

| cell | does | time (T4) |
|---|---|---|
| `cells/cell1_data_preparation.py` | downloads UCI-Messages, builds snapshots, samples targets, extracts enclosing subgraphs | ~2 min |
| `cells/cell2_strgnn_detector.py` | defines and trains StrGNN, 50 epochs | ~9 min |
| `cells/cell3_xstrgnn_explainer.py` | parity check, trains both explainers, evaluates five attribution methods | ~5 min |
| `cells/cell4_seed_variance.py` | repeats cell 3 over three seeds, reports mean ± sd | ~20 min |

Cell 1 caches to `./xstrgnn/data.pkl`; cells 2–4 reuse it. Only cell 3 needs rerunning
when iterating on the explainer.

```bash
pip install -r requirements.txt
```

### Diagnostics to check

The cells print the quantities that determine whether the run is interpretable:

- `stream: 59835 interactions, 1899 nodes` — cell 1 raises if the corpus is not
  canonical. There is deliberately no synthetic fallback.
- `score spread on test` — an explanation metric is a change in predicted probability,
  so a detector whose outputs cluster near 0.5 leaves nothing to measure.
- `max |ds|` in the parity check — should be at float32 epsilon. A non-zero metric delta
  means dropout is active and the detector is not frozen.
- `m_sd`, `g_sd` — mask and gate dispersion. Near zero means that head collapsed to a
  constant and its ranking carries no signal.
- `m_mu` — realised mask density, which should track the evaluation budget. A soft mask
  sitting near 0.5 is ranked in a density the metrics never test.
- `cf`, `tcf` — flat across epochs means the corresponding gradient path is dead,
  whatever the dispersion shows.

---

## 4. Patches to the released StrGNN implementation

Executing the official pipeline under a single protocol surfaced defects that change
reported numbers. All are documented in [`docs/PATCHES.md`](docs/PATCHES.md) and applied
in `cells/cell2_strgnn_detector.py`. In summary:

| id | defect |
|---|---|
| P1 | released evaluation reports AP and F1 on the **normal** class, not the anomaly class |
| P2 | target link not removed from the enclosing subgraph (leakage) |
| P3 | train and test use different temporal windows |
| P4 | the `(channel, time)` tensor entering the GRU is reshaped rather than transposed, so **GRU step `t` does not correspond to snapshot `t`** |
| P5 | device handling hardcoded to CUDA |

P4 is the one that matters most here: any temporal attribution claim is meaningless
without it.

---

## 5. Layout

```
cells/     the four runnable cells, in order
docs/      PATCHES.md, PROTOCOL.md
results/   RESULTS.md, seed_variance summary
```

## 6. Citation

```bibtex
@misc{nekka2026xstrgnn,
  author = {Nekka, Iyad Assaad and Seba, Hamida and Hidouci, Khaled-Walid
            and Amrouche, Karima},
  title  = {X-StrGNN: Amortised Spatio-Temporal Explanation for Dynamic-Graph
            Anomaly Detection},
  year   = {2026},
  note   = {LCSI Laboratory, Higher National School of Computer Science (ESI), Algiers}
}
```

Built on StrGNN (Cai et al., CIKM 2021), PGExplainer (Luo et al., NeurIPS 2020), and
GNNExplainer (Ying et al., NeurIPS 2019). UCI-Messages from the KONECT collection.

## 7. Licence

MIT. See [LICENSE](LICENSE).
