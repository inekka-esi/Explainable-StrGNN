# Protocol

## Corpus

UCI-Messages (KONECT edition): 1,899 nodes, 59,835 timestamped interactions,
obtained from the TADDY reference repository, which is the copy the published
DGAD benchmarks use. Columns are `src dst weight timestamp`; `%` lines are
comments.

`cells/cell1_data_preparation.py` **raises** if the parsed corpus is not exactly
59,835 interactions over 1,899 nodes. There is no synthetic fallback: a fallback
that silently substitutes a different graph produces a complete training run on
the wrong corpus, which is indistinguishable from success until the numbers are
already written up.

## Detector

| setting | value |
|---|---|
| snapshot size | 1,000 interactions (59 snapshots) |
| split | first 50% of the stream for training, temporal, no shuffling |
| window `w` | 5 |
| hop `h` | 1, with at most 20 nodes per hop |
| graph | accumulated |
| training negatives | context-dependent sampling (paper, Sec. 3.4) |
| test anomalies | 10% uniform injection into unobserved pairs (NetWalk protocol) |
| SortPooling `k` | 60th percentile of last-snapshot subgraph sizes, floor 10 |
| optimiser | Adam, lr 1e-4, batch 32, 50 epochs |
| model selection | best test AUC |

`max_nodes_per_hop = 20` bounds enclosing subgraph size for tractability; the
official code leaves it unbounded. Positive targets are capped per snapshot. Both
are deviations from the published configuration and are stated here rather than
buried in a config file.

## Explainers

All five methods are evaluated under one budget: the top **20%** of undirected
enclosing-subgraph pairs, selected per subgraph. Masks act on messages, not on the
degree normalisation, so an all-ones mask is the unmodified detector.

| method | provenance |
|---|---|
| Random | uniform importance |
| Gradient × input | ∂logit/∂mask at mask = 1 |
| GNNExplainer | per-instance mask optimisation, 100 steps, published L1 + entropy objective |
| PGExplainer (amortised) | shared MLP, published sufficiency-only objective — **the control** |
| X-StrGNN | shared MLP, sufficiency + structural counterfactual + temporal counterfactual |

The GNNExplainer baseline uses its **own** regularisation function, held fixed and
never shared with the proposed method. A baseline that shares a regulariser with
the method under test silently changes whenever the method is tuned, and the table
still prints a number for it.

## Metrics

- **nFid+** — probability drop when the top-20% is removed, normalised by the
  detector's total ablation range. Higher is better.
- **nFid−** — probability drop when only the top-20% is retained, same
  normalisation. Lower is better.
- **Characterisation** — harmonic mean of nFid+ and (1 − nFid−).
- **Temporal ratio** — drop from ablating the top-ranked snapshot divided by the
  drop from ablating a uniformly chosen one. 1.0 is the random floor.
- **Stability** — mean Spearman correlation of per-subgraph attribution rankings
  under a small input perturbation.
- **ms/expl** — wall-clock attribution time per target edge.

Normalising by the ablation range matters: raw fidelity is a probability
difference, so it scales with how confident the detector happens to be and is not
comparable across detectors or checkpoints.

## Seeds

Three seeds, frozen detector. The detector is not retrained between seeds, so the
dispersion measured is explainer variance alone — the quantity a comparison
between explainers depends on. No hyperparameter is changed between seeds.

Differences smaller than roughly twice their own standard deviation are reported
as ties.
