# Results

All numbers produced by the cells in `cells/`, on UCI-Messages, with the frozen
StrGNN checkpoint from `cell2`. Explainer tables are mean ± sd over three seeds.

## Detector

| | AUC-ROC | AP | P@100 |
|---|---|---|---|
| StrGNN (best epoch 18 of 50) | 0.8750 | 0.5163 | 0.660 |

Test-set score spread: std 0.2350, range [0.0309, 0.9357]. Detector ablation
range (largest expressible probability drop): 0.326.

## Detection parity

```
metric              StrGNN      X-StrGNN           Delta
AUC-ROC       0.8749834853  0.8749834853       0.000e+00
AP            0.5163393989  0.5163393989       0.000e+00
P@100         0.6600000000  0.6600000000       0.000e+00
max |ds|                                       1.192e-07
```

The residual on `max |ds|` is float32 epsilon in `exp()`. Both masks are
multiplicative and inactive at 1, so the explanation layer cannot alter detection
by construction; the check confirms the implementation honours that.

## Explainers (3 seeds, mean ± sd)

| explainer | nFid+ ↑ | nFid− ↓ | Charact ↑ | Temporal ↑ | Stability ↑ | ms/expl |
|---|---|---|---|---|---|---|
| Random | 0.228 ± 0.003 | 0.820 ± 0.002 | 0.201 ± 0.001 | 0.973 ± 0.050 | 0.002 ± 0.004 | 0.00 |
| Gradient × input | 0.336 ± 0.001 | 0.656 ± 0.000 | 0.340 ± 0.000 | **1.842 ± 0.034** | 0.405 ± 0.002 | 1.79 ± 0.12 |
| GNNExplainer (per-inst.) | 0.337 ± 0.001 | **0.556 ± 0.002** | **0.383 ± 0.000** | 0.836 ± 0.032 | 0.538 ± 0.002 | 177.97 ± 2.15 |
| PGExplainer (amort.) | 0.221 ± 0.005 | 0.800 ± 0.026 | 0.209 ± 0.016 | 1.081 ± 0.427 | 0.863 ± 0.005 | 0.63 ± 0.02 |
| **X-StrGNN** | 0.317 ± 0.068 | 0.739 ± 0.084 | 0.286 ± 0.078 | 1.601 ± 0.364 | **0.913 ± 0.010** | 0.66 ± 0.02 |

### Per-seed characterisation and temporal ratio

| seed | Random | Gradient | GNNExplainer | PGExplainer | X-StrGNN |
|---|---|---|---|---|---|
| 1 | 0.2005 / 1.01 | 0.3405 / 1.81 | 0.3832 / 0.79 | 0.1976 / 1.58 | 0.2685 / 2.11 |
| 2 | 0.2023 / 0.90 | 0.3402 / 1.89 | 0.3838 / 0.86 | 0.1979 / 0.54 | 0.1999 / 1.33 |
| 3 | 0.2009 / 1.01 | 0.3394 / 1.83 | 0.3832 / 0.86 | 0.2317 / 1.12 | 0.3885 / 1.36 |

The per-seed spread of the two amortised methods, against three-decimal stability
in the non-learned baselines, is finding E3.

## Contribution analysis

Differences smaller than roughly twice their own sd are ties.

| comparison | mean ± sd | ratio | verdict |
|---|---|---|---|
| Charact, X-StrGNN vs control | +0.0766 ± 0.0633 | 1.21 | tie |
| Charact, X-StrGNN vs per-instance | −0.0977 ± 0.0782 | 1.25 | not separable |
| Temporal, X-StrGNN vs control | +0.5193 ± 0.2263 | **2.29** | separable |

Model-randomisation sanity: rho(trained, re-initialised) = 0.272 ± 0.061.

## Reading

Separable: X-StrGNN's temporal attribution against its sufficiency-only control;
its stability (0.913 vs 0.538 for per-instance) and its cost (271× cheaper).

Not separable: X-StrGNN's structural fidelity against per-instance optimisation,
where the point estimate is in fact lower.

Against the field: gradient × input, which requires no training, leads temporal
fidelity outright and sits within 0.043 characterisation of the most expensive
method in the pool.

## Negative result

Rank correlation between X-StrGNN explanation mass and endpoint degree, across
five runs: +0.107, +0.096, +0.088, +0.073, +0.103. The hypothesis that StrGNN's
explanations concentrate on the degree signature exploited by parameter-free
heuristics on injected benchmarks is **not supported**.
