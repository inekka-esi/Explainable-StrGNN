# Patches to the released StrGNN implementation

Reference: https://github.com/KnowledgeDiscovery/StrGNN
Paper: Cai et al., *Structural Temporal Graph Neural Networks for Anomaly Detection
in Dynamic Graphs*, CIKM 2021 (arXiv:2005.07427).

Every defect below was found by executing the official artifact under a single
protocol, and each one changes reported numbers. All are applied in
`cells/cell2_strgnn_detector.py`. They are recorded in a spirit of repair: the
implementation is public, which is what made this audit possible at all.

---

## P1 — evaluation reports metrics on the normal class

`pytorch_DGCNN/main.py` computes average precision and F1 with real edges as the
positive class. On these prevalences that inverts the meaning of the headline
number: the released convention reports AP ≈ 0.99 where the anomaly-class AP is
roughly 0.5.

**Applied:** anomaly is the positive class for AUC, AP and P@100 throughout.

## P2 — target link retained in the enclosing subgraph

`detection/Main.py` contains the test-edge masking line commented out. The target
link is then visible inside its own enclosing subgraph, so the model can read off
the answer it is being asked to predict.

**Applied:** the target link is zeroed in every snapshot of the window after node
labelling and before message passing. Labelling is computed before removal, which
matches the SEAL double-radius construction where the direct link does not enter
the distance computation anyway.

## P3 — train and test use different temporal windows

The released configuration trains with `w=1` and evaluates with `w=5`. The GRU
therefore sees a sequence length at test time it was never trained on.

**Applied:** one window `w` shared by training and evaluation.

## P4 — the GRU time axis is scrambled

**This is the defect that matters most for any temporal claim.**

In `pytorch_DGCNN/DGCNN_embedding.py`, the output of `conv1d_params3` has shape
`(B, channels, window)` — channel-major. The code then calls

```python
conv1d_res = conv1d_res.view(batch_size, window, -1)
```

A `view` on a channel-major contiguous tensor does not exchange the axes; it
reinterprets the underlying buffer, interleaving channels into the time axis. GRU
step `t` therefore does not correspond to snapshot `t`.

Because `conv1d_params3` uses `kernel = stride = seg`, its output position `t`
aggregates exactly the columns belonging to snapshot `t`, so the axis is
recoverable — the operation required is a transpose, not a reshape.

**Applied:** `z.transpose(1, 2)` in place of `view`. Any temporal attribution
result obtained without this patch is measuring a scrambled axis.

## P5 — device hardcoded to CUDA

`.cuda()` calls prevent CPU execution.

**Applied:** a single resolved `DEVICE`.

---

## Implementation note, not a defect

The compiled `pytorch_DGCNN` C++ extension (`lib/`, built via Makefile) is not
used. Sparse batch preparation is reimplemented in pure PyTorch with
`index_add_`, which removes the build step and makes the pipeline runnable in a
notebook. This changes performance characteristics, not semantics.

One consequence is worth recording: `index_add_` is **non-deterministic on CUDA**.
Two runs with an identical seed and an identical frozen checkpoint can differ in
the third decimal of a fidelity metric. This is why `cells/cell4_seed_variance.py`
exists and why the reported tables are mean ± sd rather than single runs.
