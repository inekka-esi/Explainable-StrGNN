# =============================================================================
# CELL 2 / 3  —  StrGNN  (Cai et al., CIKM 2021)
# Structural Temporal Graph Neural Networks for Anomaly Detection in Dynamic
# Graphs. Reproduction of the official pipeline without the compiled
# pytorch_DGCNN C++ extension, so it runs directly on Colab.
#
#   ESG   enclosing subgraph + double-radius node labeling      (CELL 1)
#   GSFE  graph convolution (3x32 + 1) -> SortPooling(k) -> 1D conv stack
#   TDN   GRU(256) over the w-snapshot window -> MLP -> 2-way logits
#
# Patches relative to the released code, each of which changes results:
#   P1  metrics computed on the ANOMALY class, not the normal class. The
#       released evaluation scores the normal class, which inflates AP beyond
#       recognition on these prevalences.
#   P2  target link removed from the enclosing subgraph in every snapshot
#   P3  train and test use the same temporal window w
#   P4  the (channel, time) tensor entering the GRU is TRANSPOSED, not
#       reshaped. The released code calls .view() on a channel-major tensor,
#       which interleaves channels into the time axis, so GRU step t does not
#       correspond to snapshot t. Any temporal claim requires this fix.
#   P5  device handling not hardcoded to CUDA
#
# The forward pass accepts two optional masks used by CELL 3. Both are
# multiplicative and inactive at 1.0, so an unmasked call is the unmodified
# detector.
# =============================================================================

import os, math, time, random, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics as skm

WORK = "./xstrgnn"
with open(os.path.join(WORK, "data.pkl"), "rb") as f:
    D = pickle.load(f)
TRAIN_S, TEST_S = D["train"], D["test"]
FEAT_DIM, K_SORT, CFG = D["feat_dim"], D["k"], D["cfg"]
WINDOW = CFG["window"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE} | feat_dim {FEAT_DIM} | k {K_SORT} | window {WINDOW}")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# -----------------------------------------------------------------------------
# Batching: flatten target-major / snapshot-minor, as the official loader does
# -----------------------------------------------------------------------------
def collate(samples, idxs, feat_dim=None, window=None, device=None):
    feat_dim = feat_dim or FEAT_DIM
    window = window or WINDOW
    device = device or DEVICE
    src, dst, tags, degs, sizes = [], [], [], [], []
    off = 0
    for i in idxs:
        for sg in samples[i]["seq"]:
            ei = sg["edge_index"]
            src.append(ei[0] + off); dst.append(ei[1] + off)
            tags.append(sg["tags"]); degs.append(sg["degs"] + 1.0)
            sizes.append(sg["n"]); off += sg["n"]
    src = np.concatenate(src) if src else np.zeros(0, np.int64)
    dst = np.concatenate(dst) if dst else np.zeros(0, np.int64)
    tags = np.concatenate(tags); degs = np.concatenate(degs)
    x = np.zeros((off, feat_dim), np.float32)
    x[np.arange(off), np.clip(tags, 0, feat_dim - 1)] = 1.0
    y = np.array([samples[i]["y"] for i in idxs], np.int64)
    return dict(
        x=torch.from_numpy(x).to(device),
        src=torch.from_numpy(src).to(device),
        dst=torch.from_numpy(dst).to(device),
        degs=torch.from_numpy(degs.astype(np.float32)).unsqueeze(1).to(device),
        sizes=sizes, y=torch.from_numpy(y).to(device),
        n_graphs=len(sizes), n_targets=len(idxs), window=window,
        n_edges=int(src.shape[0]),
    )


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def glorot_(t):
    if t.dim() == 2:
        fan_in, fan_out = t.size(0), t.size(1)
    elif t.dim() == 3:
        fan_in = t.size(1) * t.size(2); fan_out = t.size(0) * t.size(2)
    else:
        fan_in = fan_out = t.numel()
    t.data.uniform_(-math.sqrt(6.0 / (fan_in + fan_out)),
                    math.sqrt(6.0 / (fan_in + fan_out)))


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            m.bias.data.zero_(); glorot_(m.weight)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden, num_class, with_dropout=False):
        super().__init__()
        self.h1 = nn.Linear(in_dim, hidden)
        self.h2 = nn.Linear(hidden, num_class)
        self.with_dropout = with_dropout
        init_weights(self)

    def forward(self, x):
        h = F.relu(self.h1(x))
        if self.with_dropout:
            h = F.dropout(h, training=self.training)
        return F.log_softmax(self.h2(h), dim=1)


class StrGNN(nn.Module):
    """edge_mask : optional (E,) weights on subgraph messages, 1.0 == unmodified
       time_gate : optional (B, w) weights on the GRU inputs, 1.0 == unmodified"""

    def __init__(self, feat_dim, k, window, latent_dim, hidden, num_class,
                 dropout, gru_hidden=256, conv1d_channels=(16, 32),
                 conv1d_kws=(0, 5), fix_temporal_axis=True):
        super().__init__()
        self.latent_dim = list(latent_dim)
        self.total_latent_dim = int(sum(latent_dim))
        self.k = int(k); self.window = int(window)
        self.fix_temporal_axis = fix_temporal_axis
        kw0 = self.total_latent_dim

        self.conv_params = nn.ModuleList([nn.Linear(feat_dim, latent_dim[0])])
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i - 1], latent_dim[i]))

        self.conv1d_1 = nn.Conv1d(1, conv1d_channels[0], kw0, kw0)
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1],
                                  conv1d_kws[1], 1)
        dd = int((self.k - 2) / 2 + 1)
        self.seg = dd - conv1d_kws[1] + 1
        self.gru_in = self.seg * conv1d_channels[1]
        self.conv1d_3 = nn.Conv1d(conv1d_channels[1], self.gru_in,
                                  self.seg, self.seg)
        self.gru = nn.GRU(self.gru_in, gru_hidden, batch_first=True)
        self.mlp = MLPClassifier(gru_hidden, hidden, num_class, dropout)
        self.act = nn.ReLU()
        init_weights(self)

    def encode(self, b, edge_mask=None):
        h = b["x"]
        cats = []
        for lv in range(len(self.latent_dim)):
            if b["n_edges"] > 0:
                msg = h[b["src"]]
                if edge_mask is not None:
                    msg = msg * edge_mask.unsqueeze(1)
                agg = torch.zeros_like(h).index_add_(0, b["dst"], msg)
            else:
                agg = torch.zeros_like(h)
            lin = self.conv_params[lv](agg + h)        # (A + I) X
            h = torch.tanh(lin.div(b["degs"]))         # D^-1
            cats.append(h)
        return torch.cat(cats, 1)

    def sortpool(self, Z, sizes):
        out = Z.new_zeros(len(sizes), self.k, self.total_latent_dim)
        keep, ch, acc = [], Z[:, -1], 0
        for i, n in enumerate(sizes):
            kk = min(self.k, n)
            idx = ch[acc:acc + n].topk(kk).indices + acc
            out[i, :kk] = Z.index_select(0, idx)
            keep.append(idx); acc += n
        return out, keep

    def head(self, pooled, n_targets, time_gate=None, return_seq=False):
        z = pooled.view(-1, 1, self.k * self.total_latent_dim)
        z = self.act(self.conv1d_1(z))
        z = self.maxpool1d(z)
        z = self.act(self.conv1d_2(z))              # (BW, 32, seg)
        z = z.view(n_targets, z.size(1), -1)        # (B, 32, w*seg)
        z = self.act(self.conv1d_3(z))              # (B, gru_in, w)
        if self.fix_temporal_axis:
            seq = z.transpose(1, 2)                 # P4: (B, w, gru_in)
        else:
            seq = z.view(n_targets, self.window, -1)
        if time_gate is not None:
            seq = seq * time_gate.unsqueeze(-1)
        seq = seq.contiguous()
        _, hn = self.gru(seq)
        logits = self.mlp(self.act(hn.view(n_targets, -1)))
        return (logits, seq) if return_seq else logits

    def forward(self, b, edge_mask=None, time_gate=None, return_internals=False):
        Z = self.encode(b, edge_mask)
        pooled, keep = self.sortpool(Z, b["sizes"])
        out = self.head(pooled, b["n_targets"], time_gate,
                        return_seq=return_internals)
        if return_internals:
            logits, seq = out
            return logits, dict(Z=Z, keep=keep, seq=seq)
        return out


# -----------------------------------------------------------------------------
# Metrics  (P1: anomaly is the positive class)
# -----------------------------------------------------------------------------
def precision_at_k(y, s, k=100):
    k = min(k, len(s))
    return float(y[np.argsort(-s)[:k]].mean())


def evaluate(model, samples, batch_size=32):
    model.eval()
    ys, ss = [], []
    with torch.no_grad():
        for lo in range(0, len(samples), batch_size):
            idxs = list(range(lo, min(lo + batch_size, len(samples))))
            b = collate(samples, idxs)
            ss.append(model(b)[:, 1].exp().cpu().numpy())
            ys.append(b["y"].cpu().numpy())
    y, s = np.concatenate(ys), np.concatenate(ss)
    return dict(auc=float(skm.roc_auc_score(y, s)),
                ap=float(skm.average_precision_score(y, s)),
                p100=precision_at_k(y, s, 100), y=y, scores=s)


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
set_seed(CFG["seed"])
MODEL = StrGNN(FEAT_DIM, K_SORT, WINDOW, CFG["latent_dim"], CFG["hidden"],
               CFG["num_class"], CFG["dropout"], CFG["gru_hidden"],
               fix_temporal_axis=CFG["fix_temporal_axis"]).to(DEVICE)
OPT = torch.optim.Adam(MODEL.parameters(), lr=CFG["lr"])

order = list(range(len(TRAIN_S)))
best, t0 = None, time.time()
for ep in range(CFG["epochs"]):
    MODEL.train(); random.shuffle(order)
    tot, nb = 0.0, 0
    for lo in range(0, len(order), CFG["batch_size"]):
        idxs = order[lo:lo + CFG["batch_size"]]
        if len(idxs) < 2:
            continue
        b = collate(TRAIN_S, idxs)
        loss = F.nll_loss(MODEL(b), b["y"])
        OPT.zero_grad(); loss.backward(); OPT.step()
        tot += float(loss); nb += 1
    ev = evaluate(MODEL, TEST_S, CFG["batch_size"])
    print(f"epoch {ep:02d} | loss {tot/max(nb,1):.4f} | test AUC {ev['auc']:.4f} "
          f"AP {ev['ap']:.4f} P@100 {ev['p100']:.3f}")
    if best is None or ev["auc"] > best["auc"]:
        best = dict(ev, epoch=ep)
        torch.save(MODEL.state_dict(), os.path.join(WORK, "strgnn.pt"))

print(f"\nbest epoch {best['epoch']} | AUC {best['auc']:.4f} | "
      f"AP {best['ap']:.4f} | P@100 {best['p100']:.3f} | {time.time()-t0:.0f}s")

with open(os.path.join(WORK, "strgnn_baseline.pkl"), "wb") as f:
    pickle.dump(dict(auc=best["auc"], ap=best["ap"], p100=best["p100"],
                     epoch=best["epoch"], scores=best["scores"], y=best["y"]), f)
print(f"saved -> {WORK}/strgnn.pt , strgnn_baseline.pkl")

# The probability spread below governs whether CELL 3 can measure anything: an
# explanation metric is a change in predicted probability, so a detector whose
# outputs are all near 0.5 leaves no room for one.
print(f"score spread on test: std {best['scores'].std():.4f}, "
      f"range [{best['scores'].min():.4f}, {best['scores'].max():.4f}]")
