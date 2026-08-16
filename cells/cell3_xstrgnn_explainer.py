# =============================================================================
# CELL 3 / 3  —  X-StrGNN
# Post-hoc spatio-temporal explanation layer for StrGNN. Run CELL 1 and CELL 2
# first, in the same session (this cell reuses the StrGNN class and collate).
#
# StrGNN scores a target edge from w enclosing subgraphs, so an explanation must
# name both a structure and a time. X-StrGNN learns two masks over the FROZEN
# detector:
#
#   structural  m_e in [0,1] on the messages of every enclosing-subgraph edge,
#               parameterised as in PGExplainer (Luo et al., NeurIPS 2020): one
#               MLP shared across all instances maps [z_i; z_j; z_u; z_v; pos(t)]
#               to omega_e, relaxed by the binary concrete distribution
#               m = sigma((omega + log u - log(1-u)) / tau).
#
#   temporal    g_t in [0,1] on the w inputs of the GRU, from a second head over
#               the snapshot representation, with a smoothness penalty so the
#               gate reads as an evolution rather than isolated spikes.
#
# Two design points determine whether this learns at all.
#
#   1. Masks stay SOFT during training. A straight-through hard top-k pass hands
#      every edge the same identity gradient, so the sufficiency term (raise all
#      masks) and the counterfactual term (lower all masks) cancel to near-zero
#      net signal and the loss goes flat. Graded per-edge gradients are what
#      produce a ranking. The hard top-p cut is applied only at evaluation.
#
#   2. The temporal gate trains through a SOFTMAX over the window, not a one-hot
#      ablation. Softmax weights sum to one, so raising one snapshot lowers the
#      others and the snapshots compete for mass. A one-hot ablation raises every
#      gate uniformly and yields no ranking.
#
# Objective: a sufficiency term (the retained subgraph reproduces the
# prediction, which is what Fid- measures) plus SEPARATE counterfactual terms
# for structure and for time (discarding the explanation destroys the
# prediction, which is what Fid+ and the temporal ratio measure). They are kept
# separate so neither head satisfies the hinge on the other's behalf.
#
# Baselines: Random, Gradient x input, per-instance GNNExplainer, and an
# amortised sufficiency-only PGExplainer. The last is the control that isolates
# what the counterfactual and temporal terms add over amortisation alone.
#
# cudnn note: the GRU backward pass raises "cudnn RNN backward can only be
# called in training mode" in eval mode. Switching the detector to train() would
# enable dropout and destroy exact parity, so cudnn is disabled around the
# backward pass instead and the detector stays in eval().
# =============================================================================

import os, math, time, pickle, random, contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics as skm
from scipy.stats import spearmanr

XCFG = dict(
    epochs       = 30,
    lr           = 3e-3,
    hidden       = 64,
    tau_start    = 5.0,
    tau_end      = 0.5,      # sharper than 1.0; keeps mask dispersion alive
    logit_bound  = 8.0,      # bounds mask logits; prevents concrete collapse
    a_density    = 2.0,      # pulls mean(m) toward dens_target (see below)
    base_size    = 5e-3,     # GNNExplainer's own L1, held fixed (see below)
    base_ent     = 1e-2,     # GNNExplainer's own entropy weight, held fixed
    a_time       = 1e-4,     # temporal gate size penalty
    b_entropy    = 1e-3,     # structural mask entropy penalty
    c_smooth     = 5e-3,     # temporal smoothness penalty
    lam_cf       = 1.0,      # structural counterfactual weight
    lam_tcf      = 3.0,      # temporal counterfactual weight
    cf_margin    = 2.0,      # hinge on the complement-mask NLL
    batch_size   = 32,
    keep_ratio   = 0.20,     # p, the explanation budget
    dens_target  = 0.20,     # density the soft mask is pulled toward
    n_eval       = 600,      # test targets used for the evaluation
    gnnexp_steps = 100,      # per-instance baseline optimisation steps
    seed         = 1,
)

WORK = "./xstrgnn"
with open(os.path.join(WORK, "data.pkl"), "rb") as f:
    D = pickle.load(f)
TRAIN_S, TEST_S = D["train"], D["test"]
FEAT_DIM, K_SORT, BCFG = D["feat_dim"], D["k"], D["cfg"]
WINDOW = BCFG["window"]

MODEL = StrGNN(FEAT_DIM, K_SORT, WINDOW, BCFG["latent_dim"], BCFG["hidden"],
               BCFG["num_class"], BCFG["dropout"], BCFG["gru_hidden"],
               fix_temporal_axis=BCFG["fix_temporal_axis"]).to(DEVICE)
MODEL.load_state_dict(torch.load(os.path.join(WORK, "strgnn.pt"),
                                 map_location=DEVICE))
MODEL.eval()
for p in MODEL.parameters():
    p.requires_grad_(False)
Z_DIM = MODEL.total_latent_dim
print(f"frozen StrGNN loaded | z_dim {Z_DIM} | k {K_SORT} | window {WINDOW}")


def safe_rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    r = spearmanr(a, b).correlation
    return r if np.isfinite(r) else np.nan


def no_cudnn():
    """Detector stays in eval(); cudnn RNN backward requires train mode.
    Also removes cudnn kernel nondeterminism from the parity check."""
    if torch.cuda.is_available():
        return torch.backends.cudnn.flags(enabled=False)
    return contextlib.nullcontext()


# -----------------------------------------------------------------------------
# Extended collate: undirected pair bookkeeping required by the mask
# -----------------------------------------------------------------------------
def collate_x(samples, idxs):
    b = collate(samples, idxs)
    pair_of_edge, pair_graph, pair_lo, pair_hi, pair_gdeg = [], [], [], [], []
    graph_u, graph_v, snap_pos, graph_npairs = [], [], [], []
    off, gi, pid = 0, 0, 0
    for i in idxs:
        for w, sg in enumerate(samples[i]["seq"]):
            ei = sg["edge_index"]
            gd = sg.get("gdeg", sg["degs"])
            book = {}
            for e in range(ei.shape[1]):
                a, c = int(ei[0, e]), int(ei[1, e])
                key = (min(a, c), max(a, c))
                if key not in book:
                    book[key] = pid
                    pair_graph.append(gi)
                    pair_lo.append(key[0] + off); pair_hi.append(key[1] + off)
                    pair_gdeg.append(float(gd[key[0]] + gd[key[1]]))
                    pid += 1
                pair_of_edge.append(book[key])
            graph_npairs.append(len(book))
            graph_u.append(off); graph_v.append(off + 1)
            snap_pos.append(w)
            off += sg["n"]; gi += 1
    to = lambda a, dt=torch.long: torch.tensor(a, dtype=dt, device=DEVICE)
    b.update(pair_of_edge=to(pair_of_edge), pair_graph=to(pair_graph),
             pair_lo=to(pair_lo), pair_hi=to(pair_hi),
             pair_gdeg=to(pair_gdeg, torch.float),
             graph_u=to(graph_u), graph_v=to(graph_v), snap_pos=to(snap_pos),
             graph_npairs=graph_npairs, n_pairs=pid)
    return b


def pair_to_edge(b, pair_vals):
    if b["n_edges"] == 0:
        return torch.ones(0, device=DEVICE)
    return pair_vals[b["pair_of_edge"]]


def topk_pair_mask(b, imp, p, keep=True):
    """Per enclosing subgraph, keep (or drop) the top-p fraction of pairs."""
    out = (torch.zeros(b["n_pairs"], device=DEVICE) if keep
           else torch.ones(b["n_pairs"], device=DEVICE))
    start = 0
    for npair in b["graph_npairs"]:
        if npair == 0:
            continue
        kk = max(1, int(math.ceil(p * npair)))
        idx = torch.topk(imp[start:start + npair], kk).indices + start
        out[idx] = 1.0 if keep else 0.0
        start += npair
    return out


# =============================================================================
# STEP 1 — pass-through identity:  X-StrGNN(m=1, g=1) == StrGNN
# =============================================================================
def identity_check(samples, bs=32):
    ys, s_base, s_x = [], [], []
    with torch.no_grad(), no_cudnn():
        for lo in range(0, len(samples), bs):
            idxs = list(range(lo, min(lo + bs, len(samples))))
            b = collate_x(samples, idxs)
            l0 = MODEL(b)
            l1 = MODEL(b,
                       edge_mask=torch.ones(max(b["n_edges"], 0), device=DEVICE),
                       time_gate=torch.ones(b["n_targets"], WINDOW, device=DEVICE))
            s_base.append(l0[:, 1].exp().cpu().numpy())
            s_x.append(l1[:, 1].exp().cpu().numpy())
            ys.append(b["y"].cpu().numpy())
    y = np.concatenate(ys); a = np.concatenate(s_base); c = np.concatenate(s_x)
    m = lambda s: (skm.roc_auc_score(y, s), skm.average_precision_score(y, s),
                   precision_at_k(y, s, 100))
    A, C = m(a), m(c)
    print("\n" + "=" * 74)
    print("STEP 1 — detection parity: StrGNN vs X-StrGNN with masks held at 1")
    print("=" * 74)
    print(f"{'metric':<12}{'StrGNN':>14}{'X-StrGNN':>14}{'Delta':>16}")
    for nm, x, z in zip(["AUC-ROC", "AP", "P@100"], A, C):
        print(f"{nm:<12}{x:>14.10f}{z:>14.10f}{z-x:>16.3e}")
    print(f"{'max |ds|':<12}{'':>14}{'':>14}{np.abs(a-c).max():>16.3e}")
    print("the explanation layer is multiplicative and inactive at m=g=1, so the")
    print("detector is unchanged; every number below explains this same model.")
    return dict(auc=A[0], ap=A[1], p100=A[2], dauc=C[0]-A[0], dap=C[1]-A[1],
                dp100=C[2]-A[2], dmax=float(np.abs(a-c).max()))


# =============================================================================
# STEP 2 — the explanation network
# =============================================================================
def concrete(omega, tau, noise=True):
    if noise:
        u = torch.rand_like(omega).clamp(1e-6, 1 - 1e-6)
        omega = omega + torch.log(u) - torch.log(1 - u)
    return torch.sigmoid(omega / tau)


class SpatioTemporalExplainer(nn.Module):
    def __init__(self, z_dim, gru_in, window, hidden=64, bound=8.0):
        super().__init__()
        self.window, self.bound = window, bound
        # Separate positional tables per head. A single shared table is indexed
        # by both edge_logits and time_logits, so any change to the structural
        # loss moves the temporal gate as a side effect and the two heads
        # compete for one set of parameters.
        self.pos_e = nn.Embedding(window, 8)
        self.pos_t = nn.Embedding(window, 8)
        self.edge_mlp = nn.Sequential(
            nn.Linear(4 * z_dim + 8, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1))
        self.time_mlp = nn.Sequential(
            nn.Linear(2 * gru_in + 8, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def edge_logits(self, b, Z):
        if b["n_pairs"] == 0:
            return torch.zeros(0, device=Z.device)
        zi = Z[b["pair_lo"]]; zj = Z[b["pair_hi"]]
        zu = Z[b["graph_u"]][b["pair_graph"]]
        zv = Z[b["graph_v"]][b["pair_graph"]]
        pe = self.pos_e(b["snap_pos"])[b["pair_graph"]]
        raw = self.edge_mlp(torch.cat([zi, zj, zu, zv, pe], 1)).squeeze(-1)
        return self.bound * torch.tanh(raw / self.bound)

    def time_logits(self, b, seq):
        B, W, Dm = seq.shape
        ctx = seq.mean(1, keepdim=True).expand(B, W, Dm)
        pe = self.pos_t(torch.arange(W, device=seq.device)).unsqueeze(0).expand(B, W, 8)
        raw = self.time_mlp(torch.cat([seq, ctx, pe], -1)).squeeze(-1)
        return self.bound * torch.tanh(raw / self.bound)


def _reg_baseline(m, g, cfg):
    """GNNExplainer's published regularisation: L1 on the mask plus an entropy
    term. Deliberately NOT the X-StrGNN regulariser. A baseline that shares a
    regulariser with the proposed method silently changes every time the method
    is tuned, which makes the comparison meaningless even though the table still
    prints a number for it."""
    ent = -(m * torch.log(m + 1e-8) + (1 - m) * torch.log(1 - m + 1e-8))
    return cfg["base_size"] * m.mean() + cfg["base_ent"] * ent.mean()


def _regularisers(m, g, cfg):
    # Density matching, not L1. The explainer is scored after a hard top-p cut,
    # but an L1 penalty this weak leaves the soft mask sitting near density 0.5,
    # so the ranking is learned in a regime where every edge is half-present and
    # then evaluated in one where 80% are absent outright. A random subset
    # survives that shift; a ranking tuned to the wrong density does not.
    l_size = (cfg["a_density"] * (m.mean() - cfg["dens_target"]) ** 2
              + cfg["a_time"] * g.mean())
    ent = -(m * torch.log(m + 1e-8) + (1 - m) * torch.log(1 - m + 1e-8))
    l_ent = cfg["b_entropy"] * ent.mean()
    l_sm = cfg["c_smooth"] * (g[:, 1:] - g[:, :-1]).abs().mean()
    return l_size, l_ent, l_sm


def train_explainer(samples, mode="xstrgnn", cfg=XCFG, verbose=True):
    """mode='xstrgnn'        sufficiency + structural CF + temporal CF
       mode='pg_sufficiency' amortised PGExplainer: sufficiency only, no
                             counterfactual, no temporal term. The control."""
    set_seed(cfg["seed"])
    exp = SpatioTemporalExplainer(Z_DIM, MODEL.gru_in, WINDOW,
                                  cfg["hidden"], cfg["logit_bound"]).to(DEVICE)
    opt = torch.optim.Adam(exp.parameters(), lr=cfg["lr"])
    order = list(range(len(samples)))
    if verbose:
        print("\n" + "=" * 74)
        print(f"STEP 2 — training the explanation network [{mode}] (detector frozen)")
        print("=" * 74)
    for ep in range(cfg["epochs"]):
        tau = cfg["tau_start"] * (cfg["tau_end"] / cfg["tau_start"]) ** (
            ep / max(1, cfg["epochs"] - 1))
        random.shuffle(order); exp.train()
        agg, nb = {}, 0
        for lo in range(0, len(order), cfg["batch_size"]):
            idxs = order[lo:lo + cfg["batch_size"]]
            if len(idxs) < 2:
                continue
            b = collate_x(samples, idxs)
            if b["n_pairs"] == 0:
                continue
            with torch.no_grad():
                l0, inte = MODEL(b, return_internals=True)
                yhat = l0.argmax(1)
            Z = inte["Z"].detach(); seq = inte["seq"].detach()

            with no_cudnn():
                m_soft = concrete(exp.edge_logits(b, Z), tau)
                tl = exp.time_logits(b, seq)
                g_soft = concrete(tl, tau)

                if mode == "pg_sufficiency":
                    logits = MODEL(b, edge_mask=pair_to_edge(b, m_soft),
                                   time_gate=g_soft)
                    l_size, l_ent, _ = _regularisers(m_soft, g_soft, cfg)
                    l_fid = F.nll_loss(logits, yhat)
                    loss = l_fid + l_size + l_ent
                    parts = dict(fid=float(l_fid), m_mu=float(m_soft.mean()))
                else:
                    # softmax makes snapshots compete; normalising by the max
                    # reproduces the evaluation probe, which zeroes the arg-max
                    # snapshot and leaves the rest at 1
                    a_t = torch.softmax(tl / tau, dim=1)
                    g_abl = 1.0 - a_t / (a_t.max(1, keepdim=True).values + 1e-8)

                    # each probe matches its evaluation counterpart: structural
                    # fidelity is measured with no time gate, temporal fidelity
                    # with no edge mask
                    logits_keep = MODEL(b, edge_mask=pair_to_edge(b, m_soft))
                    logits_drop = MODEL(b, edge_mask=pair_to_edge(b, 1.0 - m_soft))
                    logits_tabl = MODEL(b, time_gate=g_abl)

                    l_fid = F.nll_loss(logits_keep, yhat)
                    l_cf = cfg["lam_cf"] * F.relu(
                        cfg["cf_margin"] - F.nll_loss(logits_drop, yhat))
                    l_tcf = cfg["lam_tcf"] * F.relu(
                        cfg["cf_margin"] - F.nll_loss(logits_tabl, yhat))
                    l_size, l_ent, l_sm = _regularisers(m_soft, g_soft, cfg)
                    loss = l_fid + l_cf + l_tcf + l_size + l_ent + l_sm
                    parts = dict(fid=float(l_fid), cf=float(l_cf),
                                 tcf=float(l_tcf), g_sd=float(g_soft.std()),
                                 m_mu=float(m_soft.mean()))

                opt.zero_grad(); loss.backward(); opt.step()

            parts["m_sd"] = float(m_soft.std())
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
        if verbose:
            print(f"epoch {ep:02d} | tau {tau:.2f} | " +
                  " | ".join(f"{k} {v/max(nb,1):.4f}" for k, v in agg.items()))
    if verbose:
        print("m_sd / g_sd are the dispersions of the structural mask and the")
        print("temporal gate; near zero means that head collapsed to a constant.")
        print("cf / tcf flat across epochs means the corresponding gradient path")
        print("is dead, whatever the dispersion shows. m_mu should converge to")
        print(f"the evaluation budget ({cfg['keep_ratio']:.2f}); if it sits near 0.5")
        print("the mask is being ranked in a density the metrics never test.")
    exp.eval()
    return exp


# =============================================================================
# STEP 3 — attribution methods
# =============================================================================
def attr_amortised(exp, b):
    with torch.no_grad():
        _, inte = MODEL(b, return_internals=True)
        return (torch.sigmoid(exp.edge_logits(b, inte["Z"])),
                torch.sigmoid(exp.time_logits(b, inte["seq"])))


def attr_random(b):
    return (torch.rand(b["n_pairs"], device=DEVICE),
            torch.rand(b["n_targets"], WINDOW, device=DEVICE))


def attr_gradient(b):
    m = torch.ones(max(b["n_pairs"], 1), device=DEVICE, requires_grad=True)
    g = torch.ones(b["n_targets"], WINDOW, device=DEVICE, requires_grad=True)
    with no_cudnn():
        logits = MODEL(b, edge_mask=pair_to_edge(b, m), time_gate=g)
        with torch.no_grad():
            yhat = logits.argmax(1)
        s = logits.gather(1, yhat.view(-1, 1)).sum()
        gm, gg = torch.autograd.grad(s, [m, g], allow_unused=True)
    gm = torch.zeros_like(m) if gm is None else gm
    gg = torch.zeros_like(g) if gg is None else gg
    return gm.abs().detach()[:b["n_pairs"]], gg.abs().detach()


def attr_gnnexplainer(b, cfg=XCFG):
    """Per-instance mask optimisation (GNNExplainer), no parameter sharing."""
    with torch.no_grad():
        yhat = MODEL(b).argmax(1)
    wm = torch.zeros(max(b["n_pairs"], 1), device=DEVICE, requires_grad=True)
    wg = torch.zeros(b["n_targets"], WINDOW, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([wm, wg], lr=0.05)
    with no_cudnn():
        for _ in range(cfg["gnnexp_steps"]):
            m, g = torch.sigmoid(wm), torch.sigmoid(wg)
            logits = MODEL(b, edge_mask=pair_to_edge(b, m[:b["n_pairs"]]),
                           time_gate=g)
            loss = F.nll_loss(logits, yhat) + _reg_baseline(m, g, cfg)
            opt.zero_grad(); loss.backward(); opt.step()
    return torch.sigmoid(wm).detach()[:b["n_pairs"]], torch.sigmoid(wg).detach()


# =============================================================================
# STEP 4 — explainability metrics
# =============================================================================
def prob_of(b, yref, edge_mask=None, time_gate=None):
    with torch.no_grad():
        p = MODEL(b, edge_mask=edge_mask, time_gate=time_gate).exp()
    return p.gather(1, yref.view(-1, 1)).squeeze(1)


def evaluate_explainer(name, attr_fn, samples, cfg=XCFG):
    sub = samples[:cfg["n_eval"]]
    p = cfg["keep_ratio"]
    fidp, fidm, spars, kept, dropd, true_y = [], [], [], [], [], []
    tf_top, tf_rand, stab, deg_rho, frange = [], [], [], [], []
    t_spent, n_expl = 0.0, 0

    for lo in range(0, len(sub), cfg["batch_size"]):
        idxs = list(range(lo, min(lo + cfg["batch_size"], len(sub))))
        if len(idxs) < 2:
            continue
        b = collate_x(sub, idxs)
        if b["n_pairs"] == 0:
            continue
        with torch.no_grad():
            yhat = MODEL(b).argmax(1)
        p0 = prob_of(b, yhat)

        t0 = time.time()
        m, g = attr_fn(b)
        t_spent += time.time() - t0; n_expl += b["n_targets"]

        keep_m = topk_pair_mask(b, m, p, keep=True)
        drop_m = topk_pair_mask(b, m, p, keep=False)
        fidp.append((p0 - prob_of(b, yhat,
                                  edge_mask=pair_to_edge(b, drop_m))).cpu().numpy())
        fidm.append((p0 - prob_of(b, yhat,
                                  edge_mask=pair_to_edge(b, keep_m))).cpu().numpy())
        spars.append(1.0 - float(keep_m.mean()))

        # largest drop this detector can express: the normaliser
        frange.append((p0 - prob_of(
            b, yhat,
            edge_mask=torch.zeros(b["n_edges"], device=DEVICE),
            time_gate=torch.zeros(b["n_targets"], WINDOW, device=DEVICE))
        ).cpu().numpy())

        with torch.no_grad():
            kept.append(MODEL(b, edge_mask=pair_to_edge(b, keep_m))[:, 1]
                        .exp().cpu().numpy())
            dropd.append(MODEL(b, edge_mask=pair_to_edge(b, drop_m))[:, 1]
                         .exp().cpu().numpy())
        true_y.append(b["y"].cpu().numpy())

        gt = torch.ones_like(g); gt.scatter_(1, g.argmax(1, keepdim=True), 0.0)
        gr = torch.ones_like(g)
        gr.scatter_(1, torch.randint(0, WINDOW, (g.size(0), 1), device=DEVICE), 0.0)
        tf_top.append((p0 - prob_of(b, yhat, time_gate=gt)).cpu().numpy())
        tf_rand.append((p0 - prob_of(b, yhat, time_gate=gr)).cpu().numpy())

        xs = b["x"].clone()
        b["x"] = xs + 0.05 * torch.randn_like(xs)
        m2, _ = attr_fn(b)
        b["x"] = xs
        s0, aa, cc = 0, m.cpu().numpy(), m2.cpu().numpy()
        for npair in b["graph_npairs"]:
            if npair >= 4:
                r = safe_rho(aa[s0:s0 + npair], cc[s0:s0 + npair])
                if np.isfinite(r):
                    stab.append(r)
            s0 += npair

        r = safe_rho(m.cpu().numpy(), b["pair_gdeg"].cpu().numpy())
        if np.isfinite(r):
            deg_rho.append(r)

    fidp = np.concatenate(fidp); fidm = np.concatenate(fidm)
    rng = float(np.abs(np.concatenate(frange)).mean())
    y = np.concatenate(true_y)
    fp, fm = float(fidp.mean()), float(fidm.mean())
    fpn, fmn = fp / max(rng, 1e-12), fm / max(rng, 1e-12)
    suff = 1.0 - min(max(fmn, 0.0), 1.0)
    charact = 0.0 if (fpn <= 0 or suff <= 0) else 2 * fpn * suff / (fpn + suff)
    return dict(
        name=name, fid_plus=fp, fid_minus=fm, fid_range=rng,
        fid_plus_n=fpn, fid_minus_n=fmn, sparsity=float(np.mean(spars)),
        charact=charact,
        auc_keep=float(skm.roc_auc_score(y, np.concatenate(kept))),
        auc_drop=float(skm.roc_auc_score(y, np.concatenate(dropd))),
        tf_top=float(np.concatenate(tf_top).mean()),
        tf_rand=float(np.concatenate(tf_rand).mean()),
        stability=float(np.mean(stab)) if stab else float("nan"),
        deg_rho=float(np.mean(deg_rho)) if deg_rho else float("nan"),
        ms_per_expl=1000.0 * t_spent / max(n_expl, 1),
    )


def sanity_model_randomisation(exp, samples, cfg=XCFG, n=200):
    """Adebayo-style check: explanations must depend on the trained weights."""
    trained = {k: v.clone() for k, v in MODEL.state_dict().items()}
    rows = []
    for lo in range(0, min(n, len(samples)), cfg["batch_size"]):
        idxs = list(range(lo, min(lo + cfg["batch_size"], min(n, len(samples)))))
        if len(idxs) < 2:
            continue
        b = collate_x(samples, idxs)
        if b["n_pairs"] < 4:
            continue
        m_a, _ = attr_amortised(exp, b)
        rnd = StrGNN(FEAT_DIM, K_SORT, WINDOW, BCFG["latent_dim"], BCFG["hidden"],
                     BCFG["num_class"], BCFG["dropout"], BCFG["gru_hidden"],
                     fix_temporal_axis=BCFG["fix_temporal_axis"]).to(DEVICE).eval()
        MODEL.load_state_dict(rnd.state_dict())
        m_b, _ = attr_amortised(exp, b)
        MODEL.load_state_dict(trained)
        r = safe_rho(m_a.cpu().numpy(), m_b.cpu().numpy())
        if np.isfinite(r):
            rows.append(r)
    return float(np.mean(rows)) if rows else float("nan")


# =============================================================================
# STEP 5 — run
# =============================================================================
ident = identity_check(TEST_S)
exp_pg = train_explainer(TRAIN_S, mode="pg_sufficiency")
exp = train_explainer(TRAIN_S, mode="xstrgnn")

print("\n" + "=" * 74)
print("STEP 3 — explainability evaluation on the test partition")
print(f"explanation budget p = {XCFG['keep_ratio']:.0%} of enclosing-subgraph edges")
print("=" * 74)

METHODS = [
    ("Random",               attr_random),
    ("Gradient x input",     attr_gradient),
    ("GNNExplainer (inst.)", attr_gnnexplainer),
    ("PGExplainer (amort.)", lambda b: attr_amortised(exp_pg, b)),
    ("X-StrGNN (ours)",      lambda b: attr_amortised(exp, b)),
]
RES = []
for nm, fn in METHODS:
    t0 = time.time()
    r = evaluate_explainer(nm, fn, TEST_S)
    r["wall"] = time.time() - t0
    RES.append(r)
    print(f"  done {nm} ({r['wall']:.1f}s)")

print(f"\nmodel ablation range (max expressible drop) = {RES[0]['fid_range']:.6f}")
hdr = (f"\n{'explainer':<22}{'nFid+':>8}{'nFid-':>8}{'Spars':>8}{'Charact':>9}"
       f"{'AUC|keep':>10}{'AUC|drop':>10}{'Stab':>8}{'ms/exp':>9}")
print(hdr); print("-" * (len(hdr) - 1))
for r in RES:
    print(f"{r['name']:<22}{r['fid_plus_n']:>8.4f}{r['fid_minus_n']:>8.4f}"
          f"{r['sparsity']:>8.3f}{r['charact']:>9.4f}{r['auc_keep']:>10.4f}"
          f"{r['auc_drop']:>10.4f}{r['stability']:>8.3f}{r['ms_per_expl']:>9.2f}")
print("nFid+/nFid- are normalised by the ablation range above, so they compare")
print("across detectors with different confidence calibration.")
print("nFid+ HIGHER is better; nFid- LOWER is better; Charact = harmonic mean.")

print(f"\n{'explainer':<22}{'dP top snap':>14}{'dP rand snap':>14}{'ratio':>9}")
print("-" * 59)
for r in RES:
    ratio = r["tf_top"] / r["tf_rand"] if abs(r["tf_rand"]) > 1e-9 else float("nan")
    print(f"{r['name']:<22}{r['tf_top']:>14.4f}{r['tf_rand']:>14.4f}{ratio:>9.2f}")
print("temporal fidelity: drop from ablating the snapshot the explainer ranks")
print("first, against a uniformly chosen snapshot. ratio > 1 means the named")
print("snapshot carries more of the decision than an arbitrary one.")

RHO = sanity_model_randomisation(exp, TEST_S)
print(f"\nsanity (model randomisation): rho(trained, re-initialised) = {RHO:.3f}")
print("explanations that track the weights should decorrelate under this test.")

X = [r for r in RES if r["name"].startswith("X-StrGNN")][0]
P = [r for r in RES if r["name"].startswith("PGExplainer")][0]
print("\n" + "=" * 74)
print("SUMMARY")
print("=" * 74)
print(f"{'model':<18}{'AUC':>9}{'AP':>9}{'P@100':>9}{'nFid+':>9}{'nFid-':>9}{'Charact':>10}")
print("-" * 74)
print(f"{'StrGNN':<18}{ident['auc']:>9.4f}{ident['ap']:>9.4f}{ident['p100']:>9.3f}"
      f"{'n/a':>9}{'n/a':>9}{'n/a':>10}")
print(f"{'X-StrGNN':<18}{ident['auc']+ident['dauc']:>9.4f}"
      f"{ident['ap']+ident['dap']:>9.4f}{ident['p100']+ident['dp100']:>9.3f}"
      f"{X['fid_plus_n']:>9.4f}{X['fid_minus_n']:>9.4f}{X['charact']:>10.4f}")
print(f"{'AddGraph':<18}{'-':>9}{'-':>9}{'-':>9}{'n/a':>9}{'n/a':>9}{'n/a':>10}")
print("-" * 74)
print(f"detection delta StrGNN -> X-StrGNN : AUC {ident['dauc']:+.2e}  "
      f"AP {ident['dap']:+.2e}  P@100 {ident['dp100']:+.2e}")
print(f"degree-signature audit rho(mass, endpoint degree) = {X['deg_rho']:+.3f}")
print(f"X-StrGNN vs its own control (PGExplainer amort.): "
      f"dCharact {X['charact']-P['charact']:+.4f}, "
      f"dTemporal {(X['tf_top']/max(X['tf_rand'],1e-9))-(P['tf_top']/max(P['tf_rand'],1e-9)):+.2f}")
print("if both are near zero, the contribution is amortisation, not the extra")
print("terms, and the paper should be framed accordingly.")

with open(os.path.join(WORK, "xstrgnn_results.pkl"), "wb") as f:
    pickle.dump(dict(identity=ident, explainers=RES, sanity=RHO), f)
torch.save(exp.state_dict(), os.path.join(WORK, "xstrgnn_explainer.pt"))
torch.save(exp_pg.state_dict(), os.path.join(WORK, "pgexplainer_ablation.pt"))
print(f"\nsaved -> {WORK}/xstrgnn_explainer.pt , xstrgnn_results.pkl")
