# =============================================================================
# CELL 1 / 3  —  Dataset preparation
#
# UCI-Messages (KONECT edition): 1,899 nodes, 59,835 timestamped interactions.
# Obtained from the TADDY reference repository, which is the same copy used by
# the published DGAD benchmarks, so the corpus matches the one the literature
# reports on.
#
# Produces, for every target edge, a window of w enclosing subgraphs with
# double-radius node labels, which is the input representation StrGNN consumes.
#
# There is deliberately NO synthetic fallback. A dataset that silently
# substitutes itself on a failed download produces a full training run on the
# wrong corpus, which is indistinguishable from success until the numbers are
# already in a paper. A missing dataset raises here.
# =============================================================================

import os, math, time, random, pickle, shutil, subprocess, warnings
import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path

warnings.filterwarnings("ignore")

CFG = dict(
    # ---- data -------------------------------------------------------------
    workdir           = "./xstrgnn",
    snapshot_size     = 1000,     # interactions per snapshot
    train_ratio       = 0.50,     # StrGNN: first 50% of the stream for training
    anomaly_ratio     = 0.10,     # injected anomaly rate in the test partition
    window            = 5,        # w
    hop               = 1,        # h
    max_nodes_per_hop = 20,       # bounds enclosing subgraph size
    accumulated       = True,     # accumulated graph vs time-evolving
    max_train_pairs   = 3000,     # cap on positive training targets
    max_test_pairs    = 1500,     # cap on positive test targets
    # ---- model / training (consumed by CELL 2) ----------------------------
    latent_dim        = [32, 32, 32, 1],
    sortpool_ratio    = 0.6,
    hidden            = 128,
    gru_hidden        = 256,
    dropout           = True,
    num_class         = 2,
    epochs            = 50,
    lr                = 1e-4,
    batch_size        = 32,
    fix_temporal_axis = True,     # GRU step t == snapshot t (see CELL 2)
    seed              = 1,
)

os.makedirs(CFG["workdir"], exist_ok=True)


def set_seed(s):
    random.seed(s); np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    except ImportError:
        pass


set_seed(CFG["seed"])

# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------
RAW = os.path.join(CFG["workdir"], "uci_raw")
MIRROR = "https://github.com/yuetan031/TADDY_pytorch.git"

if not os.path.exists(RAW):
    print("fetching UCI-Messages ...")
    tmp = "/tmp/_uci_src"
    shutil.rmtree(tmp, ignore_errors=True)
    subprocess.run(["git", "clone", "-q", "--depth", "1", MIRROR, tmp], check=True)
    shutil.copy(os.path.join(tmp, "data", "raw", "uci"), RAW)
    shutil.rmtree(tmp, ignore_errors=True)
print(f"raw file: {RAW}")


def load_uci_messages(path):
    """Columns are 'src dst weight timestamp'; lines beginning with % are
    comments. Returns a time-sorted edge stream with contiguous node ids."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    src, dst, ts = [], [], []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("%"):
                continue
            p = line.split()
            if len(p) < 4:
                continue
            u, v, t = int(p[0]), int(p[1]), int(p[3])
            if u == v:
                continue
            src.append(u); dst.append(v); ts.append(t)
    src, dst, ts = np.array(src), np.array(dst), np.array(ts)
    if len(src) == 0:
        raise ValueError("no interactions parsed - the raw file is empty")
    order = np.argsort(ts, kind="stable")
    src, dst, ts = src[order], dst[order], ts[order]
    uniq = np.unique(np.concatenate([src, dst]))
    remap = {int(o): i for i, o in enumerate(uniq)}
    src = np.array([remap[int(x)] for x in src])
    dst = np.array([remap[int(x)] for x in dst])
    return src, dst, ts, len(uniq)


SRC, DST, TS, N_NODES = load_uci_messages(RAW)
print(f"stream: {len(SRC)} interactions, {N_NODES} nodes")
if not (len(SRC) == 59835 and N_NODES == 1899):
    raise ValueError(
        f"expected 59835 interactions / 1899 nodes, got {len(SRC)} / {N_NODES}. "
        "The corpus is not canonical UCI-Messages; stop and check the download.")
print("corpus verified against the canonical 59,835 / 1,899 figures")


# -----------------------------------------------------------------------------
# Snapshots
# -----------------------------------------------------------------------------
def build_snapshots(src, dst, n_nodes, snapshot_size, accumulated=True):
    n = len(src)
    n_snap = max(1, n // snapshot_size)
    bounds = np.linspace(0, n, n_snap + 1).astype(int)
    graphs, bin_edges = [], []
    acc_r, acc_c = [], []
    for t in range(n_snap):
        lo, hi = bounds[t], bounds[t + 1]
        r, c = src[lo:hi], dst[lo:hi]
        bin_edges.append(np.stack([r, c], 1))
        if accumulated:
            acc_r.append(r); acc_c.append(c)
            rr = np.concatenate(acc_r); cc = np.concatenate(acc_c)
        else:
            rr, cc = r, c
        A = ssp.csr_matrix(
            (np.ones(len(rr) * 2),
             (np.concatenate([rr, cc]), np.concatenate([cc, rr]))),
            shape=(n_nodes, n_nodes))
        A.data[:] = 1.0
        A.setdiag(0); A.eliminate_zeros()
        graphs.append(A)
    return graphs, bin_edges


GRAPHS, BIN_EDGES = build_snapshots(SRC, DST, N_NODES,
                                    CFG["snapshot_size"], CFG["accumulated"])
print(f"snapshots: {len(GRAPHS)}")


# -----------------------------------------------------------------------------
# Targets
#   train negatives : context-dependent sampling (StrGNN paper, Sec. 3.4)
#   test anomalies  : uniform injection into unobserved pairs (NetWalk protocol)
# -----------------------------------------------------------------------------
def sample_targets(graphs, bin_edges, cfg):
    rng = np.random.RandomState(cfg["seed"])
    n_snap = len(graphs)
    n_nodes = graphs[0].shape[0]
    n_train = int(math.ceil(n_snap * cfg["train_ratio"]))
    w = cfg["window"]

    def dedup(pairs):
        s = {(int(min(a, b)), int(max(a, b))) for a, b in pairs}
        return np.array(sorted(s)) if s else np.zeros((0, 2), int)

    def context_negative(A, pairs, k):
        out, tries = [], 0
        while len(out) < k and tries < 40 * max(k, 1):
            tries += 1
            a, b = pairs[rng.randint(len(pairs))]
            if rng.rand() < 0.5:
                a = rng.randint(n_nodes)
            else:
                b = rng.randint(n_nodes)
            if a == b or A[a, b] != 0:
                continue
            out.append((int(min(a, b)), int(max(a, b))))
        return np.array(out) if out else np.zeros((0, 2), int)

    def uniform_injection(A, k):
        out, tries = [], 0
        while len(out) < k and tries < 40 * max(k, 1):
            tries += 1
            a, b = rng.randint(n_nodes), rng.randint(n_nodes)
            if a == b or A[a, b] != 0:
                continue
            out.append((int(min(a, b)), int(max(a, b))))
        return np.array(out) if out else np.zeros((0, 2), int)

    train, test = [], []
    per_tr = max(1, cfg["max_train_pairs"] // max(1, n_train - w))
    per_te = max(1, cfg["max_test_pairs"] // max(1, n_snap - n_train))

    for t in range(w - 1, n_snap):
        pos = dedup(bin_edges[t])
        if len(pos) == 0:
            continue
        is_train = t < n_train
        cap = per_tr if is_train else per_te
        if len(pos) > cap:
            pos = pos[rng.choice(len(pos), cap, replace=False)]
        if is_train:
            neg = context_negative(graphs[t], pos, len(pos))
            bucket = train
        else:
            k = max(1, int(round(len(pos) * cfg["anomaly_ratio"] /
                                 max(1e-9, 1 - cfg["anomaly_ratio"]))))
            neg = uniform_injection(graphs[t], k)
            bucket = test
        for a, b in pos:
            bucket.append((int(a), int(b), t, 0))    # 0 = normal
        for a, b in neg:
            bucket.append((int(a), int(b), t, 1))    # 1 = anomaly
    return train, test


TR_T, TE_T = sample_targets(GRAPHS, BIN_EDGES, CFG)
print(f"targets: train {len(TR_T)} (anom {sum(x[3] for x in TR_T)}), "
      f"test {len(TE_T)} (anom {sum(x[3] for x in TE_T)})")


# -----------------------------------------------------------------------------
# Enclosing subgraph extraction + double-radius node labeling
# -----------------------------------------------------------------------------
def drnl_node_label(sub):
    K = sub.shape[0]
    if K <= 2:
        return np.ones(K, dtype=int)
    sub_wo0 = sub[1:, 1:]
    keep = [0] + list(range(2, K))
    sub_wo1 = sub[keep, :][:, keep]
    d0 = shortest_path(sub_wo0, directed=False, unweighted=True)[1:, 0]
    d1 = shortest_path(sub_wo1, directed=False, unweighted=True)[1:, 0]
    d0 = np.nan_to_num(d0, posinf=1e9)
    d1 = np.nan_to_num(d1, posinf=1e9)
    d_over_2, d_mod_2 = np.divmod(d0 + d1, 2)
    lab = 1 + np.minimum(d0, d1) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    lab = np.concatenate([np.array([1.0, 1.0]), lab])
    lab[~np.isfinite(lab)] = 0
    lab[lab > 1e6] = 0
    lab[lab < 0] = 0
    return lab.astype(int)


def extract_subgraph(u, v, A, h, max_nodes_per_hop, rng):
    visited = {u, v}
    fringe = {u, v}
    nodes = {u, v}
    for _ in range(h):
        nxt = set()
        for x in fringe:
            nxt.update(A.indices[A.indptr[x]:A.indptr[x + 1]].tolist())
        nxt -= visited
        if max_nodes_per_hop is not None and len(nxt) > max_nodes_per_hop:
            nxt = set(rng.choice(sorted(nxt), max_nodes_per_hop,
                                 replace=False).tolist())
        if not nxt:
            break
        visited |= nxt
        nodes |= nxt
        fringe = nxt
    nodes.discard(u); nodes.discard(v)
    node_list = [u, v] + sorted(nodes)
    sub = A[node_list, :][:, node_list].tolil()
    labels = drnl_node_label(sub.tocsr())
    sub[0, 1] = 0; sub[1, 0] = 0          # target link removed (no leakage)
    sub = sub.tocsr(); sub.eliminate_zeros()
    coo = sub.tocoo()
    return dict(
        n=len(node_list),
        edge_index=np.stack([coo.row, coo.col], 0).astype(np.int64),
        degs=np.asarray(sub.sum(1)).ravel(),
        gdeg=np.asarray(A[node_list].sum(1)).ravel(),   # degree in full snapshot
        tags=labels,
        nodes=np.array(node_list),
    )


def build_samples(graphs, targets, cfg, tag=""):
    rng = np.random.RandomState(cfg["seed"])
    w, h, mnh = cfg["window"], cfg["hop"], cfg["max_nodes_per_hop"]
    samples, max_lab = [], 0
    t0 = time.time()
    for i, (u, v, t, y) in enumerate(targets):
        seq = []
        for g in range(t - w + 1, t + 1):
            sg = extract_subgraph(u, v, graphs[g], h, mnh, rng)
            max_lab = max(max_lab, int(sg["tags"].max()))
            seq.append(sg)
        samples.append(dict(seq=seq, y=int(y), target=(int(u), int(v)), t=int(t)))
        if (i + 1) % 2000 == 0:
            print(f"  [{tag}] {i+1}/{len(targets)}  ({time.time()-t0:.0f}s)")
    return samples, max_lab


print("extracting enclosing subgraphs ...")
TRAIN_S, m1 = build_samples(GRAPHS, TR_T, CFG, "train")
TEST_S,  m2 = build_samples(GRAPHS, TE_T, CFG, "test")

FEAT_DIM = max(m1, m2) + 1
last_sizes = sorted([s["seq"][-1]["n"] for s in TRAIN_S + TEST_S])
K_SORT = max(10, last_sizes[int(math.ceil(
    CFG["sortpool_ratio"] * len(last_sizes))) - 1])

print(f"max node label {FEAT_DIM-1} -> feat_dim {FEAT_DIM}")
print(f"SortPooling k = {K_SORT}")

with open(os.path.join(CFG["workdir"], "data.pkl"), "wb") as f:
    pickle.dump(dict(train=TRAIN_S, test=TEST_S, feat_dim=FEAT_DIM,
                     k=K_SORT, cfg=CFG), f)

print(f"\nsaved -> {CFG['workdir']}/data.pkl")
print(f"train samples {len(TRAIN_S)} | test samples {len(TEST_S)} | "
      f"test anomaly rate {np.mean([s['y'] for s in TEST_S]):.3f}")
