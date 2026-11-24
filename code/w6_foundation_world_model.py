from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import chi2
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import w3_world_model as W
from common import CORPUS_PROC, PKG, PROTOCOL, SRC, log, update_results

OUT = W.OUT
CACHE = OUT / "cache"
SEED = W.WM_SEED + 10
ENC_CUTOFF, SNAP_FIRST, LAST = 2019, 2005, 2025
D_HID, D_Z = 128, 64
BLOCKS = {"language": slice(0, 32), "attention": slice(32, 48), "context": slice(48, 64)}
K_AFF, MASK_RATE, ENC_EPOCHS, HUBER = 10, 0.3, 40, 1.0
ROLL_K, LAM_PATH, P_EPOCHS = 3, 2.0, 20
H_FC = 3
DEV_TASK = (2016, 3)
TFM_ID = "google/timesfm-2.5-200m-pytorch"
torch.set_num_threads(12)


class Snapshots:

    def __init__(self, years, ent, M, pdoc, emb, basis=ENC_CUTOFF):
        E = len(ent)
        self.E = E
        up = years <= basis
        X = emb[up].astype(np.float32)
        mu = X.mean(0)
        w, U = np.linalg.eigh(np.cov((X - mu).T))
        idx = np.argsort(w)[::-1][:W.K_WHITE]
        Wm = (U[:, idx] / np.sqrt(np.maximum(w[idx], 1e-8))).astype(np.float32)
        Et = (emb.astype(np.float32) - mu) @ Wm
        Yd = sp.csr_matrix((np.ones(len(years), np.float32), (np.arange(len(years)), years - 2000)), shape=(len(years), 27))
        self.cnt = np.asarray((M.T @ Yd).todense())
        self.lc = np.log1p(self.cnt)
        anyc = self.cnt > 0
        self.first = np.where(anyc.any(1), anyc.argmax(1) + 2000, 2100)
        excl, types = ent.excluded.values, ent.type.values
        pyr = years[pdoc.doc.values]
        key = pdoc.a.values.astype(np.int64) * E + pdoc.b.values
        self.t = {}
        for t in range(SNAP_FIRST, LAST + 1):
            n_t = self.cnt[:, : t - 1999].sum(1)
            nodes = np.flatnonzero((n_t >= W.MIN_DOCS) & ~excl)
            nV = len(nodes)
            pos = np.full(E, -1, np.int64)
            pos[nodes] = np.arange(nV)
            wt = np.where(years <= t, np.exp(-(t - years) / W.TAU), 0.0).astype(np.float32)
            H = np.asarray(M.T @ (Et * wt[:, None]))[nodes]
            H /= np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
            u, c = np.unique(key[pyr <= t], return_counts=True)
            a, b = pos[u // E], pos[u % E]
            ok = (a >= 0) & (b >= 0)
            Aco = sp.coo_matrix((c[ok].astype(np.float32), (a[ok], b[ok])), shape=(nV, nV)).tocsr()
            Aco = (Aco + Aco.T).tocsr()
            S = H @ H.T
            np.fill_diagonal(S, -np.inf)
            nb = np.argpartition(-S, K_AFF, axis=1)[:, :K_AFF]
            Aaf = sp.csr_matrix((np.ones(nV * K_AFF, np.float32), (np.repeat(np.arange(nV), K_AFF), nb.ravel())), shape=(nV, nV))
            del S
            yi = t - 2000
            lag = np.stack([self.lc[nodes, yi - k] for k in range(5)], 1)
            n2 = self.cnt[nodes, : yi - 1].sum(1); n4 = self.cnt[nodes, : yi - 3].sum(1); nn = n_t[nodes]
            g = np.log((nn - n2 + 1) / (n2 - n4 + 1))
            xa = np.column_stack([lag, np.log1p(nn), g, np.minimum(t - self.first[nodes], 30) / 10.0])
            onehot = np.stack([(types[nodes] == ty).astype(np.float32) for ty in W.TYPES], 1)
            deg = np.asarray((Aco > 0).sum(1)).ravel(); wdeg = np.asarray(Aco.sum(1)).ravel()
            share = np.asarray(Aco @ onehot) / np.maximum(wdeg, 1e-9)[:, None]
            xc = np.column_stack([onehot, np.log1p(deg), np.log1p(wdeg), share])
            self.t[t] = {"nodes": nodes, "pos": pos, "xL": H.astype(np.float32), "xA": xa.astype(np.float32),
                         "xC": xc.astype(np.float32), "Aco": Aco, "Aaf": Aaf, "last": self.lc[nodes, yi].astype(np.float32)}
        X = np.vstack([np.hstack([s["xL"], s["xA"], s["xC"]]) for t, s in self.t.items() if t <= basis])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.dims = (self.t[basis]["xL"].shape[1], self.t[basis]["xA"].shape[1], self.t[basis]["xC"].shape[1])

    def x(self, t):
        s = self.t[t]
        return ((np.hstack([s["xL"], s["xA"], s["xC"]]) - self.mu) / self.sd).astype(np.float32)


def _rownorm(A):
    d = np.asarray(A.sum(1)).ravel()
    A = sp.diags(np.where(d > 0, 1.0 / np.maximum(d, 1e-12), 0.0)) @ A
    A = A.tocoo()
    return torch.sparse_coo_tensor(np.vstack([A.row, A.col]), A.data.astype(np.float32), A.shape).coalesce()


class SageLayer(torch.nn.Module):

    def __init__(self, n_rel=2):
        super().__init__()
        self.pool = torch.nn.ModuleList([torch.nn.Linear(D_HID, D_HID) for _ in range(n_rel)])
        self.nb = torch.nn.ModuleList([torch.nn.Linear(D_HID, D_HID, bias=False) for _ in range(n_rel)])
        self.me = torch.nn.Linear(D_HID, D_HID)
        self.norm = torch.nn.LayerNorm(D_HID)
        self.drop = torch.nn.Dropout(0.1)

    def forward(self, h, adjs):
        out = self.me(h)
        for r, A in enumerate(adjs):
            out = out + self.nb[r](torch.sparse.mm(A, torch.relu(self.pool[r](h))))
        return self.norm(h + self.drop(torch.nn.functional.gelu(out)))


class Encoder(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.inp = torch.nn.Linear(sum(dims), D_HID)
        self.mask = torch.nn.Parameter(torch.zeros(sum(dims)))
        self.layers = torch.nn.ModuleList([SageLayer(), SageLayer()])
        self.out = torch.nn.Linear(D_HID, D_Z)
        self.dec = torch.nn.ModuleList([torch.nn.Linear(b.stop - b.start, d) for b, d in zip(BLOCKS.values(), dims)])

    def forward(self, x, adjs, mask=None):
        if mask is not None:
            x = torch.where(mask[:, None], self.mask.expand_as(x), x)
        h = torch.nn.functional.gelu(self.inp(x))
        for L in self.layers:
            h = L(h, adjs)
        return self.out(h)

    def loss(self, z, x):
        o, tot, parts = 0, 0.0, []
        for (b, d), dec in zip(zip(BLOCKS.values(), self.dims), self.dec):
            l_ = torch.nn.functional.huber_loss(dec(z[:, b]), x[:, o:o + d], delta=HUBER)
            parts.append(l_); tot = tot + l_; o += d
        return tot, parts


def train_encoder(snaps, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc = Encoder(snaps.dims)
    years_tr = [t for t in snaps.t if t <= ENC_CUTOFF]
    data = {t: (torch.tensor(snaps.x(t)), [_rownorm(snaps.t[t]["Aco"]), _rownorm(snaps.t[t]["Aaf"])]) for t in years_tr}
    steps = ENC_EPOCHS * len(years_tr)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    hist = []
    for step in range(steps):
        if step % len(years_tr) == 0:
            order = rng.permutation(years_tr)
        t = order[step % len(years_tr)]
        x, adjs = data[t]
        m = torch.tensor(rng.random(x.shape[0]) < MASK_RATE)
        enc.train()
        opt.zero_grad()
        tot, parts = enc.loss(enc(x, adjs, m), x)
        tot.backward()
        opt.step(); sch.step()
        hist.append([float(tot)] + [float(p) for p in parts])
    enc.eval()
    with torch.no_grad():
        mk = []
        for t in years_tr:
            x, adjs = data[t]
            m = torch.tensor(rng.random(x.shape[0]) < MASK_RATE)
            z = enc(x, adjs, m)
            _, parts = enc.loss(z[m], x[m])
            mk.append([float(p) for p in parts])
    return enc, {"steps": steps, "final_loss": hist[-1], "first_loss": hist[0],
                 "masked_node_loss": dict(zip(BLOCKS, np.mean(mk, 0).tolist()))}


def embed_all(enc, snaps):
    Z = {}
    with torch.no_grad():
        for t in snaps.t:
            adjs = [_rownorm(snaps.t[t]["Aco"]), _rownorm(snaps.t[t]["Aaf"])]
            Z[t] = enc(torch.tensor(snaps.x(t)), adjs).numpy().astype(np.float32)
    return Z


def timesfm_forecasts(snaps, years_needed, tag):
    f = CACHE / f"timesfm_{tag}.npz"
    want = sorted(years_needed)
    if f.exists():
        z = np.load(f)
        if all(f"t{t}" in z.files for t in want):
            return {t: z[f"t{t}"] for t in want}
    import truststore
    truststore.inject_into_ssl()
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(TFM_ID)
    model.compile(timesfm.ForecastConfig(max_context=32, max_horizon=8, normalize_inputs=True, per_core_batch_size=512,
                                         force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
    series, owner = [], []
    for t in want:
        nodes = snaps.t[t]["nodes"]
        for k, e in enumerate(nodes):
            s0 = max(int(snaps.first[e]), 2000) - 2000
            series.append(snaps.lc[e, s0: t - 1999].astype(np.float32))
        owner.append((t, len(nodes)))
    pf, _ = model.forecast(horizon=H_FC, inputs=series)
    out, o = {}, 0
    for t, n in owner:
        out[t] = np.asarray(pf[o:o + n, :H_FC], np.float32); o += n
    CACHE.mkdir(exist_ok=True)
    np.savez(f, **{f"t{t}": v for t, v in out.items()})
    return out


class Adapter(torch.nn.Module):

    def __init__(self, p):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(p, 64), torch.nn.GELU(), torch.nn.Linear(64, 64), torch.nn.GELU(),
                                       torch.nn.Linear(64, H_FC))


def _adapter_in(snaps, Z, FC, t, use_z):
    s = snaps.t[t]
    parts = [FC[t], s["last"][:, None]] + ([Z[t]] if use_z else [])
    return np.hstack(parts).astype(np.float32)


def fit_adapter(snaps, Z, FC, t, use_z, seed):
    t0 = t - H_FC
    X = _adapter_in(snaps, Z, FC, t0, use_z)
    nodes = snaps.t[t0]["nodes"]
    Y = snaps.lc[nodes][:, t0 - 1999: t - 1999] - FC[t0]
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    net = Adapter(X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    Xt, Yt = torch.tensor((X - mu) / sd), torch.tensor(Y.astype(np.float32))
    va = np.random.default_rng(seed).random(len(Xt)) < 0.2
    tr = torch.tensor(~va); vt = torch.tensor(va)
    best, best_state, wait = np.inf, None, 0
    for _ in range(2000):
        net.train(); opt.zero_grad()
        l_ = torch.nn.functional.mse_loss(net.net(Xt[tr]), Yt[tr])
        l_.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            lv = float(torch.nn.functional.mse_loss(net.net(Xt[vt]), Yt[vt]))
        if lv < best - 1e-6:
            best, wait = lv, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= 30:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, mu, sd


def corrected(snaps, Z, FC, t, use_z, seed):
    net, mu, sd = fit_adapter(snaps, Z, FC, t, use_z, seed)
    with torch.no_grad():
        return FC[t] + net.net(torch.tensor((_adapter_in(snaps, Z, FC, t, use_z) - mu) / sd)).numpy()


class Predictor(torch.nn.Module):

    def __init__(self, da=2, hid=256):
        super().__init__()
        sn = torch.nn.utils.parametrizations.spectral_norm
        self.norm = torch.nn.LayerNorm(D_Z)
        self.lz = sn(torch.nn.Linear(D_Z, hid))
        self.la = torch.nn.Linear(da, hid)
        self.l2 = sn(torch.nn.Linear(hid, hid))
        self.out = sn(torch.nn.Linear(hid, D_Z * D_Z))

    def F(self, z, a):
        h = torch.nn.functional.gelu(self.lz(self.norm(z))) * torch.sigmoid(self.la(a))
        h = torch.nn.functional.gelu(self.l2(h))
        return self.out(h).view(-1, D_Z, D_Z) / D_Z ** 0.5

    def rollout(self, z, dz, A):
        zs, dzs = [], []
        for k in range(A.shape[1]):
            dz = torch.bmm(self.F(z, A[:, k]), dz.unsqueeze(2)).squeeze(2)
            z = z + dz
            zs.append(z); dzs.append(dz)
        return torch.stack(zs, 1), torch.stack(dzs, 1)


def _actions(lc_next, lc_prev):
    return np.stack([lc_next, lc_next - lc_prev], -1).astype(np.float32)


def _windows(snaps, Z, t_end):
    zs, dzs, tz, tdz, acts, msk = [], [], [], [], [], []
    for s in range(SNAP_FIRST + 1, t_end):
        S, Sp = snaps.t[s], snaps.t[s - 1]
        e = S["nodes"][Sp["pos"][S["nodes"]] >= 0]
        z0, zm = Z[s][S["pos"][e]], Z[s - 1][Sp["pos"][e]]
        tz_, tdz_, ac_, mk_ = [], [], [], []
        prev = z0
        for k in range(1, ROLL_K + 1):
            u = s + k
            if u <= t_end:
                zk = Z[u][snaps.t[u]["pos"][e]]
                tz_.append(zk); tdz_.append(zk - prev); prev = zk
                ac_.append(_actions(snaps.lc[e, u - 2000], snaps.lc[e, u - 2001])); mk_.append(np.ones(len(e), np.float32))
            else:
                tz_.append(np.zeros_like(z0)); tdz_.append(np.zeros_like(z0))
                ac_.append(np.zeros((len(e), 2), np.float32)); mk_.append(np.zeros(len(e), np.float32))
        zs.append(z0); dzs.append(z0 - zm); tz.append(np.stack(tz_, 1)); tdz.append(np.stack(tdz_, 1))
        acts.append(np.stack(ac_, 1)); msk.append(np.stack(mk_, 1))
    cat = lambda v: torch.tensor(np.concatenate(v))
    return cat(zs), cat(dzs), cat(tz), cat(tdz), cat(acts), cat(msk)


def train_predictor(snaps, Z, t_end, seed, zero_actions=False):
    torch.manual_seed(seed)
    z0, dz0, tz, tdz, A, mk = _windows(snaps, Z, t_end)
    if zero_actions:
        A = torch.zeros_like(A)
    net = Predictor()
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-3)
    n = len(z0)
    g = torch.Generator().manual_seed(seed)
    for _ in range(P_EPOCHS):
        net.train()
        perm = torch.randperm(n, generator=g)
        for k in range(0, n, 2048):
            b = perm[k:k + 2048]
            zh, dzh = net.rollout(z0[b], dz0[b], A[b])
            m = mk[b][..., None]
            l_roll = (((zh - tz[b]) ** 2) * m).sum() / (m.sum() * D_Z)
            l_path = (((dzh - tdz[b]) ** 2) * m).sum() / (m.sum() * D_Z)
            loss = l_roll + LAM_PATH * l_path
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    net.eval()
    return net, n


def previous_state(snaps, Z, t):
    e = snaps.t[t]["nodes"]
    pp = snaps.t[t - 1]["pos"][e]
    return np.where((pp >= 0)[:, None], Z[t - 1][np.maximum(pp, 0)], Z[t])


def rollout_from(net, snaps, Z, t, acts_lc, zero_actions=False):
    S = snaps.t[t]
    z0 = Z[t]
    zm = previous_state(snaps, Z, t)
    prev = np.column_stack([S["last"], acts_lc[:, :-1]])
    A = _actions(acts_lc, prev)
    if zero_actions:
        A = np.zeros_like(A)
    with torch.no_grad():
        zh, _ = net.rollout(torch.tensor(z0), torch.tensor(z0 - zm), torch.tensor(A))
    return zh.numpy()


GROUPS = ["language", "attention", "context", "rollout", "timesfm"]


class WMFeatures:

    def __init__(self, Z, R, FCc, snaps):
        self.Z, self.R, self.FCc, self.snaps = Z, R, FCc, snaps

    def block(self, t, d, ia, ib, drop=()):
        Z = self.Z[t]
        za, zb = Z[ia], Z[ib]
        cols, names = [], []
        for g, sl in BLOCKS.items():
            if g in drop:
                continue
            cols += [za[:, sl] * zb[:, sl], np.abs(za[:, sl] - zb[:, sl])]
            names += [f"{g}_had{k}" for k in range(sl.stop - sl.start)] + [f"{g}_abs{k}" for k in range(sl.stop - sl.start)]
            cols.append(_cos(za[:, sl], zb[:, sl])[:, None]); names.append(f"{g}_cos")
        if "rollout" not in drop:
            ra, rb = self.R[t][:, min(d, ROLL_K) - 1][ia], self.R[t][:, min(d, ROLL_K) - 1][ib]
            for g, sl in BLOCKS.items():
                if g not in drop:
                    cols.append(_cos(ra[:, sl], rb[:, sl])[:, None]); names.append(f"roll_{g}_cos")
            cols.append(_cos(ra, rb)[:, None]); names.append("roll_cos")
        if "timesfm" not in drop:
            k = max(1, min(d, H_FC))
            last = self.snaps.t[t]["last"]
            fa, fb = self.FCc[t][ia, :k].mean(1), self.FCc[t][ib, :k].mean(1)
            ga, gb = fa - last[ia], fb - last[ib]
            cols += [np.minimum(fa, fb)[:, None], np.maximum(fa, fb)[:, None], np.minimum(ga, gb)[:, None], np.maximum(ga, gb)[:, None]]
            names += ["fc_lo", "fc_hi", "fg_lo", "fg_hi"]
        return np.hstack(cols).astype(np.float32), names


def _cos(a, b):
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


VARIANTS = {"history": None, "history_embeddings": ("rollout", "timesfm"), "world_model": (),
            "history_language_state": "raw"}


def raw_language(feats, t, ia, ib, k=32):
    H = feats.snaps.t[t]["xL"]
    ha, hb = H[ia], H[ib]
    return np.hstack([ha[:, :k] * hb[:, :k], np.abs(ha[:, :k] - hb[:, :k]), (ha * hb).sum(1)[:, None]]).astype(np.float32)


def design(F, d, t, ia, ib, feats, variant, drop_extra=()):
    base = W.xmat(F, d, W.NOLANG)
    drop = VARIANTS[variant]
    if drop is None:
        return base
    if drop == "raw":
        return np.hstack([base, raw_language(feats, t, ia, ib)])
    blk, _ = feats.block(t, d, ia, ib, tuple(drop) + tuple(drop_extra))
    return np.hstack([base, blk])


class Head:

    def __init__(self, variant, drop_extra=()):
        self.variant, self.drop_extra = variant, tuple(drop_extra)

    def fit(self, train, d, feats):
        Xs, ys, ws = [], [], []
        for (F, t, ix, wn) in train:
            ia, ib = feats.snaps.t[t]["pos"][F.a.values[ix]], feats.snaps.t[t]["pos"][F.b.values[ix]]
            Xs.append(design(F.iloc[ix], d, t, ia, ib, feats, self.variant, self.drop_extra))
            ys.append(F[f"y{d}"].values[ix]); ws.append(wn)
        X, y, w = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)
        self.m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                                random_state=SEED).fit(X, y)
        lo = self.logit(X)
        from scipy.optimize import minimize_scalar

        def nll(c):
            z = lo + c
            return -(w * (y * -np.logaddexp(0, -z) + (1 - y) * -np.logaddexp(0, z))).sum()
        self.c = float(minimize_scalar(nll, bounds=(-30, 30), method="bounded").x)
        self.n_train, self.n_pos = int(len(y)), int(y.sum())
        return self

    def logit(self, X):
        p = np.clip(self.m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p))

    def score(self, F, d, t, ia, ib, feats):
        return self.logit(design(F, d, t, ia, ib, feats, self.variant, self.drop_extra))

    def prob(self, s):
        return 1 / (1 + np.exp(-(s + self.c)))


def build_frames(years, ent, M, pdoc, emb):
    CACHE.mkdir(exist_ok=True)
    rng = np.random.default_rng(W.WM_SEED)
    states, frames = {}, {}
    for T in range(W.TRAIN_FIRST, W.LAST_FULL):
        f = CACHE / f"frame_{T}.parquet"
        hz = [d for d in W.HORIZONS if T + d <= W.LAST_FULL]
        S = W.State(T, years, ent, M, pdoc, emb)
        if f.exists():
            F = pd.read_parquet(f)
            Zo = np.load(CACHE / f"frameZ_{T}.npy").astype(np.float32)
            F.attrs.update(json.loads((CACHE / f"frame_{T}.json").read_text()))
            _replay(S, rng, W.NEG_TEST if T in W.TEST_CUTOFFS else 400_000)
        else:
            F, Zo = W.frame(S, pdoc, years, rng, W.NEG_TEST if T in W.TEST_CUTOFFS else 400_000, horizons=hz)
            F.to_parquet(f, index=False)
            np.save(CACHE / f"frameZ_{T}.npy", Zo.astype(np.float16))
            (CACHE / f"frame_{T}.json").write_text(json.dumps({k: int(v) for k, v in F.attrs.items()}))
        states[T], frames[T] = S, (F, Zo)
        log("w6.frame", T=T, rows=len(F))
    return states, frames


def _replay(S, rng, want):
    nodes, E, known = S.nodes, S.E, S.known
    nV = len(nodes)
    samp, got = [], 0
    while got < want:
        m = int((want - got) * 1.3) + 1000
        i = rng.integers(0, nV, m); j = rng.integers(0, nV, m)
        ok = (i != j) & (S.syn[nodes[i]] != S.syn[nodes[j]])
        i, j = i[ok], j[ok]
        a = np.minimum(nodes[i], nodes[j]).astype(np.int64); b = np.maximum(nodes[i], nodes[j]).astype(np.int64)
        k = np.unique(a * E + b)
        k = k[~np.isin(k, known)]
        if samp:
            k = k[~np.isin(k, np.concatenate(samp))]
        k = rng.permutation(k)
        samp.append(k[: want - got]); got += len(samp[-1])


def case_control(frames, d, ts, rng):
    out = []
    for t in ts:
        F, _ = frames[t]
        y, w = F[f"y{d}"].values, F[f"w{d}"].values
        pos = np.flatnonzero(y == 1); neg = np.flatnonzero((y == 0) & (w > 0))
        if len(pos) == 0:
            continue
        take = rng.choice(neg, min(len(neg), W.NEG_TRAIN_RATIO * len(pos)), replace=False)
        ix = np.concatenate([pos, take])
        wn = np.where(y[ix] == 1, 1.0, (len(neg) / len(take)) * w[take[0]])
        out.append((F, t, ix, (wn / wn.mean()).astype(np.float32)))
    return out


def score_frame(heads, F, d, t, feats):
    ia, ib = feats.snaps.t[t]["pos"][F.a.values], feats.snaps.t[t]["pos"][F.b.values]
    s = {"resource_allocation": F.raw_ra.values.astype(float), "degree_product": F.raw_pa.values.astype(float)}
    for k, h in heads.items():
        s[k] = h.score(F, d, t, ia, ib, feats)
    return s


def calibration(y, w, p, bins=10):
    o = np.argsort(p, kind="stable")
    cw = np.cumsum(w[o]) / w.sum()
    grp = np.minimum((cw * bins).astype(int), bins - 1)
    rows = []
    for g in range(bins):
        ix = o[grp == g]
        W_, k = w[ix].sum(), int(y[ix].sum())
        lo = chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0
        hi = chi2.ppf(0.975, 2 * k + 2) / 2
        rows.append({"decile": g + 1, "predicted_per_10000": float(1e4 * (w[ix] * p[ix]).sum() / W_),
                     "observed_per_10000": float(1e4 * k / W_), "observed_lo": float(1e4 * lo / W_),
                     "observed_hi": float(1e4 * hi / W_), "positives": k})
    return rows


def exact_scores(S, t, d_heads, feats, old=None, chunk=400_000):
    I, J, out, raw = [], [], {k: [] for k in d_heads}, {"raw_pa": [], "raw_ra": []}
    if old:
        for k in old:
            out[k] = []
    for ii, jj in W.enumerate_candidates(S):
        for k in range(0, len(ii), chunk):
            a, b = ii[k:k + chunk], jj[k:k + chunk]
            F, Zo = W.pair_features(S, a, b)
            for name, (h, d) in d_heads.items():
                out[name].append(h.score(F, d, t, a, b, feats).astype(np.float32))
            if old:
                for name, (mod, d) in old.items():
                    out[name].append(mod.eta(W.xmat(F, d, W.BASE_X), Zo).astype(np.float32))
            for c in raw:
                raw[c].append(F[c].values.astype(np.float32))
            I.append(a.astype(np.int32)); J.append(b.astype(np.int32))
    return (np.concatenate(I), np.concatenate(J), {k: np.concatenate(v) for k, v in out.items()},
            {k: np.concatenate(v) for k, v in raw.items()})


def ci(v):
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="development task (2016, 3) only; not reported")
    args = ap.parse_args()
    DEV = args.dev
    KEY = "foundation_world_model" + ("_dev" if DEV else "")
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    years, ent, M, pdoc = W.prepare()
    emb = np.load(OUT / "emb_biomedbert.npy").astype(np.float32)
    states, frames = build_frames(years, ent, M, pdoc, emb)
    res = {"amendment": "A10, with the additions of A11", "seed": SEED, "dev": DEV,
           "sources": {"encoder": "PDFM (Agarwal et al., arXiv:2411.07207)", "dynamics": "SensorWM (Lee et al., arXiv:2611.08568)",
                       "forecasting_and_heads": "PDFM forecasting; Earth AI public health (Muslim et al., arXiv:2610.05699)",
                       "time_series_model": TFM_ID},
           "settings": {"encoder_cutoff": ENC_CUTOFF, "first_snapshot": SNAP_FIRST, "hidden": D_HID, "embedding": D_Z,
                        "blocks": {k: [v.start, v.stop] for k, v in BLOCKS.items()}, "affinity_k": K_AFF, "mask_rate": MASK_RATE,
                        "encoder_epochs": ENC_EPOCHS, "huber_delta": HUBER, "rollout_steps": ROLL_K, "lambda_path": LAM_PATH,
                        "predictor_epochs": P_EPOCHS, "forecast_horizon": H_FC, "gbdt": "300 iterations, learning rate 0.05, balanced"},
           "protocol_sha256_current": hashlib.sha256((PROTOCOL / "World_Model_Analysis_Plan.md").read_bytes()).hexdigest()}

    snaps = Snapshots(years, ent, M, pdoc, emb)
    for T in states:
        assert np.array_equal(states[T].nodes, snaps.t[T]["nodes"]), T
    enc, enc_info = train_encoder(snaps, SEED)
    ref = []
    for t in [t for t in snaps.t if t <= ENC_CUTOFF]:
        x = torch.tensor(snaps.x(t)); o = 0; row = []
        for dd in snaps.dims:
            row.append(float(torch.nn.functional.huber_loss(torch.zeros_like(x[:, o:o + dd]), x[:, o:o + dd], delta=HUBER))); o += dd
        ref.append(row)
    enc_info["mean_prediction_loss"] = dict(zip(BLOCKS, np.mean(ref, 0).tolist()))
    Z = embed_all(enc, snaps)
    res["encoder"] = {**enc_info, "nodes_by_snapshot": {int(t): int(len(s["nodes"])) for t, s in snaps.t.items()},
                      "input_dims": dict(zip(BLOCKS, snaps.dims)),
                      "edges_by_snapshot": {int(t): {"co_mention": int(s["Aco"].nnz // 2), "affinity": int(s["Aaf"].nnz)}
                                            for t, s in snaps.t.items()}}
    log("w6.encoder", **{k: round(v, 4) for k, v in enc_info["masked_node_loss"].items()}, secs=round(time.time() - t0))
    if not DEV:
        z25 = Z[LAST] - Z[LAST].mean(0)
        u_, s_, vt_ = np.linalg.svd(z25, full_matrices=False)
        nodes = snaps.t[LAST]["nodes"]
        pd.DataFrame({"pc1": z25 @ vt_[0], "pc2": z25 @ vt_[1], "type": ent.type.values[nodes],
                      "papers": snaps.cnt[nodes, :LAST - 1999].sum(1)}).to_csv(SRC / "fwm_embedding_2025.csv", index=False)
        res["encoder"]["pca_variance_share"] = (s_[:2] ** 2 / (s_ ** 2).sum()).tolist()

    need = list(range(W.TRAIN_FIRST - H_FC, LAST + 1))
    FC = timesfm_forecasts(snaps, need, "biomedbert")
    frame_years = sorted(set(range(W.TRAIN_FIRST, LAST + 1)))
    FCc = {t: corrected(snaps, Z, FC, t, True, SEED + t) for t in frame_years}
    log("w6.timesfm", secs=round(time.time() - t0))

    P, R, n_win = {}, {}, {}
    for t in frame_years:
        P[t], n_win[t] = train_predictor(snaps, Z, t, SEED + 100 + t)
        R[t] = rollout_from(P[t], snaps, Z, t, FCc[t])
        log("w6.predictor", t=t, windows=n_win[t], secs=round(time.time() - t0))
    log("w6.dynamics", secs=round(time.time() - t0))
    feats = WMFeatures(Z, R, FCc, snaps)

    T, d = W.PRIMARY if not DEV else DEV_TASK
    res["attention_forecast"] = attention_eval(snaps, Z, FC, FCc, T, rng)
    res["dynamics"] = dynamics_eval(snaps, Z, P, FCc, T, rng, DEV)
    log("w6.components", att=res["attention_forecast"]["mae"], dyn=res["dynamics"]["skill_vs_locf"], secs=round(time.time() - t0))

    tests = [DEV_TASK] if DEV else [(T_, d_) for T_ in W.TEST_CUTOFFS for d_ in W.HORIZONS if T_ + d_ <= W.LAST_FULL]
    grid, fitted, old_grid = [], {}, {(g["cutoff"], g["horizon"], g["method"]): g for g in json.loads(
        (PKG / "analysis" / "results.json").read_text(encoding="utf-8"))["world_model"]["grid"]}
    for T_, d_ in tests:
        ts = [t for t in sorted(frames) if t >= W.TRAIN_FIRST and t + d_ <= T_]
        train = case_control(frames, d_, ts, rng)
        heads = {v: Head(v).fit(train, d_, feats) for v in (VARIANTS if (T_, d_) in (W.PRIMARY, DEV_TASK) else ("history", "world_model"))}
        hz = W.fit_all([frames[t] for t in ts], d_, np.random.default_rng(SEED + 1), neural=True, glm=False, logistic=False)
        fitted[(T_, d_)] = (heads, train)
        F, Zo_ = frames[T_]
        s = score_frame(heads, F, d_, T_, feats)
        s["hazard_v1_neural"] = hz["neural"].eta(W.xmat(F, d_, W.BASE_X), Zo_)
        y, w = F[f"y{d_}"].values, F[f"w{d_}"].values
        m = w > 0
        o = old_grid.get((T_, d_, "resource_allocation"))
        if o is not None:
            assert o["n_pos"] == int(y[m].sum()) and o["n_neg_sampled"] == int((y[m] == 0).sum()), (T_, d_)
        bt = W.node_bootstrap2(F[m], d_, {k: s[k][m] for k in ("world_model", "history", "resource_allocation", "hazard_v1_neural")},
                               rng, 50 if DEV else 200,
                               contrasts=(("world_model", "resource_allocation"), ("world_model", "history"),
                                          ("world_model", "hazard_v1_neural")),
                               popularity=("world_model", "resource_allocation", "history", "hazard_v1_neural"),
                               pop_contrasts=(("world_model", "resource_allocation"), ("world_model", "history"),
                                              ("world_model", "hazard_v1_neural")))
        row = {"cutoff": T_, "horizon": d_, "label_window": f"{T_ + 1}-{T_ + d_}", "train_cutoffs": f"{min(ts)}-{max(ts)}",
               "prespecified_task": bool(T_ in W.PRESPECIFIED_CUTOFFS), "n_train": heads["world_model"].n_train}
        for k, v in s.items():
            e = W.evaluate(y, w, v, F.raw_pa.values)
            if k in bt["auroc_ci"]:
                e["auroc_ci"] = bt["auroc_ci"][k]
            if k in bt["popadj_ci"]:
                e["auroc_popularity_adjusted_ci"] = bt["popadj_ci"][k]
            grid.append({**row, "method": k, **e})
        g_old = old_grid.get((T_, d_, "neural_world_model"))
        if g_old is not None:
            grid.append({**row, "method": "hazard_v1_neural_previous_run", "auroc": g_old["auroc"],
                         "auroc_popularity_adjusted": g_old["auroc_popularity_adjusted"]})
        for k, v in bt["diff_ci"].items():
            grid.append({**row, "method": f"diff:{k}", "auroc_ci": v})
        for k, v in bt["popadj_diff_ci"].items():
            grid.append({**row, "method": f"diffpop:{k}", "auroc_ci": v})
        log("w6.task", T=T_, d=d_, **{k: round(roc_auc_score(y[m], v[m]), 4) for k, v in s.items()}, secs=round(time.time() - t0))
    G = pd.DataFrame(grid)
    G.to_csv(SRC / ("fwm_grid_dev.csv" if DEV else "fwm_grid.csv"), index=False)
    res["grid"] = G.to_dict(orient="records")
    update_results(KEY, res)

    T, d = W.PRIMARY if not DEV else DEV_TASK
    heads, train = fitted[(T, d)]
    F, Zo = frames[T]
    y, w = F[f"y{d}"].values, F[f"w{d}"].values
    m = w > 0
    s = score_frame(heads, F, d, T, feats)
    ts = [t for t in sorted(frames) if t >= W.TRAIN_FIRST and t + d <= T]
    old = W.fit_all([frames[t] for t in ts], d, np.random.default_rng(SEED + 1), logistic=False)
    so = W.scores(old, F, Zo, d)
    s["hazard_v1_neural"], s["hazard_v1"] = so["neural_world_model"], so["hazard_world_model"]
    for g in GROUPS:
        h = Head("world_model", (g,)).fit(train, d, feats)
        ia, ib = snaps.t[T]["pos"][F.a.values], snaps.t[T]["pos"][F.b.values]
        s[f"minus_{g}"] = h.score(F, d, T, ia, ib, feats)
    log("w6.ablations", secs=round(time.time() - t0))
    prim = {k: W.evaluate(y, w, v, F.raw_pa.values) for k, v in s.items()}
    meths = ["world_model", "history_embeddings", "history", "hazard_v1_neural", "hazard_v1", "resource_allocation", "degree_product"]
    cons = [("world_model", k) for k in meths[1:]] + [("history_embeddings", "history")]
    if "history_language_state" in s:
        cons += [("world_model", "history_language_state"), ("history_embeddings", "history_language_state"),
                 ("history_language_state", "history"), ("hazard_v1_neural", "history_language_state")]
    abl = [("world_model", f"minus_{g}") for g in GROUPS]
    draws = 50 if DEV else 500
    bt = W.node_bootstrap2(F[m], d, {k: v[m] for k, v in s.items()}, rng, draws, contrasts=cons + abl, popularity=list(s),
                           pop_contrasts=cons + abl)
    for k in prim:
        prim[k]["auroc_ci"] = bt["auroc_ci"][k]
        prim[k]["auroc_popularity_adjusted_ci"] = bt["popadj_ci"][k]
    res["primary"] = {"cutoff": T, "horizon": d, "label_window": f"{T + 1}-{T + d}", "metrics": prim,
                      "contrasts": bt["diff_ci"], "contrasts_popularity_adjusted": bt["popadj_diff_ci"], "bootstrap_draws": bt["draws"],
                      "n_train": heads["world_model"].n_train, "n_train_pos": heads["world_model"].n_pos,
                      "train_cutoffs": f"{min(ts)}-{max(ts)}", "transition_windows": {int(t): int(v) for t, v in n_win.items()}}
    res["calibration"] = calibration(y[m], w[m], heads["world_model"].prob(s["world_model"][m]))
    pd.DataFrame(res["calibration"]).to_csv(SRC / ("fwm_calibration_dev.csv" if DEV else "fwm_calibration.csv"), index=False)
    res["subgroups"] = subgroups(F, d, T, s, y, w, ent, years, pdoc, states[T], rng, 50 if DEV else 200)
    update_results(KEY, res)
    log("w6.primary", **{k: (round(v["auroc"], 4), round(v["auroc_popularity_adjusted"], 4)) for k, v in prim.items()},
        secs=round(time.time() - t0), **({"dev": True} if DEV else {}))
    if DEV:
        log("w6.done", secs=round(time.time() - t0))
        return

    res["probe_random_encoder"] = random_probe(years, ent, M, pdoc, frames, FC, T, d, s, F, y, w, m, rng, train)
    update_results(KEY, res)

    S = states[T]
    I, J, ex, raw = exact_scores(S, T, {k: (heads[k], d) for k in VARIANTS}, feats,
                                 old={"hazard_v1_neural": (old["neural"], d)})
    ex.update({"resource_allocation": raw["raw_ra"], "degree_product": raw["raw_pa"]})
    nV = len(S.nodes)
    pos = F.key.values[F[f"y{d}"].values == 1]
    lin = I.astype(np.int64) * nV + J
    ye = np.isin(lin, S.posarr[pos // S.E] * nV + S.posarr[pos % S.E]).astype(np.int8)
    assert ye.sum() == len(pos), (ye.sum(), len(pos))
    exm = {k: W.exact_metrics(ye, v.astype(np.float64)) for k, v in ex.items()}
    tk = W.topk_bootstrap(I, J, ye, {k: ex[k] for k in ("world_model", "history", "resource_allocation", "hazard_v1_neural")},
                          (100, 1000, 10000), rng, 500,
                          pairs=[("world_model", "resource_allocation"), ("world_model", "history"), ("world_model", "hazard_v1_neural")])
    strat = W.degree_matched_strata(S, I, J)
    res["exact_primary"] = {"n_candidates": int(len(ye)), "n_positives": int(ye.sum()), "metrics": exm, "topk_ci": tk,
                            "entity_centred": {k: W.entity_centred(I, J, ye, ex[k], nV) for k in ("world_model", "history", "resource_allocation")},
                            "degree_matched_auroc": {k: W.stratified_auroc(ye, ex[k], strat) for k in ("world_model", "history", "resource_allocation", "degree_product")}}
    rocs, prs = [], []
    for k in ("world_model", "history", "hazard_v1_neural", "resource_allocation", "degree_product"):
        g_, tp = W.roc_points(ye, ex[k])
        rocs += [{"method": k, "fpr": a_, "tpr": b_} for a_, b_ in zip(g_, tp)]
        kk, pr, rc = W.pr_points(ye, ex[k].astype(np.float64))
        prs += [{"method": k, "k": int(a_), "precision": float(b_), "recall": float(c_)} for a_, b_, c_ in zip(kk, pr, rc)]
    pd.DataFrame(rocs).to_csv(SRC / "fwm_roc_exact_primary.csv", index=False)
    pd.DataFrame(prs).to_csv(SRC / "fwm_pr_exact_primary.csv", index=False)
    del ex, I, J, lin, ye, strat
    update_results(KEY, res)
    log("w6.exact_primary", **{k: (round(v["auroc"], 4), round(v["precision_at_1000"], 4)) for k, v in exm.items()},
        secs=round(time.time() - t0))
    holdout(res, years, ent, M, pdoc, emb, frames, feats, snaps, rng, t0)
    update_results(KEY, res)
    log("w6.done", secs=round(time.time() - t0))


def attention_eval(snaps, Z, FC, FCc, T, rng):
    nodes = snaps.t[T]["nodes"]
    truth = snaps.lc[nodes][:, T - 1999: T - 1999 + H_FC]
    pred = {"last_observation": np.repeat(snaps.t[T]["last"][:, None], H_FC, 1), "timesfm": FC[T],
            "timesfm_adapter_no_embedding": corrected(snaps, Z, FC, T, False, SEED + T), "timesfm_adapter": FCc[T]}
    err = {k: np.abs(v - truth).mean(1) for k, v in pred.items()}
    out = {"cutoff": T, "entities": int(len(nodes)), "mae": {k: float(v.mean()) for k, v in err.items()}}
    n = len(nodes)
    bs = {k: [] for k in err}
    for _ in range(1000):
        ix = rng.integers(0, n, n)
        for k, v in err.items():
            bs[k].append(v[ix].mean())
    out["mae_ci"] = {k: ci(v) for k, v in bs.items()}
    out["diff_ci"] = {f"{a}_minus_{b}": ci(np.array(bs[a]) - np.array(bs[b])) for a, b in
                      (("timesfm_adapter", "timesfm"), ("timesfm_adapter", "timesfm_adapter_no_embedding"),
                       ("timesfm", "last_observation"), ("timesfm_adapter", "last_observation"))}
    out["by_year"] = {k: np.abs(v - truth).mean(0).tolist() for k, v in pred.items()}
    pd.DataFrame([{"method": k, "year": T + 1 + j, "mae": out["by_year"][k][j]} for k in pred for j in range(H_FC)]).to_csv(
        SRC / "fwm_attention_forecast.csv", index=False)
    return out


def dynamics_eval(snaps, Z, P, FCc, T, rng, dev):
    nodes = snaps.t[T]["nodes"]
    tgt = Z[T + ROLL_K][snaps.t[T + ROLL_K]["pos"][nodes]]
    obs = snaps.lc[nodes][:, T - 1999: T - 1999 + ROLL_K]
    pred = {"locf": Z[T], "rollout_forecast_activity": rollout_from(P[T], snaps, Z, T, FCc[T])[:, -1],
            "rollout_observed_activity": rollout_from(P[T], snaps, Z, T, obs)[:, -1]}
    z0w, dz0w, tzw, _, _, mkw = _windows(snaps, Z, T)
    full = (mkw[:, -1] > 0).numpy()
    Z0, D0, Y3 = z0w.numpy()[full], dz0w.numpy()[full], tzw.numpy()[full][:, -1]
    from sklearn.linear_model import Ridge
    rg = Ridge(alpha=1.0).fit(np.hstack([Z0, D0]), Y3 - Z0)
    dzT = Z[T] - previous_state(snaps, Z, T)
    pred["drift"] = Z[T] + (Y3 - Z0).mean(0)
    pred["ridge"] = Z[T] + rg.predict(np.hstack([Z[T], dzT])).astype(np.float32)
    p0, _ = train_predictor(snaps, Z, T, SEED + 300, zero_actions=True)
    pred["rollout_no_action"] = rollout_from(p0, snaps, Z, T, FCc[T], zero_actions=True)[:, -1]
    perm = np.random.default_rng(SEED + 301).permutation(len(nodes))
    pred["rollout_shuffled_activity"] = rollout_from(P[T], snaps, Z, T, FCc[T][perm])[:, -1]
    sub = np.sort(rng.choice(len(nodes), min(2000, len(nodes)), replace=False))
    import truststore
    truststore.inject_into_ssl()
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(TFM_ID)
    model.compile(timesfm.ForecastConfig(max_context=32, max_horizon=8, normalize_inputs=True, per_core_batch_size=512,
                                         force_flip_invariance=True, infer_is_positive=False, fix_quantile_crossing=True))
    series = []
    for i in sub:
        e = nodes[i]
        hist = [Z[u][snaps.t[u]["pos"][e]] for u in range(SNAP_FIRST, T + 1) if snaps.t[u]["pos"][e] >= 0]
        Hh = np.stack(hist)
        series += [Hh[:, k].astype(np.float32) for k in range(D_Z)]
    pf, _ = model.forecast(horizon=ROLL_K, inputs=series)
    tf = np.asarray(pf[:, ROLL_K - 1], np.float32).reshape(len(sub), D_Z)
    out = {"cutoff": T, "target_year": T + ROLL_K, "entities": int(len(nodes)), "timesfm_subsample": int(len(sub))}
    se = {k: ((v[sub] - tgt[sub]) ** 2) for k, v in pred.items()}
    se["timesfm_latent"] = (tf - tgt[sub]) ** 2
    out["mse"] = {k: float(v.mean()) for k, v in se.items()}
    out["skill_vs_locf"] = {k: float(1 - v.mean() / se["locf"].mean()) for k, v in se.items() if k != "locf"}
    out["skill_by_block"] = {k: {g: float(1 - v[:, sl].mean() / se["locf"][:, sl].mean()) for g, sl in BLOCKS.items()}
                             for k, v in se.items() if k != "locf"}
    n = len(sub)
    bs = {k: [] for k in out["skill_vs_locf"]}
    per = {k: v.mean(1) for k, v in se.items()}
    perb = {k: {g: v[:, sl].mean(1) for g, sl in BLOCKS.items()} for k, v in se.items()}
    bsb = {k: {g: [] for g in BLOCKS} for k in bs}
    for _ in range(1000):
        ix = rng.integers(0, n, n)
        base = per["locf"][ix].mean()
        for k in bs:
            bs[k].append(1 - per[k][ix].mean() / base)
            for g in BLOCKS:
                bsb[k][g].append(1 - perb[k][g][ix].mean() / perb["locf"][g][ix].mean())
    out["skill_ci"] = {k: ci(v) for k, v in bs.items()}
    out["skill_by_block_ci"] = {k: {g: ci(v) for g, v in d_.items()} for k, d_ in bsb.items()}
    se_all = {k: (v - tgt) ** 2 for k, v in pred.items()}
    out["skill_all_entities"] = {k: float(1 - v.mean() / se_all["locf"].mean()) for k, v in se_all.items() if k != "locf"}
    var = ((tgt[sub] - tgt[sub].mean(0)) ** 2).mean()
    out["r2"] = {k: float(1 - v.mean() / var) for k, v in se.items()}
    pd.DataFrame([{"method": k, "skill": out["skill_vs_locf"].get(k, 0.0), "lo": out["skill_ci"].get(k, [0, 0])[0],
                   "hi": out["skill_ci"].get(k, [0, 0])[1], "r2": out["r2"][k],
                   **{f"skill_{g}": out["skill_by_block"].get(k, {}).get(g, 0.0) for g in BLOCKS}} for k in se]).to_csv(
        SRC / ("fwm_dynamics_dev.csv" if dev else "fwm_dynamics.csv"), index=False)
    return out


def subgroups(F, d, T, s, y, w, ent, years, pdoc, S, rng, draws):
    m = w > 0
    rows = []
    three = ["world_model", "history", "resource_allocation"]

    def add(group, level, mask):
        mm = m & mask
        yy = y[mm]
        if yy.sum() < 10 or (yy == 0).sum() < 10:
            return
        bt = W.node_bootstrap2(F[mm], d, {k: s[k][mm] for k in three}, rng, draws,
                               contrasts=(("world_model", "resource_allocation"),), popularity=three)
        for k in three:
            ev = W.evaluate(yy, w[mm], s[k][mm], F.raw_pa.values[mm])
            rows.append({"group": group, "level": level, "method": k, **ev, "auroc_ci": bt["auroc_ci"][k],
                         "auroc_popularity_adjusted_ci": bt["popadj_ci"].get(k)})
    for tp in W.TYPEPAIRS:
        add("type_pair", tp, F.typepair.values == tp)
    add("common_neighbours", "none", F.raw_cn.values == 0)
    add("common_neighbours", "one_or_more", F.raw_cn.values > 0)
    adv = set(np.flatnonzero(ent.entity.isin(W.ADVANCED).values))
    is_adv = np.isin(F.a.values, list(adv)) | np.isin(F.b.values, list(adv))
    add("advanced_liver_disease", "involves", is_adv)
    add("advanced_liver_disease", "does_not_involve", ~is_adv)
    from common import to_group
    L = pd.read_parquet(CORPUS_PROC / "literature_included_final.parquet",
                        columns=["record_id", "pmid", "evidence_type"])
    L = L[L.pmid.notna()]
    L["group"] = to_group(L.evidence_type.values, L.record_id.values)
    D = pd.read_parquet(OUT / "docs.parquet", columns=["pmid"])
    grp = pd.Series(L.group.values, index=L.pmid.astype(np.int64).values)
    grp = grp[~grp.index.duplicated()]
    dgrp = grp.reindex(D.pmid.values).values
    fut = pdoc[(years[pdoc.doc.values] > T) & (years[pdoc.doc.values] <= T + d)].copy()
    fut["key"] = fut.a.astype(np.int64) * S.E + fut.b
    fut["g"] = dgrp[fut.doc.values]
    fut = fut[fut.key.isin(F.key[F[f"y{d}"] == 1])]
    for g in ["interventional", "observational", "basic", "synthesis"]:
        keys = set(fut.key[fut.g == g])
        add("first_reports_include", g, (y == 0) | np.isin(F.key.values, list(keys)))
    out = pd.DataFrame(rows)
    out.to_csv(SRC / "fwm_subgroups.csv", index=False)
    return out.to_dict(orient="records")


def random_probe(years, ent, M, pdoc, frames, FC, T, d, s, F, y, w, m, rng, train):
    e2 = np.load(OUT / "emb_random.npy").astype(np.float32)
    sn2 = Snapshots(years, ent, M, pdoc, e2)
    enc2, _ = train_encoder(sn2, SEED)
    Z2 = embed_all(enc2, sn2)
    ts = [t for t in sorted(frames) if t >= W.TRAIN_FIRST and t + d <= T]
    need = sorted(set(ts + [T]))
    FCc2 = {t: corrected(sn2, Z2, FC, t, True, SEED + t) for t in need}
    R2 = {}
    for t in need:
        p2, _ = train_predictor(sn2, Z2, t, SEED + 100 + t)
        R2[t] = rollout_from(p2, sn2, Z2, t, FCc2[t])
    f2 = WMFeatures(Z2, R2, FCc2, sn2)
    h = Head("world_model").fit(train, d, f2)
    ia, ib = sn2.t[T]["pos"][F.a.values], sn2.t[T]["pos"][F.b.values]
    s2 = {"world_model": s["world_model"], "world_model_random_encoder": h.score(F, d, T, ia, ib, f2)}
    bt = W.node_bootstrap2(F[m], d, {k: v[m] for k, v in s2.items()}, rng, 200,
                           contrasts=(("world_model", "world_model_random_encoder"),), popularity=list(s2),
                           pop_contrasts=(("world_model", "world_model_random_encoder"),))
    return {k: {**{q: W.evaluate(y, w, v, F.raw_pa.values)[q] for q in ("auroc", "auroc_popularity_adjusted")},
                "auroc_ci": bt["auroc_ci"][k], "auroc_popularity_adjusted_ci": bt["popadj_ci"][k]} for k, v in s2.items()} | {
        "contrast": bt["diff_ci"], "contrast_popularity_adjusted": bt["popadj_diff_ci"]}


def holdout(res, years, ent, M, pdoc, emb, frames, feats, snaps, rng, t0):
    S25 = W.State(2025, years, ent, M, pdoc, emb)
    assert np.array_equal(S25.nodes, snaps.t[2025]["nodes"])
    F25, _ = W.frame(S25, pdoc, years, np.random.default_rng(SEED + 3), 10, horizons=[1], end_override=2026)
    tr1 = case_control(frames, 1, [t for t in sorted(frames) if t + 1 <= 2025], rng)
    h1 = Head("world_model").fit(tr1, 1, feats)
    hh = Head("history").fit(tr1, 1, feats)
    tr3 = case_control(frames, 3, [t for t in sorted(frames) if t + 3 <= 2025], rng)
    h3 = Head("world_model").fit(tr3, 3, feats)
    hh3 = Head("history").fit(tr3, 3, feats)
    hz1 = W.fit_all([frames[t] for t in sorted(frames) if t + 1 <= 2025], 1, np.random.default_rng(SEED + 1), neural=True,
                    glm=False, logistic=False)["neural"]
    hz3 = W.fit_all([frames[t] for t in sorted(frames) if t + 3 <= 2025], 3, np.random.default_rng(SEED + 1), neural=True,
                    glm=False, logistic=False)["neural"]
    I, J, ex, raw = exact_scores(S25, 2025, {"world_model": (h1, 1), "history": (hh, 1), "forecast_3y": (h3, 3),
                                             "history_3y": (hh3, 3)}, feats,
                                 old={"hazard_v1_neural": (hz1, 1), "hazard_3y": (hz3, 3)})
    ex.update({"resource_allocation": raw["raw_ra"], "degree_product": raw["raw_pa"]})
    nV = len(S25.nodes)
    pos = F25.key.values[F25.y1.values == 1]
    lin = I.astype(np.int64) * nV + J
    ye = np.isin(lin, S25.posarr[pos // S25.E] * nV + S25.posarr[pos % S25.E]).astype(np.int8)
    hold = ("world_model", "history", "hazard_v1_neural", "resource_allocation", "degree_product")
    eh = {"n_candidates": int(len(ye)), "n_positives": int(ye.sum()), "label_window": "2026-01 to 2026-06",
          "metrics": {k: W.exact_metrics(ye, ex[k].astype(np.float64)) for k in hold}}
    eh["topk_ci"] = W.topk_bootstrap(I, J, ye, {k: ex[k] for k in ("world_model", "history", "resource_allocation", "hazard_v1_neural")},
                                     (100, 1000), rng, 200, pairs=[("world_model", "resource_allocation"), ("world_model", "hazard_v1_neural")])
    rocs = []
    for k in hold:
        g_, tp = W.roc_points(ye, ex[k])
        rocs += [{"method": k, "fpr": a_, "tpr": b_} for a_, b_ in zip(g_, tp)]
    pd.DataFrame(rocs).to_csv(SRC / "fwm_roc_exact_holdout.csv", index=False)
    res["exact_holdout"] = eh
    log("w6.holdout", pos=int(ye.sum()), **{k: (round(v["auroc"], 4), round(v["precision_at_1000"], 4)) for k, v in eh["metrics"].items()},
        secs=round(time.time() - t0))


if __name__ == "__main__":
    main()
