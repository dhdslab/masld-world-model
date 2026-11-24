from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from common import CORPUS_PROC, PKG, PROTOCOL, SRC, log, update_results

OUT = PKG / "analysis" / "world_model"
WM_SEED = 20261117
TAU, K_WHITE, K_Z, MIN_DOCS, POS_DOCS = 3.0, 128, 32, 3, 2
NEG_TEST, NEG_TRAIN_RATIO = 1_000_000, 20
TEST_CUTOFFS, HORIZONS, TRAIN_FIRST, LAST_FULL = [2020, 2021, 2022, 2023, 2024], [1, 2, 3], 2010, 2025
PRESPECIFIED_CUTOFFS = {2020, 2021, 2022, 2023}
PRIMARY = (2022, 3)
RNG8 = WM_SEED + 8
QUERY_RX = re.compile(r"(?i)fatty liver|steato|nafld|\bnafl\b|\bnash\b|mafld|masld|\bmash\b|metald")
ALWAYS_EXCLUDED = {"Disease:MESH:D065626", "Disease:MESH:D005234"}
ADVANCED = {f"Disease:MESH:{m}" for m in ["D008103", "D006975", "D004932", "D001201", "D006501", "D017093", "D017114",
                                         "D065290", "D006528", "D008113", "D006530", "D058625", "D020065"]}
TYPES = ["Chemical", "Disease", "Gene", "Variant"]
SPECIES = "--species-specific" in __import__("sys").argv
DEV = "--dev" in __import__("sys").argv
CACHE_TAG = "_species" if SPECIES else ""
SEM = ["cos", "triad"]
DYN = ["proj", "conv"]
DRIFT = ["drift"]
GRAPH = ["cn", "ra", "aa", "jac", "pa"]
ATT = ["n_lo", "n_hi", "g_lo", "g_hi"]
TYPEPAIRS = [f"{a}-{b}" for a, b in itertools.combinations_with_replacement(TYPES, 2)]
BASE_X = SEM + DYN + DRIFT + GRAPH + ATT


def norm_entity(e: str) -> str:
    t, ident = e.split(":", 1)
    if t == "Variant":
        m = re.search(r"RS#:(\d+)", ident)
        if m:
            return f"Variant:rs{m.group(1)}"
        h = re.search(r"HGVS:([^;]+)", ident)
        g = re.search(r"CorrespondingGene:(\d+)", ident)
        if h:
            return f"Variant:{h.group(1)}|{g.group(1) if g else ''}"
        core = re.search(r"tmVar:([^;]+)", ident)
        if core:
            return f"Variant:{core.group(1)}|{g.group(1) if g else ''}"
    if t == "Gene" and ";" in ident:
        ident = ident.split(";")[0]
    return f"{t}:{ident}"


def human_orthologs():
    cache = OUT / "human_ortholog_map.csv"
    if cache.exists():
        m = pd.read_csv(cache)
        return dict(zip(m.other.astype(str), m.human.astype(str)))
    rows = []
    for ch in pd.read_csv(OUT / "gene_orthologs.gz", sep="\t", chunksize=2_000_000, dtype=str,
                          names=["tax", "gene", "rel", "otax", "ogene"], skiprows=1):
        a = ch[(ch.tax == "9606") & (ch.rel == "Ortholog")]
        rows.append(pd.DataFrame({"other": a.ogene.values, "human": a.gene.values}))
        b = ch[(ch.otax == "9606") & (ch.rel == "Ortholog")]
        rows.append(pd.DataFrame({"other": b.gene.values, "human": b.ogene.values}))
    m = pd.concat(rows).drop_duplicates("other")
    m.to_csv(cache, index=False)
    return dict(zip(m.other, m.human))


_D = r"[\s\-–‐]+"
DISEASE_NAME_RX = re.compile(
    rf"(?i)\bmetabolic{_D}(?:dysfunction{_D})?(?:and{_D}alcohol{_D})?(?:associated|related){_D}"
    rf"(?:fatty{_D}liver(?:{_D}disease)?|steatotic{_D}liver{_D}disease|steatohepatitis|liver{_D}disease)"
    rf"|\bnon{_D}?alcoholic{_D}(?:fatty{_D}liver(?:{_D}disease)?|steatohepatitis|steatotic{_D}liver{_D}disease)"
    rf"|\bnonalcoholic{_D}(?:fatty{_D}liver(?:{_D}disease)?|steatohepatitis)"
    rf"|\bsteatotic{_D}liver{_D}disease")
TOKRX = re.compile(r"[a-z0-9α-ωµ]+")


def drop_disease_name_spans(A, D):
    rows = []
    for pm, ti, ab in zip(D.pmid.values, D.title.fillna("").values, D.abstract.fillna("").values):
        for part, t in (("title", ti), ("abstract", ab)):
            for m in DISEASE_NAME_RX.finditer(t):
                rows.append((pm, part, m.start(), m.end()))
    S = pd.DataFrame(rows, columns=["pmid", "part", "s", "e"])
    A = A.reset_index(drop=True)
    A["_i"] = np.arange(len(A))
    j = A[["_i", "pmid", "part", "start", "end"]].merge(S, on=["pmid", "part"])
    hit = j[(j.start < j.e) & (j.end > j.s)]._i.unique()
    keep = ~np.isin(A._i.values, hit)
    return A[keep].drop(columns="_i").reset_index(drop=True), int((~keep).sum()), int(len(S))


def merge_same_name(A):
    top = A.assign(m=A.mention.str.lower()).groupby(["entity", "m"]).size().reset_index(name="n")
    top = top.sort_values(["entity", "n"], ascending=[True, False]).drop_duplicates("entity")
    key = top.m.str.replace(r"[^a-z0-9]", "", regex=True)
    top["key"] = np.where(key == "", top.entity, key)
    top["tot"] = A.entity.value_counts().reindex(top.entity).values
    canon = top.sort_values(["key", "tot"], ascending=[True, False]).drop_duplicates("key").set_index("key").entity
    mp = dict(zip(top.entity, canon.loc[top.key].values))
    n_merged = int(sum(1 for k, v in mp.items() if k != v))
    A["entity"] = A.entity.map(mp)
    A["type"] = A.entity.str.split(":").str[0]
    return A, n_merged


def annotation_drift(A, D, years, ratio=3.0, min_text=20, min_ann=20):
    from collections import Counter
    ed = A[["entity", "doc"]].drop_duplicates()
    ann = ed.assign(y=years[ed.doc.values]).groupby(["entity", "y"]).size().unstack(fill_value=0)
    cand = ann.index[ann.max(1) >= min_ann]
    top = A[A.entity.isin(cand)].assign(m=A.mention.str.lower()).groupby(["entity", "m"]).size().reset_index(name="n")
    top = top.sort_values(["entity", "n"], ascending=[True, False]).drop_duplicates("entity").set_index("entity").m
    toks = {e: tuple(TOKRX.findall(m)) for e, m in top.items()}
    tgt = {}
    for e, t in toks.items():
        if t and len(t) <= 6:
            tgt.setdefault(len(t), set()).add(t)
    cnt = Counter()
    for y, ti, ab in zip(years, D.title.fillna("").values, D.abstract.fillna("").values):
        tk = TOKRX.findall((ti + " " + ab).lower())
        for L, st in tgt.items():
            if L <= len(tk):
                for g in set(zip(*(tk[i:] for i in range(L)))) & st:
                    cnt[(g, int(y))] += 1
    yrs = sorted(ann.columns)
    out = []
    for e, t in toks.items():
        if not t or len(t) > 6:
            continue
        tx = np.array([cnt.get((t, y), 0) for y in yrs], float)
        an = ann.loc[e, yrs].values.astype(float)
        best = (1.0, None)
        for k in range(1, len(yrs)):
            if tx[k - 1] >= min_text and tx[k] >= min_text and max(an[k - 1], an[k]) >= min_ann:
                r0, r1 = an[k - 1] / tx[k - 1], an[k] / tx[k]
                q = max(r1, 1e-9) / max(r0, 1e-9)
                q = max(q, 1 / q)
                if q > best[0]:
                    best = (q, yrs[k])
        if best[0] >= ratio:
            out.append({"entity": e, "mention": top[e], "max_ratio": round(best[0], 2), "year": best[1]})
    return pd.DataFrame(out, columns=["entity", "mention", "max_ratio", "year"])


def prepare():
    cache = OUT / f"prep{CACHE_TAG}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        ent = pd.read_parquet(OUT / f"entities{CACHE_TAG}.parquet")
        pdoc = pd.read_parquet(OUT / f"pair_doc{CACHE_TAG}.parquet")
        M = sp.csr_matrix((z["m_data"], z["m_ind"], z["m_ptr"]), shape=tuple(z["m_shape"]))
        return z["years"], ent, M, pdoc
    D = pd.read_parquet(OUT / "docs.parquet")
    A = pd.read_parquet(OUT / "ann.parquet")
    A = A[A.pmid.isin(D.pmid)].copy()
    A["entity"] = A.entity.map(norm_entity)
    orth = {} if SPECIES else human_orthologs()
    isg = A.entity.str.startswith("Gene:")
    A.loc[isg, "entity"] = "Gene:" + A.loc[isg, "entity"].str[5:].map(lambda g: orth.get(g, g))
    row = pd.Series(np.arange(len(D)), index=D.pmid.values)
    A["doc"] = row.loc[A.pmid.values].values
    years = D.year.values.astype(np.int16)
    A, n_span, n_name_spans = drop_disease_name_spans(A, D)
    n_merged = 0
    if not SPECIES:
        A, n_merged = merge_same_name(A)
    drift = annotation_drift(A, D, years)
    A["qhit"] = A.mention.str.contains(QUERY_RX)
    share = A[A.type == "Disease"].groupby("entity").qhit.mean()
    excl_masld = set(share[share >= 0.5].index) | ALWAYS_EXCLUDED
    excl = excl_masld | set(drift.entity)
    names = A.groupby("entity").mention.agg(lambda s: s.str.lower().value_counts().index[0])
    ent = pd.DataFrame({"entity": names.index, "name": names.values})
    ent["type"] = ent.entity.str.split(":").str[0]
    ent["masld_family"] = ent.entity.isin(excl_masld)
    ent["annotation_drift"] = ent.entity.isin(set(drift.entity)) & ~ent.masld_family
    ent["excluded"] = ent.entity.isin(excl)
    (OUT / f"prep_stats{CACHE_TAG}.json").write_text(json.dumps(
        {"annotations_in_disease_names_dropped": n_span, "disease_name_spans": n_name_spans, "entities_merged_same_name": n_merged,
         "entities_excluded_annotation_drift": int(ent.annotation_drift.sum()),
         "drift_examples": drift.sort_values("max_ratio", ascending=False).head(15).to_dict(orient="records")}, ensure_ascii=False))
    drift.to_csv(SRC / f"wm_annotation_drift{CACHE_TAG}.csv", index=False)
    code = pd.Series(np.arange(len(ent)), index=ent.entity.values)
    A["e"] = code.loc[A.entity.values].values
    de = A[["doc", "e"]].drop_duplicates()
    M = sp.csr_matrix((np.ones(len(de), np.float32), (de.doc.values, de.e.values)), shape=(len(D), len(ent)))
    ent["n_docs"] = np.asarray(M.sum(0)).ravel().astype(int)
    sid = np.zeros(len(A), dtype=np.int32)
    ab = A.part.values == "abstract"
    texts = D.set_index("pmid").abstract.to_dict()
    bounds = {}
    for pm in A.pmid.values[ab]:
        if pm not in bounds:
            t = texts.get(pm) or ""
            bounds[pm] = np.array([0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", t)])
    starts = A.start.values
    pms = A.pmid.values
    for i in np.flatnonzero(ab):
        sid[i] = 1 + int(np.searchsorted(bounds[pms[i]], starts[i], side="right")) - 1
    A["sid"] = sid
    S = A[~A.entity.isin(excl)][["doc", "sid", "e"]].drop_duplicates()
    P = S.merge(S, on=["doc", "sid"])
    P = P[P.e_x < P.e_y]
    pdoc = P[["e_x", "e_y", "doc"]].drop_duplicates().rename(columns={"e_x": "a", "e_y": "b"}).astype(np.int32)
    ent.to_parquet(OUT / f"entities{CACHE_TAG}.parquet", index=False)
    pdoc.to_parquet(OUT / f"pair_doc{CACHE_TAG}.parquet", index=False)
    pd.DataFrame({"doc": A.doc.values.astype(np.int32), "e": A.e.values.astype(np.int32), "mention": A.mention.str.lower().values,
                  "qhit": A.qhit.values, "type": A.type.values}).to_parquet(OUT / f"mentions{CACHE_TAG}.parquet", index=False)
    np.savez(cache, years=years, m_data=M.data, m_ind=M.indices, m_ptr=M.indptr, m_shape=np.array(M.shape))
    log("w3.prepare", docs=len(D), entities=len(ent), excluded=len(excl), mentions=len(A), sentence_pairs=len(pdoc))
    return years, ent, M, pdoc


class State:

    def __init__(self, T, years, ent, M, pdoc, emb, tau=TAU, year_override=None):
        yrs = years if year_override is None else year_override
        self.T = T
        upto = yrs <= T
        n_T = np.asarray(M[upto].sum(0)).ravel()
        keep = (n_T >= MIN_DOCS) & (~ent.excluded.values)
        self.nodes = np.flatnonzero(keep)
        self.pos = {c: i for i, c in enumerate(self.nodes)}
        self.posarr = np.full(len(ent), -1, dtype=np.int64)
        self.posarr[self.nodes] = np.arange(len(self.nodes))
        self.E = len(ent)
        nV = len(self.nodes)
        pk = pdoc[yrs[pdoc.doc.values] <= T]
        self.known = np.unique(pk.a.values.astype(np.int64) * self.E + pk.b.values)
        inV = keep[pk.a.values] & keep[pk.b.values]
        ia = self.posarr[pk.a.values[inV]]
        ib = self.posarr[pk.b.values[inV]]
        A = sp.coo_matrix((np.ones(len(ia), np.float32), (ia, ib)), shape=(nV, nV)).tocsr()
        A = ((A + A.T) > 0).astype(np.float32).tocsr()
        self.A = A
        self.deg = np.asarray(A.sum(1)).ravel()
        X = emb[upto].astype(np.float32)
        mu = X.mean(0)
        C = np.cov((X - mu).T)
        w, U = np.linalg.eigh(C)
        idx = np.argsort(w)[::-1][:K_WHITE]
        Wm = U[:, idx] / np.sqrt(np.maximum(w[idx], 1e-8))
        Et = (emb.astype(np.float32) - mu) @ Wm.astype(np.float32)
        Ms = M[:, self.nodes]

        def hstate(t):
            wt = np.where(yrs <= t, np.exp(-(t - yrs) / tau) if np.isfinite(tau) else 1.0, 0.0).astype(np.float32)
            H = np.asarray(Ms.T @ sp.diags(wt) @ Et)
            nrm = np.linalg.norm(H, axis=1, keepdims=True)
            return np.where(nrm > 0, H / np.maximum(nrm, 1e-12), 0.0).astype(np.float32), nrm.ravel() > 0
        self.H, _ = hstate(T)
        H2, ok2 = hstate(T - 2)
        self.H2 = np.where(ok2[:, None], H2, self.H)
        self.Vel = self.H - self.H2
        rec = (yrs > T - 2) & (yrs <= T)
        old = (yrs > T - 4) & (yrs <= T - 2)
        self.u = (Et[rec].mean(0) - Et[old].mean(0)).astype(np.float32)
        n2 = np.asarray(M[yrs <= T - 2].sum(0)).ravel()[self.nodes]
        n4 = np.asarray(M[yrs <= T - 4].sum(0)).ravel()[self.nodes]
        self.n = n_T[self.nodes]
        self.g = np.log((self.n - n2 + 1) / (n2 - n4 + 1))
        cen = np.asarray(A @ self.H)
        cn = np.linalg.norm(cen, axis=1, keepdims=True)
        self.Cn = np.where(cn > 0, cen / np.maximum(cn, 1e-12), 0.0).astype(np.float32)
        self.types = ent.type.values[self.nodes]
        self.codes = self.nodes
        self.syn = synonym_codes(ent)

    def n_candidates(self):
        nV = len(self.nodes)
        kn = self.known
        a, b = kn // self.E, kn % self.E
        inV = np.isin(a, self.nodes) & np.isin(b, self.nodes)
        same = self.syn[a] == self.syn[b]
        grp = pd.Series(self.syn[self.nodes]).value_counts()
        same_pairs = int((grp * (grp - 1) // 2).sum())
        return nV * (nV - 1) // 2 - int(inV.sum()) - same_pairs + int((inV & same).sum())


def synonym_codes(ent):
    if SPECIES:
        return np.arange(len(ent))
    key = ent.name.fillna("").str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    key = np.where(key == "", ent.entity.values, key)
    return pd.factorize(pd.Series(key))[0]


def future_counts(pdoc, years, T, end, docmask=None):
    yr = years[pdoc.doc.values]
    sel = (yr > T) & (yr <= end)
    if docmask is not None:
        sel &= docmask[pdoc.doc.values]
    f = pdoc[sel]
    return f.groupby(["a", "b"]).doc.nunique()


def pair_features(S: State, ia, ib):
    H, H2 = S.H, S.H2
    f = {}
    f["cos"] = (H[ia] * H[ib]).sum(1)
    for d in (0.5, 1, 2, 3):
        pa_ = H[ia] + (d / 2) * S.Vel[ia]
        pb_ = H[ib] + (d / 2) * S.Vel[ib]
        f[f"proj{d}"] = (pa_ * pb_).sum(1) / np.maximum(np.linalg.norm(pa_, axis=1) * np.linalg.norm(pb_, axis=1), 1e-12)
    f["conv"] = f["cos"] - (H2[ia] * H2[ib]).sum(1)
    f["triad"] = 0.5 * ((H[ia] * S.Cn[ib]).sum(1) + (H[ib] * S.Cn[ia]).sum(1))
    f["drift"] = (0.5 * (H[ia] + H[ib])) @ S.u
    na, nb = np.log1p(S.n[ia]), np.log1p(S.n[ib])
    f["n_lo"], f["n_hi"] = np.minimum(na, nb), np.maximum(na, nb)
    f["g_lo"], f["g_hi"] = np.minimum(S.g[ia], S.g[ib]), np.maximum(S.g[ia], S.g[ib])
    deg = S.deg
    inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1), 0.0)
    invlog = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2)), 0.0)
    cn = np.zeros(len(ia)); ra = np.zeros(len(ia)); aa = np.zeros(len(ia))
    for k in range(0, len(ia), 100_000):
        Xa = S.A[ia[k:k + 100_000]]
        Xb = S.A[ib[k:k + 100_000]]
        P = Xa.multiply(Xb).tocsr()
        cn[k:k + 100_000] = np.asarray(P.sum(1)).ravel()
        ra[k:k + 100_000] = P @ inv
        aa[k:k + 100_000] = P @ invlog
    da, db = deg[ia], deg[ib]
    f["cn"], f["ra"], f["aa"] = np.log1p(cn), np.log1p(ra), np.log1p(aa)
    f["jac"] = np.where(da + db - cn > 0, cn / np.maximum(da + db - cn, 1), 0.0)
    f["pa"] = np.log1p(da * db)
    f["raw_ra"], f["raw_pa"], f["raw_cn"] = ra, da * db, cn
    ta, tb = S.types[ia], S.types[ib]
    tp = np.where(ta <= tb, np.char.add(np.char.add(ta.astype(str), "-"), tb.astype(str)),
                  np.char.add(np.char.add(tb.astype(str), "-"), ta.astype(str)))
    out = pd.DataFrame(f)
    out["typepair"] = tp
    Z = S.H[:, :K_Z]
    zf = np.hstack([Z[ia] * Z[ib], np.abs(Z[ia] - Z[ib])]).astype(np.float32)
    return out, zf


def frame(S: State, pdoc, years, rng, n_sample, horizons=HORIZONS, end_override=None, docmask=None, pos_docs=POS_DOCS):
    E, nodes, inV = S.E, S.nodes, np.zeros(S.E, bool)
    inV[nodes] = True
    known = S.known
    labels, pos_keys = {}, []
    for d in horizons:
        end = end_override if end_override is not None else S.T + d
        fc = future_counts(pdoc, years, S.T, end, docmask)
        fc = fc[fc >= pos_docs]
        a = fc.index.get_level_values(0).values.astype(np.int64)
        b = fc.index.get_level_values(1).values.astype(np.int64)
        ok = inV[a] & inV[b]
        ok &= S.syn[a] != S.syn[b]
        k = a[ok] * E + b[ok]
        k = k[~np.isin(k, known)]
        labels[d] = set(k.tolist())
        pos_keys.append(k)
    pos_all = np.unique(np.concatenate(pos_keys)) if pos_keys else np.array([], np.int64)
    nV = len(nodes)
    want = n_sample
    samp = []
    got = 0
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
    sample = np.concatenate(samp)
    n_cand = S.n_candidates()
    keys = np.unique(np.concatenate([sample, pos_all]))
    a, b = keys // E, keys % E
    ia = S.posarr[a]; ib = S.posarr[b]
    F, Z = pair_features(S, ia, ib)
    F["key"], F["a"], F["b"] = keys, a, b
    F["in_sample"] = np.isin(keys, sample)
    for d in horizons:
        F[f"y{d}"] = np.isin(keys, np.fromiter(labels[d], np.int64)).astype(np.int8)
        n_pos = int(F[f"y{d}"].sum())
        n_neg_pop = n_cand - n_pos
        n_neg_s = int((F.in_sample & (F[f"y{d}"] == 0)).sum())
        F[f"w{d}"] = np.where(F[f"y{d}"] == 1, 1.0, np.where(F.in_sample, n_neg_pop / max(n_neg_s, 1), 0.0))
    F["T"] = S.T
    F.attrs.update({"n_candidates": n_cand, "n_nodes": nV})
    return F, Z


def xmat(F, d, cols):
    X = F[[c if c != "proj" else f"proj{d}" for c in cols]].values.astype(np.float32)
    T = np.stack([(F.typepair.values == tp).astype(np.float32) for tp in TYPEPAIRS], 1)
    return np.hstack([X, T])


def train_sets(frames, d, cols, rng, use_z=True):
    Xs, Zs, ys, ws = [], [], [], []
    for F, Z in frames:
        y, w = F[f"y{d}"].values, F[f"w{d}"].values
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero((y == 0) & (w > 0))
        if len(pos) == 0:
            continue
        take = rng.choice(neg, min(len(neg), NEG_TRAIN_RATIO * len(pos)), replace=False)
        ix = np.concatenate([pos, take])
        wn = np.where(y[ix] == 1, 1.0, (len(neg) / len(take)) * w[take[0]] if len(take) else 0.0)
        wn = wn / wn.mean()
        Xs.append(xmat(F.iloc[ix], d, cols)); Zs.append(Z[ix] if use_z else np.zeros_like(Z[ix])); ys.append(y[ix]); ws.append(wn)
    return np.vstack(Xs), np.vstack(Zs), np.concatenate(ys).astype(np.float32), np.concatenate(ws).astype(np.float32)


class Hazard:

    def __init__(self, neural, seed=WM_SEED):
        self.neural, self.seed = neural, seed

    def fit(self, X, Z, y, w, d):
        import torch
        torch.manual_seed(self.seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xs = torch.tensor((X - self.mu) / self.sd)
        Zt = torch.tensor(Z)
        yt = torch.tensor(y)
        if self.neural:
            npos = float(y.sum()); nneg = float(len(y) - npos)
            wt = torch.tensor(np.where(y == 1, len(y) / (2 * npos), len(y) / (2 * nneg)).astype(np.float32))
        else:
            wt = torch.tensor(w)
        self.c = 0.0
        p_in = X.shape[1]
        self.lin = torch.nn.Linear(p_in, 1)
        params = list(self.lin.parameters())
        if self.neural:
            self.mlp = torch.nn.Sequential(torch.nn.Linear(p_in + Z.shape[1], 64), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                           torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(64, 1))
            params += list(self.mlp.parameters())
        off = math.log(d)

        def nll(xb, zb, yb, wb):
            eta = self.lin(xb).squeeze(1) + off
            if self.neural:
                eta = eta + self.mlp(torch.cat([xb, zb], 1)).squeeze(1)
            eta = eta.clamp(-30, 8)
            logp = torch.log(-torch.expm1(-torch.exp(eta)) + 1e-12)
            log1mp = -torch.exp(eta)
            return -(wb * (yb * logp + (1 - yb) * log1mp)).mean()
        if not self.neural:
            opt = torch.optim.LBFGS(params, lr=0.5, max_iter=200, line_search_fn="strong_wolfe")
            lam = 1.0 / len(y)

            def closure():
                opt.zero_grad()
                loss = nll(Xs, Zt, yt, wt) + lam * (self.lin.weight ** 2).sum() / 2
                loss.backward()
                return loss
            opt.step(closure)
        else:
            opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
            n = len(y)
            for ep in range(30):
                self.mlp.train()
                perm = torch.randperm(n)
                for k in range(0, n, 4096):
                    b = perm[k:k + 4096]
                    opt.zero_grad()
                    loss = nll(Xs[b], Zt[b], yt[b], wt[b])
                    loss.backward()
                    opt.step()
            self.mlp.eval()
            from scipy.optimize import minimize_scalar
            e = self.eta(X, Z) + off

            def wnll(c):
                ee = np.clip(e + c, -30, 8)
                return -(w * (y * np.log(-np.expm1(-np.exp(ee)) + 1e-12) - (1 - y) * np.exp(ee))).sum()
            self.c = float(minimize_scalar(wnll, bounds=(-25, 25), method="bounded").x)
        return self

    def eta(self, X, Z):
        import torch
        with torch.no_grad():
            xb = torch.tensor((X - self.mu) / self.sd)
            e = self.lin(xb).squeeze(1)
            if self.neural:
                e = e + self.mlp(torch.cat([xb, torch.tensor(Z)], 1)).squeeze(1)
        return e.numpy().astype(np.float64) + getattr(self, "c", 0.0)

    def prob(self, X, Z, d):
        return 1 - np.exp(-d * np.exp(np.clip(self.eta(X, Z), -30, 8)))

    def coef(self, names):
        w = self.lin.weight.detach().numpy().ravel()
        return {"per_unit": dict(zip(names, (w / self.sd).tolist())),
                "standardized": dict(zip(names, w.tolist())),
                "intercept": float(self.lin.bias.detach().numpy()[0])}


def fit_all(frames_tr, d, rng, cols=BASE_X, neural=True, glm=True, logistic=True, use_z=True):
    X, Z, y, w = train_sets(frames_tr, d, cols, rng, use_z)
    out = {}
    if glm:
        out["hazard"] = Hazard(False).fit(X, Z, y, w, d)
    if neural:
        out["neural"] = Hazard(True).fit(X, Z, y, w, d)
    if logistic:
        Xl = np.column_stack([X[:, cols.index("ra")], X[:, cols.index("cos")]])
        sc_mu, sc_sd = Xl.mean(0), Xl.std(0) + 1e-9
        bal = np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (len(y) - y.sum())))
        lr = LogisticRegression(C=1.0, max_iter=2000).fit((Xl - sc_mu) / sc_sd, y, sample_weight=bal)
        out["logistic"] = (lr, sc_mu, sc_sd)
    out["n_train"], out["n_train_pos"] = int(len(y)), int(y.sum())
    return out


def scores(models, F, Z, d, cols=BASE_X):
    s = {"degree_product": F.raw_pa.values.astype(float), "resource_allocation": F.raw_ra.values.astype(float),
         "semantic": F.cos.values.astype(float)}
    X = xmat(F, d, cols)
    if "logistic" in models:
        lr, mu, sd = models["logistic"]
        s["logistic_ra_semantic"] = lr.decision_function((np.column_stack([X[:, cols.index("ra")], X[:, cols.index("cos")]]) - mu) / sd)
    if "hazard" in models:
        s["hazard_world_model"] = models["hazard"].eta(X, Z)
    if "neural" in models:
        s["neural_world_model"] = models["neural"].eta(X, Z)
    return s


def evaluate(y, w, s, pa=None, ks=(100, 1000)):
    m = w > 0
    y, w, s = y[m], w[m], s[m]
    if pa is not None:
        pa = np.asarray(pa)[m]
    out = {"n_pos": int(y.sum()), "n_neg_sampled": int((y == 0).sum()), "auroc": float(roc_auc_score(y, s)),
           "ap": float(average_precision_score(y, s, sample_weight=w)), "prevalence": float((w * y).sum() / w.sum())}
    o = np.argsort(-s, kind="stable")
    cw = np.cumsum(w[o]); cy = np.cumsum((y * w)[o])
    for k in ks:
        j = int(np.searchsorted(cw, k))
        out[f"precision_at_{k}"] = float(cy[min(j, len(cy) - 1)] / max(cw[min(j, len(cw) - 1)], 1e-9))
    if pa is not None:
        edges = np.unique(np.quantile(pa[y == 0], np.linspace(0, 1, 11)))
        strat = np.clip(np.searchsorted(edges, pa, side="right") - 1, 0, len(edges) - 2)
        num = den = 0.0
        for q in np.unique(strat):
            sel = strat == q
            npos, nneg = int(y[sel].sum()), int((y[sel] == 0).sum())
            if npos >= 10 and nneg >= 10:
                num += npos * roc_auc_score(y[sel], s[sel]); den += npos
        out["auroc_popularity_adjusted"] = float(num / den) if den else float("nan")
    return out


def node_bootstrap(F, d, s_dict, rng, draws=500, contrasts=(("neural_world_model", "resource_allocation"),)):
    codes = np.unique(np.concatenate([F.a.values, F.b.values]))
    ia = np.searchsorted(codes, F.a.values); ib = np.searchsorted(codes, F.b.values)
    y = F[f"y{d}"].values
    base = (F[f"w{d}"].values > 0).astype(float)
    res = {k: [] for k in s_dict}
    diff = {f"{a}_minus_{b}": [] for a, b in contrasts}
    for _ in range(draws):
        m = rng.poisson(1.0, len(codes)).astype(float)
        w = m[ia] * m[ib] * base
        if (w * y).sum() == 0:
            continue
        au = {k: roc_auc_score(y, v, sample_weight=w) for k, v in s_dict.items()}
        for k, v in au.items():
            res[k].append(v)
        for a_, b_ in contrasts:
            diff[f"{a_}_minus_{b_}"].append(au[a_] - au[b_])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {k: ci(v) for k, v in res.items()}, {k: ci(v) for k, v in diff.items()}, len(res[next(iter(res))])


def roc_points(y, s, n=400):
    fpr, tpr, _ = roc_curve(y, s)
    grid = np.unique(np.r_[0.0, np.logspace(-6, 0, n)])
    return grid, tpr[np.searchsorted(fpr, grid, side="right") - 1]


def strata_of(pa, y):
    edges = np.unique(np.quantile(pa[y == 0], np.linspace(0, 1, 11)))
    return np.clip(np.searchsorted(edges, pa, side="right") - 1, 0, len(edges) - 2)


def popadj(y, s, strat, keep, w=None):
    num = den = 0.0
    for q in keep:
        sel = strat == q
        yy = y[sel]
        ww = np.ones(len(yy)) if w is None else w[sel]
        wp, wn = (ww * yy).sum(), (ww * (1 - yy)).sum()
        if wp <= 0 or wn <= 0:
            continue
        num += wp * roc_auc_score(yy, s[sel], sample_weight=None if w is None else ww); den += wp
    return num / den if den else float("nan")


class FastAUC:

    def __init__(self, s, y):
        self.o = np.argsort(s, kind="mergesort")
        ss = s[self.o]
        self.grp = np.r_[0, np.cumsum(ss[1:] != ss[:-1])] if len(ss) else np.array([], int)
        self.y = y[self.o].astype(bool)
        self.ng = int(self.grp[-1]) + 1 if len(ss) else 0

    def __call__(self, w):
        wo = w[self.o]
        wp = np.bincount(self.grp, weights=wo * self.y, minlength=self.ng)
        wn = np.bincount(self.grp, weights=wo * ~self.y, minlength=self.ng)
        den = wp.sum() * wn.sum()
        return float(((np.cumsum(wn) - wn + 0.5 * wn) * wp).sum() / den) if den > 0 else float("nan"), float(wp.sum())


def node_bootstrap2(F, d, s_dict, rng, draws, contrasts=(), popularity=(), pop_contrasts=(), strat=None):
    codes = np.unique(np.concatenate([F.a.values, F.b.values]))
    ia = np.searchsorted(codes, F.a.values); ib = np.searchsorted(codes, F.b.values)
    y = F[f"y{d}"].values.astype(float)
    base = (F[f"w{d}"].values > 0).astype(float)
    strat = strata_of(F.raw_pa.values, y) if strat is None else strat
    keep = [q for q in np.unique(strat) if y[strat == q].sum() >= 10 and (y[strat == q] == 0).sum() >= 10]
    fa = {k: FastAUC(np.asarray(v, float), y) for k, v in s_dict.items()}
    idx = {q: np.flatnonzero(strat == q) for q in keep}
    fq = {k: {q: FastAUC(np.asarray(s_dict[k], float)[ix], y[ix]) for q, ix in idx.items()} for k in popularity}
    au = {k: [] for k in s_dict}; pu = {k: [] for k in popularity}
    for _ in range(draws):
        mlt = rng.poisson(1.0, len(codes)).astype(float)
        w = mlt[ia] * mlt[ib] * base
        if (w * y).sum() == 0:
            continue
        for k in s_dict:
            au[k].append(fa[k](w)[0])
        for k in popularity:
            num = den = 0.0
            for q, ix in idx.items():
                a_, wp = fq[k][q](w[ix])
                if np.isfinite(a_) and wp > 0:
                    num += wp * a_; den += wp
            pu[k].append(num / den if den else np.nan)
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    out = {"draws": len(next(iter(au.values()))), "auroc_ci": {k: ci(v) for k, v in au.items()},
           "popadj_ci": {k: ci(v) for k, v in pu.items()},
           "diff_ci": {f"{a}_minus_{b}": ci(np.array(au[a]) - np.array(au[b])) for a, b in contrasts},
           "popadj_diff_ci": {f"{a}_minus_{b}": ci(np.array(pu[a]) - np.array(pu[b])) for a, b in pop_contrasts}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="biomedbert")
    ap.add_argument("--quick", action="store_true", help="primary task only, few bootstrap draws (development)")
    ap.add_argument("--primary-only", action="store_true", help="encoder probes: same training cutoffs, primary task only")
    ap.add_argument("--species-specific", action="store_true", help="pre-specified gene identifiers (before amendment A6)")
    ap.add_argument("--dev", action="store_true", help="development run of the full code path on few cutoffs (not reported)")
    args = ap.parse_args()
    global TAG, KEY
    TAG = args.encoder + ("_prespecified" if SPECIES else "") + ("_dev" if args.dev else "")
    KEY = ("world_model" if args.encoder == "biomedbert" else f"world_model_{args.encoder}") + ("_prespecified" if SPECIES else "")         + ("_dev" if args.dev else "")
    rng = np.random.default_rng(WM_SEED)
    t0 = time.time()
    years, ent, M, pdoc = prepare()
    emb = np.load(OUT / f"emb_{args.encoder}.npy").astype(np.float32)
    proto = (PROTOCOL / "World_Model_Analysis_Plan.md").read_bytes()
    corpus = {"papers": int(len(years)), "papers_by_period": {"2000-2009": int(((years >= 2000) & (years <= 2009)).sum()),
              "2010-2019": int(((years >= 2010) & (years <= 2019)).sum()), "2020-2025": int(((years >= 2020) & (years <= 2025)).sum()),
              "2026H1": int((years == 2026).sum())},
              "entities": int(len(ent)), "entities_excluded_masld_family": int(ent.masld_family.sum()),
              "entities_excluded_annotation_drift": int(ent.annotation_drift.sum()),
              "entity_types": ent[~ent.excluded].type.value_counts().to_dict(), "sentence_co_mentions": int(len(pdoc)),
              "a9": json.loads((OUT / f"prep_stats{CACHE_TAG}.json").read_text())}
    cut_all = list(range(TRAIN_FIRST, LAST_FULL))
    if args.quick or args.dev:
        cut_all = list(range(2013, 2020)) + [2022]
    if args.primary_only:
        cut_all = list(range(TRAIN_FIRST, 2020)) + [2022]
    states, frames, task_sizes = {}, {}, []
    for T in cut_all:
        S = State(T, years, ent, M, pdoc, emb)
        hz = [d for d in HORIZONS if T + d <= LAST_FULL]
        n_s = NEG_TEST if T in TEST_CUTOFFS else 400_000
        F, Z = frame(S, pdoc, years, rng, n_s, horizons=hz)
        states[T], frames[T] = S, (F, Z)
        task_sizes.append({"cutoff": T, "nodes": int(len(S.nodes)), "candidates": int(F.attrs["n_candidates"]),
                           "known_edges_in_graph": int(S.A.nnz // 2),
                           **{f"positives_{d}": int(F[f'y{d}'].sum()) for d in hz}})
        log("w3.frame", T=T, nodes=len(S.nodes), candidates=F.attrs["n_candidates"],
            **{f"pos{d}": int(F[f'y{d}'].sum()) for d in hz}, secs=round(time.time() - t0))
    res = {"protocol_sha256_current": hashlib.sha256(proto).hexdigest(), "encoder": args.encoder, "corpus": corpus,
           "settings": {"tau": TAU, "whitening_dims": K_WHITE, "z_dims": K_Z, "min_docs_node": MIN_DOCS, "positive_docs": POS_DOCS,
                        "test_negatives": NEG_TEST, "train_negative_ratio": NEG_TRAIN_RATIO, "seed": WM_SEED},
           "task_sizes": task_sizes,
           "papers_by_year": {int(y): int((years == y).sum()) for y in np.unique(years)}}
    grid = []
    tests = [(T, d) for T in TEST_CUTOFFS for d in HORIZONS if T + d <= LAST_FULL and T in frames]
    fitted = {}
    for T, d in tests:
        tr = [frames[t] for t in sorted(frames) if t >= TRAIN_FIRST and t + d <= T]
        mods = fit_all(tr, d, rng)
        fitted[(T, d)] = mods
        F, Z = frames[T]
        s = scores(mods, F, Z, d)
        y, w = F[f"y{d}"].values, F[f"w{d}"].values
        used = [t for t in sorted(frames) if t >= TRAIN_FIRST and t + d <= T]
        row = {"cutoff": T, "horizon": d, "label_window": f"{T + 1}-{T + d}", "n_nodes": F.attrs["n_nodes"],
               "n_candidates": F.attrs["n_candidates"], "n_train": mods["n_train"], "n_train_pos": mods["n_train_pos"],
               "train_cutoffs": f"{min(used)}-{max(used)}", "prespecified_task": bool(T in PRESPECIFIED_CUTOFFS)}
        for k, v in s.items():
            e = evaluate(y, w, v, F.raw_pa.values)
            grid.append({**row, "method": k, **e})
        log("w3.test", T=T, d=d, **{k: round(roc_auc_score(y[w > 0], v[w > 0]), 4) for k, v in s.items()})
    G = pd.DataFrame(grid)
    G.to_csv(SRC / f"wm_auroc_grid_{TAG}.csv", index=False)
    res["grid"] = G.to_dict(orient="records")
    T, d = PRIMARY
    F, Z = frames[T]
    mods = fitted[(T, d)]
    s = scores(mods, F, Z, d)
    y, w = F[f"y{d}"].values, F[f"w{d}"].values
    m = w > 0
    prim = {k: evaluate(y, w, v, F.raw_pa.values) for k, v in s.items()}
    draws = 50 if (args.quick or args.dev) else (200 if args.primary_only else 500)
    pairs_ = (("neural_world_model", "resource_allocation"), ("neural_world_model", "semantic"),
              ("neural_world_model", "hazard_world_model"), ("hazard_world_model", "resource_allocation"),
              ("neural_world_model", "logistic_ra_semantic"), ("neural_world_model", "degree_product"))
    bt = node_bootstrap2(F[m], d, {k: v[m] for k, v in s.items()}, rng, draws, contrasts=pairs_, popularity=list(s),
                         pop_contrasts=pairs_)
    for k in prim:
        prim[k]["auroc_ci"] = bt["auroc_ci"][k]
        prim[k]["auroc_popularity_adjusted_ci"] = bt["popadj_ci"][k]
    res["primary"] = {"cutoff": T, "horizon": d, "label_window": f"{T + 1}-{T + d}", "metrics": prim,
                      "contrasts": bt["diff_ci"], "contrasts_popularity_adjusted": bt["popadj_diff_ci"],
                      "bootstrap_draws": bt["draws"], "n_nodes": F.attrs["n_nodes"], "n_candidates": F.attrs["n_candidates"],
                      "hazard_coefficients": mods["hazard"].coef(BASE_X + TYPEPAIRS)}
    rocs = []
    for k, v in s.items():
        fp, tp = roc_points(y[m], v[m])
        rocs += [{"method": k, "fpr": a_, "tpr": b_} for a_, b_ in zip(fp, tp)]
    pd.DataFrame(rocs).to_csv(SRC / f"wm_roc_primary_{TAG}.csv", index=False)
    log("w3.primary", **{k: (round(v["auroc"], 4), [round(x, 4) for x in v["auroc_ci"]], round(v["auroc_popularity_adjusted"], 4))
                         for k, v in prim.items()}, **({"dev": True} if args.dev else {}))
    update_results(KEY, res)
    if args.quick or args.primary_only:
        return
    keep = secondary(res, frames, states, fitted, years, ent, M, pdoc, emb, rng, args)
    update_results(KEY, res)
    if args.encoder == "biomedbert":
        amendment_a8(res, frames, states, fitted, years, ent, M, pdoc, emb, keep, args)
        update_results(KEY, res)
    log("w3.done", secs=round(time.time() - t0))


def secondary(res, frames, states, fitted, years, ent, M, pdoc, emb, rng, args):
    T, d = PRIMARY
    F, Z = frames[T]
    mods = fitted[(T, d)]
    s = scores(mods, F, Z, d)
    y, w = F[f"y{d}"].values, F[f"w{d}"].values
    m = w > 0
    sub = []

    def add(group, level, mask):
        mm = m & mask
        yy = y[mm]
        if yy.sum() < 10 or (yy == 0).sum() < 10:
            return
        two = ("neural_world_model", "resource_allocation")
        five = ["neural_world_model", "hazard_world_model", "resource_allocation", "semantic", "degree_product"]
        bt = node_bootstrap2(F[mm], d, {k: s[k][mm] for k in five}, rng, 200, contrasts=(two,), popularity=two,
                             pop_contrasts=(two,))
        for k in five:
            ev = evaluate(yy, w[mm], s[k][mm], F.raw_pa.values[mm])
            extra = {"auroc_ci": bt["auroc_ci"][k]}
            if k in bt["popadj_ci"]:
                extra["auroc_popularity_adjusted_ci"] = bt["popadj_ci"][k]
            if k == "neural_world_model":
                extra["minus_resource_allocation_ci"] = bt["diff_ci"]["neural_world_model_minus_resource_allocation"]
                extra["popadj_minus_resource_allocation_ci"] = bt["popadj_diff_ci"]["neural_world_model_minus_resource_allocation"]
            sub.append({"group": group, "level": level, "method": k, **ev, **extra})
    for tp in TYPEPAIRS:
        add("type_pair", tp, F.typepair.values == tp)
    add("common_neighbours", "none", F.raw_cn.values == 0)
    add("common_neighbours", "one_or_more", F.raw_cn.values > 0)
    adv = set(ent.index[ent.entity.isin(ADVANCED)])
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
    fut["key"] = fut.a.astype(np.int64) * states[T].E + fut.b
    fut["g"] = dgrp[fut.doc.values]
    fut = fut[fut.key.isin(F.key[F[f"y{d}"] == 1])]
    for g in ["interventional", "observational", "basic", "synthesis"]:
        keys = set(fut.key[fut.g == g])
        add("first_reports_include", g, (y == 0) | np.isin(F.key.values, list(keys)))
    S_ = pd.DataFrame(sub)
    S_.to_csv(SRC / f"wm_subgroups_{TAG}.csv", index=False)
    res["subgroups"] = S_.to_dict(orient="records")
    p = mods["neural"].prob(xmat(F, d, BASE_X), Z, d)
    o = np.argsort(p[m]); cw = np.cumsum(w[m][o]); dec = np.minimum((cw / cw[-1] * 10).astype(int), 9)
    from scipy.stats import chi2
    cal = pd.DataFrame({"dec": dec, "p": p[m][o], "y": y[m][o], "w": w[m][o]}).groupby("dec").apply(
        lambda g: pd.Series({"mean_predicted": np.average(g.p, weights=g.w), "observed": np.average(g.y, weights=g.w),
                             "n_pos": int(g.y.sum()), "weighted_n": float(g.w.sum())})).reset_index()
    k_ = cal.n_pos.values
    cal["observed_lo"] = np.where(k_ > 0, chi2.ppf(0.025, 2 * k_) / 2, 0.0) / cal.weighted_n
    cal["observed_hi"] = chi2.ppf(0.975, 2 * (k_ + 1)) / 2 / cal.weighted_n
    cal.to_csv(SRC / f"wm_calibration_{TAG}.csv", index=False)
    lg = np.log(np.clip(p[m], 1e-12, 1 - 1e-12) / (1 - np.clip(p[m], 1e-12, 1 - 1e-12)))
    slope_fit = LogisticRegression(C=1e6, max_iter=5000).fit(lg[:, None], y[m], sample_weight=w[m] / w[m].mean())
    res["calibration"] = {"bins": cal.to_dict(orient="records"),
                          "mean_predicted": float(np.average(p[m], weights=w[m])), "observed": float(np.average(y[m], weights=w[m])),
                          "slope": float(slope_fit.coef_[0, 0]), "intercept": float(slope_fit.intercept_[0])}
    first = fut.assign(yr=years[fut.doc.values]).groupby("key").yr.min()
    pk = F[F[f"y{d}"] == 1].set_index("key")
    fy = first.reindex(pk.index).values.astype(float)
    eta = pd.Series(s["neural_world_model"], index=F.key.values).reindex(pk.index).values
    num = den = 0
    for i in range(len(fy)):
        sel = fy[i] < fy
        num += (eta[i] > eta[sel]).sum() + 0.5 * (eta[i] == eta[sel]).sum()
        den += sel.sum()
    res["time_to_report_concordance"] = float(num / max(den, 1))
    res["time_to_report_pairs"] = int(den)
    blocks = {"semantic_state": SEM, "semantic_dynamics": DYN, "field_drift": DRIFT, "graph": GRAPH, "attention": ATT}
    tr = [frames[t] for t in sorted(frames) if t >= TRAIN_FIRST and t + d <= T]
    abl = [{"removed": "none", "model": k, **{q: evaluate(y, w, s[k], F.raw_pa.values)[q] for q in ("auroc", "auroc_popularity_adjusted")}}
           for k in ["neural_world_model", "hazard_world_model"]]
    abl_scores = {}
    for name, cols in blocks.items():
        keep = [c for c in BASE_X if c not in cols]
        use_z = name != "semantic_state"
        mo = fit_all(tr, d, rng, cols=keep, logistic=False, use_z=use_z)
        X = xmat(F, d, keep)
        Zb = Z if use_z else np.zeros_like(Z)
        for k, h in [("neural_world_model", mo["neural"]), ("hazard_world_model", mo["hazard"])]:
            sc = h.eta(X, Zb)
            abl_scores[(name, k)] = sc
            ev = evaluate(y, w, sc, F.raw_pa.values)
            abl.append({"removed": name, "model": k, "auroc": ev["auroc"], "auroc_popularity_adjusted": ev["auroc_popularity_adjusted"]})
    A_ = pd.DataFrame(abl)
    A_.to_csv(SRC / f"wm_ablations_{TAG}.csv", index=False)
    res["ablations"] = A_.to_dict(orient="records")
    S25 = State(2025, years, ent, M, pdoc, emb)
    F25, Z25 = frame(S25, pdoc, years, rng, NEG_TEST, horizons=[1], end_override=2026)
    F25 = F25.rename(columns={"y1": "yH", "w1": "wH"})
    tr1 = [frames[t] for t in sorted(frames) if t >= TRAIN_FIRST and t + 1 <= 2025]
    m1 = fit_all(tr1, 1, rng)
    F25["y1"], F25["w1"] = F25.yH, F25.wH
    s25 = scores(m1, F25, Z25, 1)
    y25, w25 = F25.y1.values, F25.w1.values
    near = {k: evaluate(y25, w25, v, F25.raw_pa.values) for k, v in s25.items()}
    res["near_prospective"] = {"cutoff": 2025, "label_window": "2026-01 to 2026-06", "metrics": near,
                               "n_nodes": F25.attrs["n_nodes"], "n_candidates": F25.attrs["n_candidates"]}
    rocs = []
    for k, v in s25.items():
        fp, tp = roc_points(y25[w25 > 0], v[w25 > 0])
        rocs += [{"method": k, "fpr": a_, "tpr": b_} for a_, b_ in zip(fp, tp)]
    pd.DataFrame(rocs).to_csv(SRC / f"wm_roc_near_prospective_{TAG}.csv", index=False)
    log("w3.near", **{k: round(v["auroc"], 4) for k, v in near.items()})
    keep = {"abl_scores": abl_scores, "s": s, "S25": S25, "F25": F25, "Z25": Z25, "s25": s25, "tr1": tr1, "m1": m1,
            "dgrp": dgrp}
    if args.encoder != "biomedbert":
        return keep
    if not DEV:
        sens = []

        rng_s = np.random.default_rng(RNG8 + 1)

        def sens_run(name, frames_tr, F_te, Z_te, yv=None, wv=None):
            mo = fit_all(frames_tr, d, rng, logistic=False)
            sc = scores(mo, F_te, Z_te, d)
            yy = F_te[f"y{d}"].values if yv is None else yv
            ww = F_te[f"w{d}"].values if wv is None else wv
            mm_ = ww > 0
            two = ("neural_world_model", "resource_allocation")
            bt = node_bootstrap2(F_te[mm_], d, {k: sc[k][mm_] for k in two}, rng_s, 200, contrasts=(two,))
            for k in ["neural_world_model", "hazard_world_model", "resource_allocation", "semantic"]:
                extra = {"auroc_ci": bt["auroc_ci"][k]} if k in two else {}
                if k == "neural_world_model":
                    extra["minus_resource_allocation_ci"] = bt["diff_ci"]["neural_world_model_minus_resource_allocation"]
                sens.append({"analysis": name, "method": k, **evaluate(yy, ww, sc[k], F_te.raw_pa.values), **extra})
        for thr in (1, 3):
            fr = {}
            for t in sorted(frames):
                if t + d <= T or t == T:
                    fr[t] = frame(states[t], pdoc, years, rng, NEG_TEST if t == T else 400_000, horizons=[d], pos_docs=thr)
            sens_run(f"label_{thr}_papers", [fr[t] for t in fr if t + d <= T], *fr[T])
        prim_mask = np.isin(dgrp, ["interventional", "observational", "basic"])
        fr = {t: frame(states[t], pdoc, years, rng, NEG_TEST if t == T else 400_000, horizons=[d], docmask=prim_mask)
              for t in sorted(frames) if t + d <= T or t == T}
        sens_run("primary_research_labels", [fr[t] for t in fr if t + d <= T], *fr[T])
        fr = {}
        for t in sorted(frames):
            if t + d <= T or t == T:
                St = State(t, years, ent, M, pdoc, emb, tau=np.inf)
                fr[t] = frame(St, pdoc, years, rng, NEG_TEST if t == T else 400_000, horizons=[d])
        sens_run("no_time_decay", [fr[t] for t in fr if t + d <= T], *fr[T])
        shuf = years.copy()
        tm = years <= T - d
        shuf[tm] = rng.permutation(years[tm])
        fr = []
        for t in sorted(frames):
            if t + d <= T:
                St = State(t, shuf, ent, M, pdoc, emb, year_override=shuf)
                fr.append(frame(St, pdoc, shuf, rng, 400_000, horizons=[d]))
        sens_run("shuffled_training_years", fr, F, Z)
        SE = pd.DataFrame(sens)
        SE.to_csv(SRC / "wm_sensitivity.csv", index=False)
        res["sensitivity"] = SE.to_dict(orient="records")
    E_ = S25.E
    allk = pd.Series(pdoc.a.values.astype(np.int64) * E_ + pdoc.b.values).value_counts()
    inV = np.zeros(E_, bool); inV[S25.nodes] = True
    strong = allk[allk >= 2].index.values
    strong = strong[inV[strong // E_] & inV[strong % E_] & (S25.syn[strong // E_] != S25.syn[strong % E_])]
    held = rng.choice(strong, int(0.2 * len(strong)), replace=False)
    keepk = np.setdiff1d(S25.known, held)
    S_h = State(2025, years, ent, M, pdoc, emb)
    ka, kb = keepk // E_, keepk % E_
    ok = inV[ka] & inV[kb]
    ia = S_h.posarr[ka[ok]]; ib = S_h.posarr[kb[ok]]
    Ah = sp.coo_matrix((np.ones(len(ia), np.float32), (ia, ib)), shape=S_h.A.shape).tocsr()
    S_h.A = ((Ah + Ah.T) > 0).astype(np.float32).tocsr(); S_h.deg = np.asarray(S_h.A.sum(1)).ravel()
    never = []
    while sum(len(x) for x in never) < 2 * len(held):
        i = rng.integers(0, len(S_h.nodes), 4 * len(held)); j = rng.integers(0, len(S_h.nodes), 4 * len(held))
        a = np.minimum(S_h.nodes[i], S_h.nodes[j]).astype(np.int64); b = np.maximum(S_h.nodes[i], S_h.nodes[j]).astype(np.int64)
        k = np.unique((a * E_ + b)[(a != b) & (S_h.syn[a] != S_h.syn[b])])
        never.append(k[~np.isin(k, allk.index.values)])
    neg = rng.permutation(np.concatenate(never))[: 2 * len(held)]
    keys = np.concatenate([held, neg])
    ia = S_h.posarr[keys // E_]; ib = S_h.posarr[keys % E_]
    Fh, _ = pair_features(S_h, ia, ib)
    yh = np.r_[np.ones(len(held)), np.zeros(len(neg))]
    prim_au = res["primary"]["metrics"]
    res["design_contrast"] = {
        "random_holdout_never_negatives": {"resource_allocation": float(roc_auc_score(yh, Fh.raw_ra)),
                                           "semantic": float(roc_auc_score(yh, Fh.cos)), "degree_product": float(roc_auc_score(yh, Fh.raw_pa))},
        "temporal_primary": {k: prim_au[k]["auroc"] for k in ["resource_allocation", "semantic", "degree_product", "neural_world_model"]}}
    Fh["a"], Fh["b"] = keys // E_, keys % E_
    Fh["yH"], Fh["wH"] = yh.astype(np.int8), 1.0
    keep["holdout"] = Fh
    return keep


NOLANG = GRAPH + ATT


class GBDT:

    def __init__(self, cols):
        self.cols = cols

    def fit(self, frames_tr, d, rng):
        from sklearn.ensemble import HistGradientBoostingClassifier
        X, _, y, _ = train_sets(frames_tr, d, self.cols, rng, use_z=False)
        self.m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                                random_state=WM_SEED).fit(X, y)
        return self

    def eta(self, F, d):
        p = np.clip(self.m.predict_proba(xmat(F, d, self.cols))[:, 1], 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p))


def fit_baselines(frames_tr, d, rng):
    out = {}
    X, Z, y, w = train_sets(frames_tr, d, ATT, rng, use_z=False)
    out["attention_only"] = (Hazard(False).fit(X, Z, y, w, d), ATT)
    X, Z, y, w = train_sets(frames_tr, d, NOLANG, rng, use_z=False)
    out["no_language_hazard"] = (Hazard(False).fit(X, Z, y, w, d), NOLANG)
    out["no_language_neural"] = (Hazard(True).fit(X, Z, y, w, d), NOLANG)
    out["gbdt_no_language"] = GBDT(NOLANG).fit(frames_tr, d, rng)
    out["gbdt_all_features"] = GBDT(BASE_X).fit(frames_tr, d, rng)
    return out


def score_baselines(base, F, Z, d):
    s = {}
    for k, v in base.items():
        if isinstance(v, GBDT):
            s[k] = v.eta(F, d)
        else:
            h, cols = v
            s[k] = h.eta(xmat(F, d, cols), np.zeros_like(Z))
    return s


def refeaturize(frame_dict, make_state):
    out = {}
    for t, (F, Z) in frame_dict.items():
        S2 = make_state(t)
        F2, Z2 = pair_features(S2, S2.posarr[F.a.values], S2.posarr[F.b.values])
        for c in F.columns:
            if c not in F2.columns:
                F2[c] = F[c].values
        F2.attrs.update(F.attrs)
        out[t] = (F2, Z2)
    return out


def enumerate_candidates(S, block=300):
    nV = len(S.nodes)
    syn = S.syn[S.nodes]
    A = S.A.tocsr()
    for i0 in range(0, nV, block):
        i1 = min(nV, i0 + block)
        ii, jj = np.nonzero(~(A[i0:i1].toarray() > 0))
        ii = ii + i0
        ok = jj > ii
        ii, jj = ii[ok], jj[ok]
        ok = syn[ii] != syn[jj]
        yield ii[ok], jj[ok]


def exact_scores(S, fns, chunk=400_000):
    I, J, out, raw = [], [], {k: [] for k in fns}, {"raw_pa": [], "raw_cn": [], "raw_ra": [], "cos": []}
    for ii, jj in enumerate_candidates(S):
        for k in range(0, len(ii), chunk):
            a, b = ii[k:k + chunk], jj[k:k + chunk]
            F, Z = pair_features(S, a, b)
            for name, fn in fns.items():
                out[name].append(np.asarray(fn(F, Z), np.float32))
            for c in raw:
                raw[c].append(F[c].values.astype(np.float32))
            I.append(a.astype(np.int32)); J.append(b.astype(np.int32))
    return (np.concatenate(I), np.concatenate(J), {k: np.concatenate(v) for k, v in out.items()},
            {k: np.concatenate(v) for k, v in raw.items()})


def topk_precision(y, s, k):
    part = np.argpartition(-s, k - 1)[:k]
    thr = s[part].min()
    gt = s > thr
    ngt = int(gt.sum())
    tie = s == thr
    hits = y[gt].sum() + (k - ngt) * y[tie].mean()
    return float(hits / k)


def exact_metrics(y, s, ks=(100, 1000, 10000)):
    out = {"auroc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s))}
    for k in ks:
        out[f"precision_at_{k}"] = topk_precision(y, s, k)
        out[f"recall_at_{k}"] = out[f"precision_at_{k}"] * k / float(y.sum())
    return out


def pr_points(y, s, ks=None):
    ks = np.unique(np.logspace(1, 6.5, 60).astype(int)) if ks is None else ks
    o = np.argsort(-s, kind="stable")
    cy = np.cumsum(y[o])
    return ks, cy[ks - 1] / ks, cy[ks - 1] / y.sum()


def topk_bootstrap(I, J, y, s_dict, ks, rng, draws=500, pairs=()):
    nodes = np.unique(np.concatenate([I, J]))
    ia, ib = np.searchsorted(nodes, I), np.searchsorted(nodes, J)
    tops = {(m, k): np.argpartition(-s, k - 1)[:k] for m, s in s_dict.items() for k in ks}
    vals = {key: [] for key in tops}
    for _ in range(draws):
        mlt = rng.poisson(1.0, len(nodes)).astype(float)
        for key, ix in tops.items():
            w = mlt[ia[ix]] * mlt[ib[ix]]
            vals[key].append((w * y[ix]).sum() / max(w.sum(), 1e-12))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = {f"{m}@{k}": ci(v) for (m, k), v in vals.items()}
    for a, b in pairs:
        for k in ks:
            out[f"{a}_minus_{b}@{k}"] = ci(np.array(vals[(a, k)]) - np.array(vals[(b, k)]))
    return out


def entity_centred(I, J, y, s, nV):
    from scipy.stats import rankdata
    e = np.concatenate([I, J]); sc = np.concatenate([s, s]); yy = np.concatenate([y, y])
    o = np.lexsort((-sc, e))
    e, sc, yy = e[o], sc[o], yy[o]
    st = np.searchsorted(e, np.arange(nV)); en = np.searchsorted(e, np.arange(nV), side="right")
    au, hit, rr, npos_ = [], [], [], []
    for i in range(nV):
        a, b = st[i], en[i]
        yv = yy[a:b]
        npos = int(yv.sum())
        if npos == 0 or npos == b - a:
            continue
        r = rankdata(sc[a:b])
        au.append((r[yv == 1].sum() - npos * (npos + 1) / 2) / (npos * (b - a - npos)))
        hit.append(float(yv[:10].any()))
        rr.append(1.0 / (1 + int(np.argmax(yv))))
        npos_.append(npos)
    return {"entities": len(au), "mean_auroc": float(np.mean(au)), "hits_at_10": float(np.mean(hit)), "mrr": float(np.mean(rr))}


def degree_matched_strata(S, I, J, bins=20):
    from scipy.stats import rankdata
    q = np.minimum((rankdata(S.deg) / (len(S.deg) + 1) * bins).astype(int), bins - 1)
    lo, hi = np.minimum(q[I], q[J]), np.maximum(q[I], q[J])
    return lo * bins + hi


def stratified_auroc(y, s, strat):
    o = np.argsort(strat, kind="stable")
    st, ys, ss = strat[o], y[o], s[o]
    cut = np.flatnonzero(np.diff(st)) + 1
    num = den = 0.0
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(st)]):
        npos = int(ys[a:b].sum())
        if npos >= 10 and (b - a - npos) >= 10:
            num += npos * roc_auc_score(ys[a:b], ss[a:b]); den += npos
    return float(num / den) if den else float("nan")


def lexical_flags(S, ent, years, T, ratio_cut=3.0):
    from collections import Counter
    Mn = pd.read_parquet(OUT / f"mentions{CACHE_TAG}.parquet", columns=["doc", "e", "mention"])
    Mn = Mn[years[Mn.doc.values] <= T]
    vc = Mn.groupby(["e", "mention"]).size().reset_index(name="n").sort_values(["e", "n"], ascending=[True, False])
    top = vc.drop_duplicates("e").set_index("e").mention
    tokrx = re.compile(r"[a-z0-9α-ωµ]+")
    names = top.reindex(np.arange(S.E)).fillna("").values
    toks = [tuple(tokrx.findall(n)) for n in names]
    tgt = {}
    for c in S.nodes:
        if toks[c] and len(toks[c]) <= 6:
            tgt.setdefault(len(toks[c]), set()).add(toks[c])
    D = pd.read_parquet(OUT / "docs.parquet", columns=["title", "abstract"])
    texts = (D.title.fillna("") + " " + D.abstract.fillna("")).values[years <= T]
    cnt = Counter()
    for tx in texts:
        tk = tokrx.findall(tx.lower())
        for L, st in tgt.items():
            if L <= len(tk):
                cnt.update(set(zip(*(tk[i:] for i in range(L)))) & st)
    n_ann = np.asarray(S.n, float)
    ratio = np.array([cnt.get(toks[c], 0) for c in S.nodes]) / np.maximum(n_ann, 1)
    key = np.array(["".join(toks[c]) for c in S.nodes])
    kc = pd.Series(key).value_counts()
    ambiguous = (pd.Series(key).map(kc).values >= 2) & (key != "")
    generic = ratio >= ratio_cut
    bad = np.zeros(S.E, bool)
    bad[S.nodes[generic | ambiguous]] = True
    pad = np.array([" " + " ".join(t) + " " if t else "" for t in toks], dtype=object)
    return {"bad": bad, "pad": pad, "generic": generic, "ambiguous": ambiguous, "ratio": ratio, "names": names}


def contains_pair(pad, a, b):
    pa, pb = pad[a], pad[b]
    return np.fromiter(((x != "" and y != "" and (x in y or y in x)) for x, y in zip(pa, pb)), bool, len(a))


def clean_frame(F, Z, lex):
    keep = ~lex["bad"][F.a.values] & ~lex["bad"][F.b.values] & ~contains_pair(lex["pad"], F.a.values, F.b.values)
    F2 = F[keep].copy()
    F2.attrs.update(F.attrs)
    return F2, Z[keep]


def rule_check(S, ent, years, pdoc, T, d):
    Mn = pd.read_parquet(OUT / f"mentions{CACHE_TAG}.parquet")
    Mt = Mn[years[Mn.doc.values] <= T]
    vc = Mt.groupby(["e", "mention"]).size().reset_index(name="n").sort_values(["e", "n"], ascending=[True, False])
    top = vc.drop_duplicates("e").set_index("e").mention.reindex(np.arange(S.E)).fillna("")
    key = top.str.replace(r"[^a-z0-9]", "", regex=True).values
    key = np.where(key == "", ent.entity.values, key)
    syn_t = pd.factorize(pd.Series(key))[0]
    nodes = S.nodes
    g = pd.DataFrame({"all": S.syn[nodes], "t": syn_t[nodes]})
    c2 = lambda s: int((s * (s - 1) // 2).sum())
    same_all, same_t, same_both = c2(g["all"].value_counts()), c2(g["t"].value_counts()), c2(g.value_counts())
    fc = future_counts(pdoc, years, T, T + d)
    fc = fc[fc >= POS_DOCS]
    a = fc.index.get_level_values(0).values.astype(np.int64); b = fc.index.get_level_values(1).values.astype(np.int64)
    inV = np.zeros(S.E, bool); inV[nodes] = True
    ok = inV[a] & inV[b] & ~np.isin(a * S.E + b, S.known)
    lost = int((ok & (S.syn[a] == S.syn[b]) & (syn_t[a] != syn_t[b])).sum())
    extra = int((ok & (S.syn[a] != S.syn[b]) & (syn_t[a] == syn_t[b])).sum())
    qs = Mt[Mt.type == "Disease"].groupby("e").qhit.mean()
    excl_t = set(qs[qs >= 0.5].index) | set(np.flatnonzero(ent.entity.isin(ALWAYS_EXCLUDED).values))
    excl_all = set(np.flatnonzero(ent.masld_family.values))
    only_all = np.array(sorted(excl_all - excl_t), dtype=np.int64)
    only_t = np.array(sorted(excl_t - excl_all), dtype=np.int64)
    pos_ok = ok & (S.syn[a] != S.syn[b])
    add_ub = int((~np.isin(a * S.E + b, S.known) & ((np.isin(a, only_all) & inV[b]) | (np.isin(b, only_all) & inV[a]))).sum())
    rem = int((pos_ok & (np.isin(a, only_t) | np.isin(b, only_t))).sum())
    out_extra = {"masld_rule_positives_added_upper_bound": add_ub, "masld_rule_positives_removed": rem,
                 "positives_total": int(pos_ok.sum())}
    return {**out_extra,"same_name_pairs_all_years": same_all, "same_name_pairs_up_to_T": same_t, "same_name_pairs_both": same_both,
            "positives_removed_by_all_years_rule_only": lost, "positives_removed_by_time_respecting_rule_only": extra,
            "masld_family_all_years": len(excl_all), "masld_family_up_to_T": len(excl_t),
            "masld_family_differences": sorted(ent.entity.values[list(excl_all ^ excl_t)].tolist())}


def state_forecast(states, years, M, emb, rng, T=2022, d=3):
    from scipy.stats import spearmanr
    from sklearn.linear_model import PoissonRegressor
    from sklearn.metrics import d2_tweedie_score
    S = states[T]
    upto = years <= T
    X = emb[upto].astype(np.float32)
    mu = X.mean(0)
    w_, U = np.linalg.eigh(np.cov((X - mu).T))
    idx = np.argsort(w_)[::-1][:K_WHITE]
    Et = (emb.astype(np.float32) - mu) @ (U[:, idx] / np.sqrt(np.maximum(w_[idx], 1e-8))).astype(np.float32)
    t = T + d
    wt = np.where(years <= t, np.exp(-(t - years) / TAU), 0.0).astype(np.float32)
    Hr = np.asarray(M[:, S.nodes].T @ sp.diags(wt) @ Et)
    Hr /= np.maximum(np.linalg.norm(Hr, axis=1, keepdims=True), 1e-12)
    new = np.asarray(M[(years > T) & (years <= t)][:, S.nodes].sum(0)).ravel() > 0
    P = S.H + (d / 2) * S.Vel
    P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    cp, c0 = (P * Hr).sum(1)[new], (S.H * Hr).sum(1)[new]
    bd = [np.mean((cp - c0)[ix]) for ix in (rng.integers(0, len(cp), len(cp)) for _ in range(500))]
    sem = {"entities": int(new.sum()), "cosine_projection": float(cp.mean()), "cosine_persistence": float(c0.mean()),
           "difference": float((cp - c0).mean()), "difference_ci": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
           "share_projection_closer": float((cp > c0).mean())}

    def node_x(St):
        tp = np.stack([(St.types == ty).astype(float) for ty in TYPES], 1)
        return np.column_stack([np.log1p(St.n), St.g, np.log1p(St.deg), tp])

    def target(St, tt):
        return np.asarray(M[(years > tt) & (years <= tt + d)][:, St.nodes].sum(0)).ravel()
    tts = [tt for tt in range(TRAIN_FIRST, T - d + 1) if tt in states]
    Xtr = np.vstack([node_x(states[tt]) for tt in tts])
    ytr = np.concatenate([target(states[tt], tt) for tt in tts])
    pr = PoissonRegressor(alpha=1e-4, max_iter=3000).fit(Xtr, ytr)
    yte = target(S, T)
    pred = pr.predict(node_x(S))
    n2 = np.asarray(M[years <= T - 2].sum(0)).ravel()[S.nodes]
    pers = np.maximum((S.n - n2) / 2 * d, 0.1)
    att = {"entities": int(len(yte)), "spearman_model": float(spearmanr(pred, yte)[0]), "spearman_persistence": float(spearmanr(pers, yte)[0]),
           "d2_model": float(d2_tweedie_score(yte, pred, power=1)), "d2_persistence": float(d2_tweedie_score(yte, pers, power=1)),
           "observed_new_papers": int(yte.sum()), "predicted_new_papers": float(pred.sum())}
    return {"semantic": sem, "attention": att}


def amendment_a8(res, frames, states, fitted, years, ent, M, pdoc, emb, keep, args):
    rng8 = np.random.default_rng(RNG8)
    t0 = time.time()
    T, d = PRIMARY
    F, Z = frames[T]
    y, w = F[f"y{d}"].values, F[f"w{d}"].values
    m = w > 0
    s = keep["s"]
    tr = {t: frames[t] for t in sorted(frames) if t >= TRAIN_FIRST and t + d <= T}
    draws = 50 if DEV else 200
    out = {"description": "analyses added in amendment A8 (2026-11-18), after review; separate random generator"}

    base = fit_baselines(list(tr.values()), d, rng8)
    sb = score_baselines(base, F, Z, d)
    allsc = {**s, **sb}
    ev = {k: evaluate(y, w, v, F.raw_pa.values) for k, v in sb.items()}
    names = list(sb)
    bt = node_bootstrap2(F[m], d, {k: allsc[k][m] for k in ["neural_world_model", "resource_allocation"] + names}, rng8, draws,
                         contrasts=[("neural_world_model", k) for k in names], popularity=["neural_world_model"] + names,
                         pop_contrasts=[("neural_world_model", k) for k in names])
    for k in names:
        ev[k]["auroc_ci"] = bt["auroc_ci"][k]
        ev[k]["auroc_popularity_adjusted_ci"] = bt["popadj_ci"][k]
    out["baselines"] = {"metrics": ev, "contrasts": bt["diff_ci"], "contrasts_popularity_adjusted": bt["popadj_diff_ci"],
                        "draws": bt["draws"]}
    log("w3.a8.baselines", **{k: (round(v["auroc"], 4), round(v["auroc_popularity_adjusted"], 4)) for k, v in ev.items()},
        secs=round(time.time() - t0))

    enc = {}
    for e_name in ("random", "neuml"):
        e2 = np.load(OUT / f"emb_{e_name}.npy").astype(np.float32)
        fr2 = refeaturize({**tr, T: frames[T]}, lambda t, e2=e2: State(t, years, ent, M, pdoc, e2))
        mo = fit_all([fr2[t] for t in tr], d, rng8, logistic=False)
        F2, Z2 = fr2[T]
        s2 = scores(mo, F2, Z2, d)
        enc[e_name] = {"scores": {"neural_world_model": s2["neural_world_model"], "hazard_world_model": s2["hazard_world_model"],
                                  "semantic": s2["semantic"]}}
        del fr2, e2
    sd = {"neural_world_model": s["neural_world_model"], "semantic": s["semantic"], "hazard_world_model": s["hazard_world_model"]}
    for e_name, v in enc.items():
        for k, sc in v["scores"].items():
            sd[f"{k}__{e_name}"] = sc
    cons = [("neural_world_model", f"neural_world_model__{e}") for e in enc] + [("semantic", f"semantic__{e}") for e in enc]
    bt = node_bootstrap2(F[m], d, {k: v[m] for k, v in sd.items()}, rng8, draws, contrasts=cons,
                         popularity=["neural_world_model"] + [f"neural_world_model__{e}" for e in enc],
                         pop_contrasts=[("neural_world_model", f"neural_world_model__{e}") for e in enc])
    out["encoders_paired"] = {"metrics": {k: {**{q: evaluate(y, w, v, F.raw_pa.values)[q] for q in ("auroc", "auroc_popularity_adjusted", "ap")},
                                              "auroc_ci": bt["auroc_ci"][k], **({"auroc_popularity_adjusted_ci": bt["popadj_ci"][k]} if k in bt["popadj_ci"] else {})}
                                          for k, v in sd.items()},
                              "contrasts": bt["diff_ci"], "contrasts_popularity_adjusted": bt["popadj_diff_ci"], "draws": bt["draws"]}
    log("w3.a8.encoders", **{k: round(v["auroc"], 4) for k, v in out["encoders_paired"]["metrics"].items()}, secs=round(time.time() - t0))

    ab = {"none__neural_world_model": s["neural_world_model"], "none__hazard_world_model": s["hazard_world_model"]}
    for (blk, k), sc in keep["abl_scores"].items():
        ab[f"{blk}__{k}"] = sc
    ab["all_language_features__neural_world_model"] = sb["no_language_neural"]
    ab["all_language_features__hazard_world_model"] = sb["no_language_hazard"]
    abl_keys = [k for k in ab if not k.startswith("none__")]
    con = [(f"none__{k.split('__')[1]}", k) for k in abl_keys]
    bt = node_bootstrap2(F[m], d, {k: v[m] for k, v in ab.items()}, rng8, draws, contrasts=con, popularity=list(ab), pop_contrasts=con)
    out["ablation_intervals"] = {"auroc_change_ci": {b: bt["diff_ci"][f"{a}_minus_{b}"] for a, b in con},
                                 "popadj_change_ci": {b: bt["popadj_diff_ci"][f"{a}_minus_{b}"] for a, b in con},
                                 "all_language_features": {k: {q: evaluate(y, w, ab[k], F.raw_pa.values)[q] for q in ("auroc", "auroc_popularity_adjusted")}
                                                           for k in ab if k.startswith("all_language")},
                                 "note": "intervals are for (full model) minus (model without the block)", "draws": bt["draws"]}
    log("w3.a8.ablation", secs=round(time.time() - t0))

    gi = []
    for (Tt, dd), mods in fitted.items():
        Ft, Zt = frames[Tt]
        st = scores(mods, Ft, Zt, dd)
        mt = Ft[f"w{dd}"].values > 0
        two = ("neural_world_model", "resource_allocation")
        b2 = node_bootstrap2(Ft[mt], dd, {k: st[k][mt] for k in two}, rng8, draws, contrasts=(two,), popularity=two, pop_contrasts=(two,))
        gi.append({"cutoff": Tt, "horizon": dd, "prespecified_task": bool(Tt in PRESPECIFIED_CUTOFFS),
                   "neural_auroc_ci": b2["auroc_ci"]["neural_world_model"], "ra_auroc_ci": b2["auroc_ci"]["resource_allocation"],
                   "difference_ci": b2["diff_ci"]["neural_world_model_minus_resource_allocation"],
                   "neural_popadj_ci": b2["popadj_ci"]["neural_world_model"], "ra_popadj_ci": b2["popadj_ci"]["resource_allocation"],
                   "popadj_difference_ci": b2["popadj_diff_ci"]["neural_world_model_minus_resource_allocation"]})
    out["grid_intervals"] = gi
    pd.DataFrame([{k: (v if not isinstance(v, list) else f"{v[0]:.4f} to {v[1]:.4f}") for k, v in row.items()} for row in gi]).to_csv(
        SRC / f"wm_grid_intervals_{TAG}.csv", index=False)
    pm = {**res["primary"]["metrics"], **out["baselines"]["metrics"]}
    pd.DataFrame([{"method": k, "auroc": v["auroc"], "auroc_ci_low": v["auroc_ci"][0], "auroc_ci_high": v["auroc_ci"][1],
                   "auroc_popularity_adjusted": v["auroc_popularity_adjusted"],
                   "popularity_adjusted_ci_low": v.get("auroc_popularity_adjusted_ci", [np.nan, np.nan])[0],
                   "popularity_adjusted_ci_high": v.get("auroc_popularity_adjusted_ci", [np.nan, np.nan])[1],
                   "average_precision": v["ap"], "n_pos": v["n_pos"], "n_neg_sampled": v["n_neg_sampled"]} for k, v in pm.items()]).to_csv(
        SRC / f"wm_primary_metrics_{TAG}.csv", index=False)
    log("w3.a8.grid", secs=round(time.time() - t0))

    F25, Z25, s25 = keep["F25"], keep["Z25"], keep["s25"]
    base1 = fit_baselines(keep["tr1"], 1, rng8)
    sb25 = score_baselines(base1, F25, Z25, 1)
    all25 = {**s25, **sb25}
    m25 = F25.w1.values > 0
    near = res["near_prospective"]["metrics"]
    for k, v in sb25.items():
        near[k] = evaluate(F25.y1.values, F25.w1.values, v, F25.raw_pa.values)
    ks25 = ["neural_world_model", "hazard_world_model", "resource_allocation", "semantic", "degree_product",
            "logistic_ra_semantic"] + list(sb25)
    bt = node_bootstrap2(F25[m25], 1, {k: all25[k][m25] for k in ks25}, rng8, draws,
                         contrasts=[("neural_world_model", k) for k in ks25 if k != "neural_world_model"],
                         popularity=ks25, pop_contrasts=[("neural_world_model", k) for k in ks25 if k != "neural_world_model"])
    for k in ks25:
        near[k]["auroc_ci"] = bt["auroc_ci"][k]
        near[k]["auroc_popularity_adjusted_ci"] = bt["popadj_ci"][k]
    res["near_prospective"]["contrasts"] = bt["diff_ci"]
    res["near_prospective"]["contrasts_popularity_adjusted"] = bt["popadj_diff_ci"]
    res["near_prospective"]["bootstrap_draws"] = bt["draws"]
    log("w3.a8.near", secs=round(time.time() - t0))

    Fh = keep["holdout"]
    hs = {"resource_allocation": Fh.raw_ra.values.astype(float), "semantic": Fh.cos.values.astype(float),
          "degree_product": Fh.raw_pa.values.astype(float)}
    bt = node_bootstrap2(Fh, "H", hs, rng8, draws)
    res["design_contrast"]["random_holdout_ci"] = bt["auroc_ci"]
    res["design_contrast"]["temporal_primary_ci"] = {k: res["primary"]["metrics"][k]["auroc_ci"]
                                                     for k in res["design_contrast"]["temporal_primary"]}
    res["design_contrast"]["n_holdout_positives"] = int(Fh.yH.sum())

    S = states[T]
    mods = fitted[(T, d)]
    fns = {"neural_world_model": lambda F_, Z_: mods["neural"].eta(xmat(F_, d, BASE_X), Z_),
           "hazard_world_model": lambda F_, Z_: mods["hazard"].eta(xmat(F_, d, BASE_X), Z_),
           "no_language_neural": lambda F_, Z_: base["no_language_neural"][0].eta(xmat(F_, d, NOLANG), np.zeros_like(Z_)),
           "gbdt_no_language": lambda F_, Z_: base["gbdt_no_language"].eta(F_, d)}
    I, J, ex, raw = exact_scores(S, fns)
    ex.update({"resource_allocation": raw["raw_ra"], "degree_product": raw["raw_pa"], "semantic": raw["cos"]})
    nV = len(S.nodes)
    pos = F.key.values[F[f"y{d}"].values == 1]
    plin = S.posarr[pos // S.E] * nV + S.posarr[pos % S.E]
    ye = np.isin(I.astype(np.int64) * nV + J, plin).astype(np.int8)
    ec = {"n_candidates": int(len(ye)), "n_positives": int(ye.sum()), "n_candidates_check": int(S.n_candidates()), "metrics": {}}
    rocs, prs = [], []
    strat_fine = degree_matched_strata(S, I, J)
    for k, sc in ex.items():
        mt = exact_metrics(ye, sc.astype(np.float64))
        mt["auroc_degree_matched"] = stratified_auroc(ye, sc, strat_fine)
        mt.update({f"entity_{q}": v for q, v in entity_centred(I, J, ye, sc, nV).items()})
        ec["metrics"][k] = mt
        g, tp = roc_points(ye, sc)
        rocs += [{"method": k, "fpr": a_, "tpr": b_} for a_, b_ in zip(g, tp)]
        kk, pp, rr = pr_points(ye, sc)
        prs += [{"method": k, "k": int(a_), "precision": b_, "recall": c_} for a_, b_, c_ in zip(kk, pp, rr)]
    ec["topk_ci"] = topk_bootstrap(I, J, ye, {k: ex[k] for k in ["neural_world_model", "hazard_world_model", "resource_allocation",
                                                               "no_language_neural", "gbdt_no_language", "degree_product", "semantic"]},
                                   (100, 1000, 10000), rng8, draws, pairs=[("neural_world_model", "resource_allocation"),
                                                                            ("neural_world_model", "no_language_neural"),
                                                                            ("hazard_world_model", "resource_allocation")])
    pd.DataFrame(rocs).to_csv(SRC / f"wm_roc_exact_primary_{TAG}.csv", index=False)
    pd.DataFrame(prs).to_csv(SRC / f"wm_pr_exact_primary_{TAG}.csv", index=False)
    out["exact_primary"] = ec
    log("w3.a8.exact", n=len(ye), pos=int(ye.sum()), **{k: (round(v["auroc"], 4), round(v["precision_at_1000"], 3), round(v["auroc_degree_matched"], 3))
                                                         for k, v in ec["metrics"].items()}, secs=round(time.time() - t0))

    lex = lexical_flags(S, ent, years, T)
    frc = {t: clean_frame(*frames[t], lex) for t in tr}
    Fc, Zc = clean_frame(F, Z, lex)
    moc = fit_all(list(frc.values()), d, rng8, logistic=False)
    scc = scores(moc, Fc, Zc, d)
    yc, wc = Fc[f"y{d}"].values, Fc[f"w{d}"].values
    mc = wc > 0
    two = ("neural_world_model", "resource_allocation")
    bt = node_bootstrap2(Fc[mc], d, {k: scc[k][mc] for k in two}, rng8, draws, contrasts=(two,), popularity=two, pop_contrasts=(two,))
    pos_flag = (lex["bad"][F.a.values] | lex["bad"][F.b.values] | contains_pair(lex["pad"], F.a.values, F.b.values))[y == 1]
    out["cleaned"] = {"rule": "entities whose most frequent mention up to T occurs in >=3 times as many papers as it is annotated "
                              "(generic words) or is shared by another entity (ambiguous), and pairs whose names contain each other",
                      "nodes_flagged_generic": int(lex["generic"].sum()), "nodes_flagged_ambiguous": int(lex["ambiguous"].sum()),
                      "nodes_flagged": int((lex["generic"] | lex["ambiguous"]).sum()),
                      "nodes": int(len(S.nodes)), "positives_flagged": int(pos_flag.sum()), "positives": int(len(pos_flag)),
                      "positives_retained": int(yc.sum()),
                      "metrics": {k: {**evaluate(yc, wc, v, Fc.raw_pa.values), **({"auroc_ci": bt["auroc_ci"][k],
                                      "auroc_popularity_adjusted_ci": bt["popadj_ci"][k]} if k in two else {})}
                                  for k, v in scc.items()},
                      "contrast": bt["diff_ci"], "contrast_popularity_adjusted": bt["popadj_diff_ci"]}
    out["rule_check"] = rule_check(S, ent, years, pdoc, T, d)
    log("w3.a8.cleaned", flagged=int((lex["generic"] | lex["ambiguous"]).sum()), pos_flagged=int(pos_flag.sum()),
        **{k: round(v["auroc"], 4) for k, v in out["cleaned"]["metrics"].items()}, secs=round(time.time() - t0))

    out["state_forecast"] = state_forecast(states, years, M, emb, rng8, T, d)
    log("w3.a8.state", **{k: {q: (round(x, 4) if isinstance(x, float) else x) for q, x in v.items() if not isinstance(x, list)}
                          for k, v in out["state_forecast"].items()}, secs=round(time.time() - t0))

    S25 = keep["S25"]
    tr3 = [frames[t] for t in sorted(frames) if t >= TRAIN_FIRST and t + 3 <= 2025]
    m3 = fit_all(tr3, 3, rng8, logistic=False)
    m1 = keep["m1"]
    fns = {"neural_world_model": lambda F_, Z_: m1["neural"].eta(xmat(F_, 1, BASE_X), Z_),
           "hazard_world_model": lambda F_, Z_: m1["hazard"].eta(xmat(F_, 1, BASE_X), Z_),
           "no_language_neural": lambda F_, Z_: base1["no_language_neural"][0].eta(xmat(F_, 1, NOLANG), np.zeros_like(Z_)),
           "forecast_eta_3y": lambda F_, Z_: m3["neural"].eta(xmat(F_, 3, BASE_X), Z_)}
    I, J, ex, raw = exact_scores(S25, fns)
    ex.update({"resource_allocation": raw["raw_ra"], "degree_product": raw["raw_pa"], "semantic": raw["cos"]})
    nV = len(S25.nodes)
    pos = F25.key.values[F25.y1.values == 1]
    lin = I.astype(np.int64) * nV + J
    ye = np.isin(lin, S25.posarr[pos // S25.E] * nV + S25.posarr[pos % S25.E]).astype(np.int8)
    eh = {"n_candidates": int(len(ye)), "n_positives": int(ye.sum()), "metrics": {}}
    rocs = []
    for k in ["neural_world_model", "hazard_world_model", "no_language_neural", "resource_allocation", "degree_product", "semantic"]:
        eh["metrics"][k] = exact_metrics(ye, ex[k].astype(np.float64))
        g, tp = roc_points(ye, ex[k])
        rocs += [{"method": k, "fpr": a_, "tpr": b_} for a_, b_ in zip(g, tp)]
    eh["topk_ci"] = topk_bootstrap(I, J, ye, {k: ex[k] for k in ["neural_world_model", "resource_allocation", "degree_product"]},
                                   (100, 1000), rng8, draws, pairs=[("neural_world_model", "resource_allocation")])
    pd.DataFrame(rocs).to_csv(SRC / f"wm_roc_exact_holdout_{TAG}.csv", index=False)
    out["exact_holdout"] = eh
    log("w3.a8.exact_holdout", n=len(ye), pos=int(ye.sum()), **{k: (round(v["auroc"], 4), round(v["precision_at_1000"], 4))
                                                              for k, v in eh["metrics"].items()}, secs=round(time.time() - t0))
    res["a8"] = out


if __name__ == "__main__":
    main()
