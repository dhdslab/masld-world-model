from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import rankdata

import w3_world_model as W
from common import TOPICS_ANA, CORPUS_PROC, REF_RUN, SRC, log, update_results

T, D = W.PRIMARY
MIN_LINKED = 3
MIN_POS_DOMAIN = 30
SEED = W.WM_SEED + 10
INPUTS = ["cos", "triad", "proj3", "conv", "drift", "n_lo", "n_hi", "g_lo", "g_hi", "cn", "ra", "aa", "jac", "pa"]
BLOCK = {"cos": "semantic", "triad": "semantic", "proj3": "dynamics", "conv": "dynamics", "drift": "drift",
         "n_lo": "attention", "n_hi": "attention", "g_lo": "attention", "g_hi": "attention",
         "cn": "graph", "ra": "graph", "aa": "graph", "jac": "graph", "pa": "graph"}
METRICS = ["attention", "growth", "connectivity", "speed", "drift_alignment", "coherence", "distinctiveness"]


def auroc(y, s):
    y = np.asarray(y, bool)
    r = rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def domain_order():
    Z = np.load(TOPICS_ANA / f"{REF_RUN}_linkage_ward.npy")
    cat = pd.read_csv(TOPICS_ANA / "topic_catalogue_named.csv")
    r = cat[~cat.meta_topic].reset_index(drop=True)
    leaves = dendrogram(Z, no_plot=True)["leaves"]
    order = list(dict.fromkeys(r.domain_name.values[leaves]))
    return order, r, Z, leaves


def doc_domains(order):
    docs = pd.read_parquet(W.OUT / "docs.parquet", columns=["pmid", "year"])
    lit = pd.read_parquet(CORPUS_PROC / "literature_included_final.parquet", columns=["record_id", "pmid"]).dropna(subset=["pmid"])
    ti = pd.read_parquet(TOPICS_ANA / f"{REF_RUN}_topics_items.parquet", columns=["stream", "item_id", "topic"])
    ti = ti[ti.stream == "literature"]
    cat = pd.read_csv(TOPICS_ANA / "topic_catalogue_named.csv")
    t2d = cat[~cat.meta_topic].set_index("topic").domain_name
    m = lit.merge(ti, left_on="record_id", right_on="item_id")
    m = m[m.topic.isin(t2d.index)]
    m["dom"] = m.topic.map(t2d).map({d: i for i, d in enumerate(order)})
    m["pmid"] = m.pmid.astype(str).str.strip()
    m = m.drop_duplicates("pmid")
    key = docs.pmid.astype("int64").astype(str)
    dom = key.map(m.set_index("pmid").dom).fillna(-1).astype(int).values
    return dom, docs.year.values, {"linked_documents": int((dom >= 0).sum()), "documents": int(len(dom)),
                                   "evidence_map_papers_with_domain": int(len(m))}


def entity_domains(S, M, years, ddom, nd):
    keep = (ddom >= 0) & (years <= T)
    rows = np.flatnonzero(keep)
    Ms = M[rows][:, S.nodes].tocsc()
    cnt = np.zeros((len(S.nodes), nd))
    for k in range(nd):
        sel = ddom[rows] == k
        cnt[:, k] = np.asarray(Ms[sel].sum(0)).ravel()
    tot = cnt.sum(1)
    p = cnt.sum(0) / cnt.sum()
    expct = np.maximum(tot[:, None] * p, 1e-12)
    resid = (cnt - expct) / np.sqrt(expct)
    best = np.argmax(resid, 1)
    dom = np.where(tot >= MIN_LINKED, best, -1)
    share = np.where(tot > 0, cnt[np.arange(len(tot)), best] / np.maximum(tot, 1), np.nan)
    return dom, tot, share, p


def state_metrics(S, ref):
    H = S.H
    hbar = H.mean(0); hbar = hbar / np.linalg.norm(hbar)
    u = S.u / np.linalg.norm(S.u)
    met = pd.DataFrame({
        "attention": np.log1p(S.n),
        "growth": S.g,
        "connectivity": np.log1p(S.deg),
        "speed": np.linalg.norm(S.Vel, axis=1),
        "drift_alignment": H @ u,
        "coherence": (H * S.Cn).sum(1),
        "distinctiveness": 1.0 - H @ hbar})
    m = met[ref]
    z = (met - m.mean()) / m.std(ddof=0)
    return met, z


def polar(S, dom, met, order):
    nd = len(order)
    n_d = np.array([(dom == k).sum() for k in range(nd)])
    gap = np.deg2rad(2.0)
    span = 2 * np.pi - nd * gap
    start = np.concatenate([[0.0], np.cumsum(span * n_d / n_d.sum() + gap)])[:-1]
    theta = np.full(len(dom), np.nan)
    for k in range(nd):
        ix = np.flatnonzero(dom == k)
        if len(ix) == 0:
            continue
        Hk = S.H[ix] - S.H[ix].mean(0)
        pc = Hk @ np.linalg.svd(Hk, full_matrices=False)[2][0] if len(ix) > 1 else np.zeros(1)
        rank = (rankdata(pc) - 0.5) / len(ix)
        theta[ix] = start[k] + rank * span * n_d[k] / n_d.sum()
    sectors = pd.DataFrame({"domain": order, "entities": n_d, "theta_start": start,
                            "theta_end": start + span * n_d / n_d.sum()})
    return theta, sectors


def load_frame(t):
    F = pd.read_parquet(W.OUT / "cache" / f"frame_{t}.parquet")
    Z = np.load(W.OUT / "cache" / f"frameZ_{t}.npy").astype(np.float32)
    return F, Z


def main():
    years, ent, M, pdoc = W.prepare()
    emb = np.load(W.OUT / "emb_biomedbert.npy").astype(np.float32)
    S = W.State(T, years, ent, M, pdoc, emb)
    order, topics, Zlink, leaves = domain_order()
    nd = len(order)
    ddom, dyear, link = doc_domains(order)
    dom, nlinked, purity, dshare = entity_domains(S, M, dyear, ddom, nd)
    met, z = state_metrics(S, dom >= 0)
    theta, sectors = polar(S, dom, met, order)

    E = pd.DataFrame({"entity": ent.entity.values[S.nodes], "name": ent.name.values[S.nodes], "type": S.types,
                      "domain": [order[k] if k >= 0 else "" for k in dom], "linked_papers_to_2022": nlinked.astype(int),
                      "domain_share": np.round(purity, 4), "theta": theta, "n_papers": S.n.astype(int)})
    E = pd.concat([E, met.add_prefix("raw_"), z.round(4).add_prefix("z_")], axis=1)
    E.to_csv(SRC / "atlas_entity_states.csv", index=False)
    prof = z.groupby(dom).mean().reindex(range(nd))
    prof.index = order
    prof["entities"] = sectors.entities.values
    prof.round(4).rename_axis("domain").reset_index().to_csv(SRC / "atlas_domain_profiles.csv", index=False)

    cen = np.vstack([S.H[dom == k].mean(0) for k in range(nd)])
    cen = cen / np.linalg.norm(cen, axis=1, keepdims=True)
    prox = pd.DataFrame(cen @ cen.T, index=order, columns=order)
    prox.round(4).rename_axis("domain").reset_index().to_csv(SRC / "atlas_domain_proximity.csv", index=False)

    F, Zo = load_frame(T)
    y = F[f"y{D}"].values
    neg = F.in_sample.values & (y == 0)
    ev = (y == 1) | neg
    edom = np.full(len(ent), -1)
    edom[S.nodes] = dom
    da, db = edom[F.a.values], edom[F.b.values]
    within = (da >= 0) & (da == db)
    both = (da >= 0) & (db >= 0)
    ts = [t for t in range(W.TRAIN_FIRST, W.LAST_FULL) if t + D <= T]
    frames = {t: load_frame(t) for t in ts}
    hz = W.fit_all([frames[t] for t in ts], D, np.random.default_rng(SEED + 1), neural=True, glm=False, logistic=False)
    wm = hz["neural"].eta(W.xmat(F, D, W.BASE_X), Zo)
    del frames
    cols = {**{c: F[c].values for c in INPUTS}, "world_model": wm}
    rows = []
    groups = [("all", ev)] + [(order[k], ev & ((da == k) | (db == k))) for k in range(nd)]
    for g, m in groups:
        npos, nneg = int((y[m] == 1).sum()), int((y[m] == 0).sum())
        for c, v in cols.items():
            a = auroc(y[m] == 1, v[m]) if npos >= (1 if g == "all" else MIN_POS_DOMAIN) and nneg > 0 else np.nan
            rows.append({"group": g, "input": c, "block": BLOCK.get(c, "world model"), "auroc": a,
                         "positives": npos, "negatives": nneg})
    A = pd.DataFrame(rows)
    A.round(4).to_csv(SRC / "atlas_pair_input_auroc.csv", index=False)

    mp, mn = (y == 1) & both, neg & both
    share_within_pos = float((within[mp]).mean())
    share_within_neg = float((within[mn]).mean())
    mix = pd.crosstab(pd.Categorical(np.minimum(da, db)[mp], range(nd)), pd.Categorical(np.maximum(da, db)[mp], range(nd)),
                      dropna=False)
    mix.index, mix.columns = order, order
    mix.rename_axis("domain").reset_index().to_csv(SRC / "atlas_new_association_domain_pairs.csv", index=False)

    dyn = pd.read_csv(SRC / "topic_dynamics.csv")[["topic", "trend", "rgr"]]
    rings = topics[["topic", "topic_label", "domain_name", "n_assigned", "n_core", "mean_sim"]].merge(dyn, on="topic", how="left")
    rings["leaf_position"] = pd.Series(np.arange(len(leaves)), index=np.array(leaves)).reindex(range(len(rings))).values
    rings.sort_values("leaf_position").to_csv(SRC / "atlas_topic_rings.csv", index=False)
    pd.DataFrame(Zlink, columns=["left", "right", "height", "size"]).to_csv(SRC / "atlas_ward_linkage.csv", index=False)

    a_all = A[A.group == "all"].set_index("input").auroc
    single = a_all.drop("world_model")
    dA = A[(A.group != "all") & A.auroc.notna()]
    dom_wm = dA[dA.input == "world_model"].set_index("group")
    dom_best_single = dA[dA.input != "world_model"].sort_values("auroc", ascending=False).groupby("group").head(1).set_index("group")
    wm_beats = int(sum(dom_wm.auroc[g] > dom_best_single.auroc[g] for g in dom_wm.index))
    sem_block = dA[dA.input.isin(["cos", "triad", "proj3"])].groupby("group").auroc.max()
    res = {
        "cutoff": T, "horizon": D, "min_linked_papers": MIN_LINKED, "min_positives_domain": MIN_POS_DOMAIN,
        "domains_order": order, "linking": link,
        "entities": int(len(S.nodes)), "entities_with_domain": int((dom >= 0).sum()),
        "entities_with_domain_share": float((dom >= 0).mean()),
        "entity_domain_median_share": float(np.nanmedian(purity[dom >= 0])),
        "domain_paper_share_to_cutoff": dict(zip(order, np.round(dshare, 4).tolist())),
        "entities_by_domain": dict(zip(order, sectors.entities.astype(int).tolist())),
        "profiles": {d: {k: float(prof.loc[d, k]) for k in METRICS} for d in order},
        "metrics": METRICS,
        "profile_extremes": {k: {"max": str(prof[k].idxmax()), "max_value": float(prof[k].max()),
                                 "min": str(prof[k].idxmin()), "min_value": float(prof[k].min())} for k in METRICS},
        "proximity_offdiag_range": [float(prox.values[~np.eye(nd, dtype=bool)].min()), float(prox.values[~np.eye(nd, dtype=bool)].max())],
        "proximity_closest_pair": list(prox.where(~np.eye(nd, dtype=bool)).stack().idxmax()),
        "proximity_closest_value": float(prox.where(~np.eye(nd, dtype=bool)).stack().max()),
        "proximity_farthest_pair": list(prox.where(~np.eye(nd, dtype=bool)).stack().idxmin()),
        "proximity_farthest_value": float(prox.where(~np.eye(nd, dtype=bool)).stack().min()),
        "domain_input_auroc": {g: {c: float(v) for c, v in A[A.group == g].set_index("input").auroc.items()} for g in order},
        "most_studied_by_domain": {d: str(E[E.domain == d].sort_values("n_papers", ascending=False).name.iloc[0]) for d in order},
        "most_studied_distance_percentile": {d: float((met.distinctiveness.values < met.distinctiveness.values[
            E[E.domain == d].sort_values("n_papers", ascending=False).index[0]]).mean()) for d in order},
        "attention_distance_spearman": float(met.attention.corr(met.distinctiveness, method="spearman")),
        "single_input_rank_spearman_min": float(min(
            pd.Series(A[A.group == g].set_index("input").auroc.drop("world_model")).corr(single, method="spearman")
            for g in order)),
        "pair_auroc_all": {k: float(v) for k, v in a_all.items()},
        "pair_auroc_best_single": str(single.idxmax()), "pair_auroc_best_single_value": float(single.max()),
        "pair_auroc_worst_single": str(single.idxmin()), "pair_auroc_worst_single_value": float(single.min()),
        "pair_auroc_world_model_all": float(a_all["world_model"]),
        "positives_all": int((y[ev] == 1).sum()), "negatives_all": int((y[ev] == 0).sum()),
        "domains_reported": dom_wm.index.tolist(),
        "domain_positives": {g: int(A[(A.group == g)].positives.iloc[0]) for g in order},
        "domain_world_model_auroc": {g: float(v) for g, v in dom_wm.auroc.items()},
        "domain_world_model_auroc_range": [float(dom_wm.auroc.min()), float(dom_wm.auroc.max())],
        "domain_best_single": {g: {"input": str(r.input), "auroc": float(r.auroc)} for g, r in dom_best_single.iterrows()},
        "domains_world_model_above_best_single": wm_beats,
        "domain_semantic_best_range": [float(sem_block.min()), float(sem_block.max())],
        "new_associations_both_assigned": int(mp.sum()),
        "new_associations_within_domain_share": share_within_pos,
        "candidate_pairs_within_domain_share": share_within_neg,
        "world_model_reestimated_check": "hazard_v1_neural of foundation_world_model.primary (same fit)",
    }
    update_results("semantic_atlas", res)
    log("w10.semantic_atlas", entities=res["entities"], with_domain=res["entities_with_domain"],
        wm_all=round(res["pair_auroc_world_model_all"], 4), best_single=res["pair_auroc_best_single"],
        domains=len(res["domains_reported"]), within_pos=round(share_within_pos, 3), within_neg=round(share_within_neg, 3))
    print(json.dumps({k: res[k] for k in ("linking", "entities", "entities_with_domain", "entities_by_domain", "pair_auroc_all",
                                           "domain_positives", "domain_world_model_auroc", "domain_best_single",
                                           "new_associations_within_domain_share", "candidate_pairs_within_domain_share",
                                           "profile_extremes", "proximity_closest_pair", "proximity_farthest_pair")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
