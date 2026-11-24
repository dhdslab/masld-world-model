from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from common import (ANA, TOPICS_ANA, TOPICS_EMB, CORPUS_PROC, ORIG_RUN, REF_RUN, SEED, SRC, log,
                    update_results)


def auroc_cv(X, y, seed=SEED):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    s = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=5)
    return {"mean": float(s.mean()), "folds": [float(v) for v in s], "n": int(len(y)), "n_pos": int(y.sum())}


def main():
    items = pd.read_parquet(TOPICS_EMB / "items.parquet", columns=["stream", "item_id", "year"])
    lit_all = pd.read_parquet(CORPUS_PROC / "literature_included_final.parquet",
                              columns=["record_id", "year", "nom_nafld", "nom_mafld", "nom_masld", "pub_format",
                                       "open_access", "fulltext_collected", "language"])
    labels = pd.read_parquet(CORPUS_PROC / "openfda_labels.parquet")
    counts = items.groupby(["stream", "year"]).size().unstack(fill_value=0)
    counts.to_csv(SRC / "corpus_stream_year_counts.csv")
    flow = __import__("json").loads((CORPUS_PROC / "flow_counts_all.json").read_text(encoding="utf-8"))
    corpus = {
        "flow": flow,
        "stream_totals": items.stream.value_counts().to_dict(),
        "fda_labels": int(len(labels)),
        "embedded_items": int(len(items)),
        "literature_by_year": lit_all.year.value_counts().sort_index().to_dict(),
        "growth_2020_2025": float((lit_all.year == 2025).sum() / (lit_all.year == 2020).sum()),
        "english_share": float((lit_all.language.fillna("").str.lower().isin(["eng", "en", "english"])).mean()),
    }

    lit = np.where(items.stream.values == "literature")[0]
    L = lit_all.set_index("record_id").loc[items.item_id.values[lit]].reset_index()
    era = np.where(L.nom_nafld & ~L.nom_masld, 0, np.where(L.nom_masld & ~L.nom_nafld, 1, -1))
    yr = L.year.values
    sel = (era >= 0) & (yr >= 2023)
    sel24 = (era >= 0) & (yr == 2024)
    out = {"n_era_labelled_2023_2026": int(sel.sum()), "n_current": int((era[sel] == 1).sum()),
           "n_legacy": int((era[sel] == 0).sum()),
           "current_share_by_year": {int(y): float((era[(era >= 0) & (yr == y)] == 1).mean()) for y in range(2020, 2027)},
           "labelled_by_year": {int(y): int(((era >= 0) & (yr == y)).sum()) for y in range(2020, 2027)},
           "term_share_all_papers_by_year": {k: {int(y): float(v) for y, v in lit_all.groupby("year")[c].mean().items()}
                                             for k, c in [("legacy", "nom_nafld"), ("interim", "nom_mafld"), ("current", "nom_masld")]}}
    t_ref = pd.read_parquet(TOPICS_ANA / f"{REF_RUN}_topics_items.parquet").topic.values[lit]
    t_orig = pd.read_parquet(TOPICS_ANA / f"{ORIG_RUN}_topics_items.parquet").topic.values[lit]
    for name, emb, topics in [("original", "pubmedbert", t_orig), ("neutral", "pubmedbert_neutral", t_ref)]:
        E = np.load(TOPICS_EMB / f"{emb}.npy", mmap_mode="r")[lit].astype(np.float32)
        r = {"era_auroc_2023_2026": auroc_cv(E[sel], era[sel]),
             "era_auroc_2024_only": auroc_cv(E[sel24], era[sel24])}
        for fam, code in [("legacy_only", 0), ("current_only", 1)]:
            m = (era == code) & np.isin(yr, [2023, 2025])
            r[f"drift_2023_vs_2025_{fam}"] = auroc_cv(E[m], (yr[m] == 2025).astype(int))
        r["nmi_topic_era"] = float(normalized_mutual_info_score(topics[sel], era[sel]))
        share = pd.Series(era[sel]).groupby(topics[sel]).mean()
        cnt = pd.Series(era[sel]).groupby(topics[sel]).size()
        ok = cnt >= 30
        r["name_segregated_topics"] = int(((share[ok] >= 0.8) | (share[ok] <= 0.2)).sum())
        r["topics_evaluated"] = int(ok.sum())
        r["topic_current_share"] = {int(k): float(v) for k, v in share[ok].items()}
        out[name] = r
        log("a1.nomenclature", run=name, auroc=round(r["era_auroc_2023_2026"]["mean"], 4),
            auroc24=round(r["era_auroc_2024_only"]["mean"], 4),
            drift_legacy=round(r["drift_2023_vs_2025_legacy_only"]["mean"], 4),
            drift_current=round(r["drift_2023_vs_2025_current_only"]["mean"], 4),
            segregated=r["name_segregated_topics"])
        del E
    rows = []
    for name, run in [("original", ORIG_RUN), ("neutral", REF_RUN)]:
        U = np.load(TOPICS_ANA / f"{run}_umap2d.npy")[lit]
        rows.append(pd.DataFrame({"map": name, "record_id": L.record_id.values[sel], "year": yr[sel],
                                  "era_current": era[sel], "x": U[sel, 0], "y": U[sel, 1]}))
    pd.concat(rows).to_csv(SRC / "nomenclature_map_points.csv", index=False)
    update_results("corpus", corpus)
    update_results("nomenclature", out)


if __name__ == "__main__":
    main()
