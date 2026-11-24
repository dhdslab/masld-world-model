import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, balanced_accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, cohen_kappa_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from ku_common import *

CLASSES = ["Guideline", "SR/MA", "RCT", "Case report", "Commentary", "Narrative review", "Primary/other"]


def nlm_label(pt):
    pt = str(pt or "")
    if any(x in pt for x in ("Practice Guideline", "Guideline", "Consensus Development Conference")): return "Guideline"
    if "Meta-Analysis" in pt or "Systematic Review" in pt: return "SR/MA"
    if "Randomized Controlled Trial" in pt: return "RCT"
    if "Case Reports" in pt: return "Case report"
    if any(x in pt for x in ("Editorial", "Comment", "Letter")): return "Commentary"
    if "Review" in pt: return "Narrative review"
    return "Primary/other"


RULE_MAP = {"Guideline/consensus": "Guideline", "Systematic review/meta-analysis": "SR/MA", "Randomized controlled trial": "RCT",
            "Case report/series": "Case report", "Editorial/comment/letter": "Commentary", "Narrative review": "Narrative review"}


def metrics(y, P, classes, B=1000, seed=SEED):
    yhat = classes[P.argmax(1)]
    out = {"macro_f1": f1_score(y, yhat, average="macro"), "balanced_accuracy": balanced_accuracy_score(y, yhat), "n": int(len(y))}
    pr, rc, f1, sup = precision_recall_fscore_support(y, yhat, labels=classes, zero_division=0)
    out["per_class"] = {c: {"precision": pr[i], "recall": rc[i], "f1": f1[i], "support": int(sup[i]),
                            "auroc": roc_auc_score((y == c).astype(int), P[:, i]) if 0 < (y == c).sum() < len(y) else np.nan}
                        for i, c in enumerate(classes)}
    rng = np.random.default_rng(seed); bs = {"macro_f1": [], **{c: [] for c in classes}}
    for _ in range(B):
        ix = rng.integers(0, len(y), len(y)); yb, Pb = y[ix], P[ix]
        bs["macro_f1"].append(f1_score(yb, classes[Pb.argmax(1)], average="macro"))
        for i, c in enumerate(classes):
            if 0 < (yb == c).sum() < len(yb):
                bs[c].append(roc_auc_score((yb == c).astype(int), Pb[:, i]))
    out["macro_f1_ci"] = list(np.percentile(bs["macro_f1"], [2.5, 97.5]))
    for c in classes:
        out["per_class"][c]["auroc_ci"] = list(np.percentile(bs[c], [2.5, 97.5])) if bs[c] else [np.nan, np.nan]
    return out, yhat


if __name__ == "__main__":
    seed_all()
    items = pd.read_parquet(os.path.join(EMB, "items.parquet"))
    lit = np.where(items.stream.values == "literature")[0]
    X = np.hstack([np.load(os.path.join(EMB, "pubmedbert.npy"))[lit], np.load(os.path.join(EMB, "bge_small.npy"))[lit]]).astype(np.float32)
    L = pd.read_parquet(os.path.join(DATA, "literature_included_final.parquet"),
                        columns=["record_id", "year", "pub_types", "mesh", "in_pubmed", "evidence_type"])
    L = L.set_index("record_id").loc[items.item_id.values[lit]].reset_index()
    indexed = (L.in_pubmed.fillna(False).astype(bool) & (L.mesh.fillna("").str.len() > 0)).values
    y_all = np.array([nlm_label(p) for p in L.pub_types])
    lab = np.where(indexed)[0]
    y = y_all[lab]
    log_event("classifier.labels", n_labelled=len(lab), **pd.Series(y).value_counts().to_dict())
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    grid = GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.03, 0.1, 0.3]}, scoring="f1_macro",
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=5)
    res = {}
    tr, te = train_test_split(np.arange(len(lab)), test_size=0.2, stratify=y, random_state=SEED)
    grid.fit(X[lab[tr]], y[tr]); bestC = grid.best_params_["logisticregression__C"]
    m = grid.best_estimator_; cls = m.classes_
    P = m.predict_proba(X[lab[te]])
    res["random_holdout"], yh = metrics(y[te], P, cls); res["random_holdout"]["C"] = bestC
    pd.DataFrame(confusion_matrix(y[te], yh, labels=CLASSES), index=CLASSES, columns=CLASSES).to_csv(os.path.join(ANA, "classifier_confusion_random.csv"))
    yr = L.year.values[lab]
    trt, tet = np.where(yr <= 2023)[0], np.where(yr >= 2024)[0]
    mt = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=bestC)).fit(X[lab[trt]], y[trt])
    Pt = mt.predict_proba(X[lab[tet]])
    res["temporal_holdout"], yht = metrics(y[tet], Pt, mt.classes_); res["temporal_holdout"]["train_n"] = int(len(trt))
    pd.DataFrame(confusion_matrix(y[tet], yht, labels=CLASSES), index=CLASSES, columns=CLASSES).to_csv(os.path.join(ANA, "classifier_confusion_temporal.csv"))
    mf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", C=bestC)).fit(X[lab], y)
    Pa = mf.predict_proba(X)
    pred = mf.classes_[Pa.argmax(1)]
    out = pd.DataFrame({"record_id": L.record_id, "nlm_indexed": indexed, "nlm_label": np.where(indexed, y_all, None),
                        "pred_label": pred, "pred_prob": Pa.max(1)})
    for i, c in enumerate(mf.classes_):
        out["p_" + c.replace("/", "").replace(" ", "").lower()] = Pa[:, i]
    out["rule_label"] = L.evidence_type.map(RULE_MAP).fillna("Primary/other").values
    out.to_parquet(os.path.join(ANA, "classifier_predictions.parquet"), index=False)
    ni = ~indexed
    res["agreement_rule_vs_model_unindexed"] = {"n": int(ni.sum()), "kappa": cohen_kappa_score(out.rule_label[ni], out.pred_label[ni]),
                                               "raw_agreement": float((out.rule_label[ni] == out.pred_label[ni]).mean())}
    res["agreement_rule_vs_model_indexed"] = {"n": int(indexed.sum()), "kappa": cohen_kappa_score(out.rule_label[indexed], out.pred_label[indexed])}
    res["predicted_distribution_unindexed"] = pd.Series(pred[ni]).value_counts().to_dict()
    json.dump(res, open(os.path.join(ANA, "classifier_metrics.json"), "w"), indent=1, default=float)
    log_event("classifier.done", bestC=bestC, macroF1_random=round(res["random_holdout"]["macro_f1"], 3),
              macroF1_temporal=round(res["temporal_holdout"]["macro_f1"], 3),
              kappa_unindexed=round(res["agreement_rule_vs_model_unindexed"]["kappa"], 3))
