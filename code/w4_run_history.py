from __future__ import annotations

import json

import pandas as pd

from common import LOG, SRC, update_results

RUNS = {
    "2026-11-17T01:52:39": ("Partial corpus (PubTator3 export incomplete); plan v1.0 with A1–A2", "no"),
    "2026-11-17T01:55:54": ("Partial corpus; after A3 and A4", "no"),
    "2026-11-17T02:33:30": ("Full corpus; A1–A5, as pre-specified (negative sampler error; no variants)", "no"),
    "2026-11-17T02:54:58": ("Sentence-tuned encoder probe; A1–A5 (sampler error)", "no"),
    "2026-11-17T02:58:58": ("Random-weight encoder; A1–A5 (sampler error)", "no"),
    "2026-11-17T06:06:23": ("Full corpus; A1–A6 (sampler error; no variants)", "no"),
    "2026-11-17T06:29:40": ("Sentence-tuned encoder probe; A1–A6 (sampler error)", "no"),
    "2026-11-17T06:33:58": ("Random-weight encoder; A1–A6 (sampler error)", "no"),
    "2026-11-17T06:48:27": ("Full corpus; A1–A7 (no variants)", "no"),
    "2026-11-17T07:10:15": ("Sentence-tuned encoder probe; A1–A7 (no variants)", "no"),
    "2026-11-17T07:14:25": ("Random-weight encoder; A1–A7 (no variants)", "no"),
    "2026-11-17T07:18:49": ("Species-specific identifiers; A1–A5 and A7 (no variants)", "no"),
    "2026-11-18T00:52:35": ("Full corpus; A1–A8 (annotation artefacts, corrected in A9)", "no"),
    "2026-11-18T01:37:25": ("Species-specific identifiers; A1–A5, A7 and A8 (annotation artefacts)", "no"),
}
DEV = {"2026-11-17T15:47:07": "Development run of the A8 code (training cutoffs from 2013, 50 draws; forecast file cb9e5fc overwritten)",
       "2026-11-18T05:36:42": "Development run of the foundation-model extension (A10; development task 2016, labels 2017–2019; no forecast)"}
LATER = [("World model, full corpus; A1–A9 (main analysis)", "yes"),
         ("World model, species-specific identifiers; A1–A5, A7–A9 (sensitivity analysis)", "yes")]
FWM = "Foundation-model extension and re-estimated world model; A1–A11 (comparison experiment)"


def main():
    rows, cur, papers = [], None, None
    for line in open(LOG / "pipeline_log.jsonl", encoding="utf-8"):
        o = json.loads(line)
        st, utc = o["step"], o["utc"][:19]
        if st == "w3.prepare":
            papers = o.get("docs")
        elif st == "w3.primary":
            g = lambda k, i: (o[k][i] if isinstance(o.get(k), list) and len(o[k]) > i else None)
            cur = {"utc": utc, "papers": papers, "neural_auroc": g("neural_world_model", 0),
                   "neural_auroc_popularity_adjusted": g("neural_world_model", 2), "hazard_auroc": g("hazard_world_model", 0),
                   "resource_allocation_auroc": g("resource_allocation", 0),
                   "resource_allocation_popularity_adjusted": g("resource_allocation", 2), "holdout_neural_auroc": None}
            rows.append(cur)
        elif st == "w3.near" and cur is not None:
            cur["holdout_neural_auroc"] = o.get("neural_world_model")
        elif st == "w6.primary":
            g = lambda k, i: (o[k][i] if isinstance(o.get(k), list) and len(o[k]) > i else None)
            cur = {"utc": utc, "papers": papers, "neural_auroc": g("world_model", 0),
                   "neural_auroc_popularity_adjusted": g("world_model", 1), "hazard_auroc": g("hazard_v1", 0),
                   "resource_allocation_auroc": g("resource_allocation", 0),
                   "resource_allocation_popularity_adjusted": g("resource_allocation", 1), "holdout_neural_auroc": None,
                   "model": "foundation"}
            rows.append(cur)
        elif st == "w6.holdout" and cur is not None:
            cur["holdout_neural_auroc"] = (o.get("world_model") or [None])[0]
    H = pd.DataFrame(rows)
    H = H[H.utc >= "2026-11-17T01:50"]
    devs = dev_runs()
    later = iter(LATER)
    lab = []
    H["papers"] = H["papers"].ffill()
    for u, mdl in zip(H.utc, H.get("model", [None] * len(H))):
        if u in devs:
            lab.append((DEV.get(u, "Development run (few cutoffs)"), "no"))
        elif mdl == "foundation":
            lab.append((FWM.replace("A1–A11", "A1–A10") if u < "2026-11-18T08:00" else FWM, "yes"))
        else:
            lab.append(RUNS.get(u) or next(later, ("Later run", "no")))
    H["run"], H["reported"] = [x[0] for x in lab], [x[1] for x in lab]
    H.to_csv(SRC / "wm_run_history.csv", index=False)
    update_results("world_model_run_history", H.to_dict(orient="records"))
    print(H[["utc", "run", "neural_auroc", "resource_allocation_auroc", "reported"]].to_string(index=False))


def dev_runs():
    out = list(DEV)
    for line in open(LOG / "pipeline_log.jsonl", encoding="utf-8"):
        o = json.loads(line)
        if o["step"] in ("w3.primary", "w6.primary") and o.get("dev"):
            out.append(o["utc"][:19])
    return out


if __name__ == "__main__":
    main()
