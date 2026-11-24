from __future__ import annotations

import pandas as pd

from common import PKG, SRC

FILES = {
    "Figure1": ["fig1a_papers_by_year.csv", "architecture_dimensions.csv"],
    "Figure2": ["fig3a_map_points.csv", "ed3_dendrogram_leaves.csv", "ed3_ward_linkage.csv", "ed3_domains.csv",
                "ed3_topics_shown.csv", "topic_dynamics.csv"],
    "Figure3": ["atlas_topic_rings.csv", "atlas_ward_linkage.csv", "atlas_entity_states.csv", "atlas_domain_profiles.csv",
                "atlas_pair_input_auroc.csv", "atlas_domain_proximity.csv", "atlas_new_association_domain_pairs.csv",
                "fig3_atlas_ring_summary.csv"],
    "Figure4": ["topk_precision_intervals.csv", "wm_roc_exact_primary_biomedbert.csv", "wm_pr_exact_primary_biomedbert.csv",
                "wm_primary_metrics_biomedbert.csv", "wm_auroc_grid_biomedbert.csv", "wm_grid_intervals_biomedbert.csv",
                "wm_roc_exact_holdout_biomedbert.csv", "wm_exact_metrics_primary.csv", "wm_design_contrast.csv"],
    "Figure5": ["wm_subgroups_biomedbert.csv"],
    "Figure6": ["topk_precision_intervals.csv", "fwm_grid.csv", "fwm_primary_metrics.csv", "fwm_contributions.csv",
                "fwm_dynamics.csv", "fwm_attention_forecast.csv", "fwm_exact_metrics.csv"],
    "ExtendedDataFigures": ["fig1_design_numbers.csv",
                     "wm_papers_by_year.csv", "fwm_graphs_by_year.csv", "wm_task_sizes.csv", "fwm_encoder_reconstruction.csv",
                     "fig2_summary.csv", "nomenclature_map_points.csv", "ed3c_probe_folds.csv", "ed3d_topic_current_share.csv",
                     "wm_ablations_biomedbert.csv", "wm_ablation_intervals.csv", "wm_calibration_biomedbert.csv",
                     "wm_sensitivity.csv", "wm_cleaned_entities.csv", "wm_encoders_paired.csv", "wm_state_forecast.csv",
                     "fwm_settings.csv",
                     "trial_landscape_masld_targeted.csv", "trial_landscape_advanced_any_aetiology.csv",
                     "trial_publication_km_curves.csv", "trial_publication_completed.csv",
                     "wm_run_history.csv", "wm_annotation_drift.csv"],
}


SHEET = {"fig3a_map_points.csv": "fig2c_umap_points", "ed3_dendrogram_leaves.csv": "fig2b_dendrogram_leaves",
         "ed3_ward_linkage.csv": "fig2b_ward_linkage", "ed3_domains.csv": "fig2_domains", "ed3_topics_shown.csv": "fig2e_topics_shown",
         "fig3_atlas_ring_summary.csv": "fig3a_ring_summary", "fig2_summary.csv": "ed3_renaming_summary",
         "fig1_design_numbers.csv": "ed1_design_numbers", "ed3c_probe_folds.csv": "ed3c_probe_folds",
         "ed3d_topic_current_share.csv": "ed3d_topic_current_share", "topk_precision_intervals.csv": "topk_precision_ci",
         "architecture_dimensions.csv": "architecture_dimensions"}


def derived_fwm():
    import json
    F = json.loads((PKG / "analysis" / "results.json").read_text(encoding="utf-8"))["foundation_world_model"]
    E = F["encoder"]
    pd.DataFrame([{"year": int(t), "entities": v, "co_mention_edges": E["edges_by_snapshot"][t]["co_mention"],
                   "language_neighbour_edges": E["edges_by_snapshot"][t]["affinity"]}
                  for t, v in E["nodes_by_snapshot"].items()]).to_csv(SRC / "fwm_graphs_by_year.csv", index=False)
    pd.DataFrame([{"setting": k, "value": json.dumps(v) if isinstance(v, (dict, list)) else v} for k, v in F["settings"].items()]
                 + [{"setting": f"source_{k}", "value": v} for k, v in F["sources"].items()]).to_csv(SRC / "fwm_settings.csv", index=False)
    pd.DataFrame([{"modality": k, "masked_node_huber_loss": v} for k, v in E["masked_node_loss"].items()]).to_csv(
        SRC / "fwm_encoder_reconstruction.csv", index=False)
    fmt = lambda v: f"{v[0]:.5f} to {v[1]:.5f}" if isinstance(v, list) else v
    pd.DataFrame([{"method": k, **{q: fmt(x) for q, x in m.items()}} for k, m in F["primary"]["metrics"].items()]).to_csv(
        SRC / "fwm_primary_metrics.csv", index=False)
    rows = [{"contrast": k, "metric": "auroc", "ci_95": fmt(v)} for k, v in F["primary"]["contrasts"].items()]
    rows += [{"contrast": k, "metric": "auroc_popularity_adjusted", "ci_95": fmt(v)}
             for k, v in F["primary"]["contrasts_popularity_adjusted"].items()]
    pr = F.get("probe_random_encoder", {})
    for k in ("contrast", "contrast_popularity_adjusted"):
        for c, v in pr.get(k, {}).items():
            rows.append({"contrast": c, "metric": "auroc" if k == "contrast" else "auroc_popularity_adjusted", "ci_95": fmt(v)})
    pd.DataFrame(rows).to_csv(SRC / "fwm_contributions.csv", index=False)
    X = F["exact_primary"]
    pd.DataFrame([{"method": k, "estimation": "exact, every candidate pair", **m} for k, m in X["metrics"].items()]
                 + [{"method": k, "estimation": "exact hold-out (January-June 2026)", **m} for k, m in F["exact_holdout"]["metrics"].items()]
                 ).to_csv(SRC / "fwm_exact_metrics.csv", index=False)
    pd.DataFrame([{"method": k, "degree_matched_auroc": X["degree_matched_auroc"].get(k),
                   **({f"entity_{q}": v for q, v in X["entity_centred"][k].items()} if k in X["entity_centred"] else {})}
                  for k in X["degree_matched_auroc"]]).to_csv(SRC / "fwm_beyond_popularity.csv", index=False)


def derived():
    import json
    W = json.loads((PKG / "analysis" / "results.json").read_text(encoding="utf-8"))["world_model"]
    A8 = W["a8"]
    pd.DataFrame(W["task_sizes"]).to_csv(SRC / "wm_task_sizes.csv", index=False)
    DC = W["design_contrast"]
    pd.DataFrame([{"design": dz, "method": k, "auroc": v,
                   "auroc_ci_95": (lambda c: f"{c[0]:.4f} to {c[1]:.4f}" if c else "")(DC.get(cik, {}).get(k))}
                  for dz, cik in (("random_holdout_never_negatives", "random_holdout_ci"), ("temporal_primary", "temporal_primary_ci"))
                  for k, v in DC[dz].items()]).to_csv(SRC / "wm_design_contrast.csv", index=False)
    pd.DataFrame({"year": list(W["papers_by_year"]), "papers": list(W["papers_by_year"].values())}).to_csv(
        SRC / "wm_papers_by_year.csv", index=False)
    co = W["primary"]["hazard_coefficients"]
    pd.DataFrame({"feature": list(co["standardized"]), "standardized": list(co["standardized"].values()),
                  "per_unit": [co["per_unit"][k] for k in co["standardized"]]}).to_csv(SRC / "wm_hazard_coefficients.csv", index=False)
    AI = A8["ablation_intervals"]
    pd.DataFrame([{"removed_and_model": k, "auroc_change_ci_full_minus_removed": f"{v[0]:.5f} to {v[1]:.5f}",
                   "popadj_change_ci_full_minus_removed": f"{AI['popadj_change_ci'][k][0]:.5f} to {AI['popadj_change_ci'][k][1]:.5f}"}
                  for k, v in AI["auroc_change_ci"].items()]).to_csv(SRC / "wm_ablation_intervals.csv", index=False)
    pd.DataFrame([{"method_and_encoder": k, **{q: (v if not isinstance(v, list) else f"{v[0]:.4f} to {v[1]:.4f}") for q, v in m.items()}}
                  for k, m in A8["encoders_paired"]["metrics"].items()]).to_csv(SRC / "wm_encoders_paired.csv", index=False)
    pd.DataFrame([{"method": k, "estimation": "exact, every candidate pair", **m} for k, m in A8["exact_primary"]["metrics"].items()]).to_csv(
        SRC / "wm_exact_metrics_primary.csv", index=False)
    rows = [{"analysis": "semantic state", **{k: v for k, v in A8["state_forecast"]["semantic"].items() if not isinstance(v, list)}},
            {"analysis": "attention", **A8["state_forecast"]["attention"]}]
    pd.DataFrame(rows).to_csv(SRC / "wm_state_forecast.csv", index=False)
    pd.DataFrame([{"method": k, **{q: (v if not isinstance(v, list) else f"{v[0]:.4f} to {v[1]:.4f}") for q, v in m.items()}}
                  for k, m in A8["cleaned"]["metrics"].items()]).to_csv(SRC / "wm_cleaned_entities.csv", index=False)
    pm = SRC / "wm_primary_metrics_biomedbert.csv"
    if pm.exists():
        df = pd.read_csv(pm)
        df["estimation"] = "all positives and a uniform sample of negatives; average precision with inverse-sampling weights"
        df.to_csv(pm, index=False)


def derived_review():
    import json
    R = json.loads((PKG / "analysis" / "results.json").read_text(encoding="utf-8"))
    N = R["nomenclature"]
    rows = []
    for emb in ("original", "neutral"):
        for t, v in N[emb]["topic_current_share"].items():
            rows.append({"embedding": emb, "topic": int(t), "share_current_names": v})
    pd.DataFrame(rows).to_csv(SRC / "ed3d_topic_current_share.csv", index=False)
    rows = []
    for emb in ("original", "neutral"):
        for k in ("era_auroc_2023_2026", "era_auroc_2024_only"):
            e = N[emb][k]
            rows += [{"embedding": emb, "analysis": k, "fold": i + 1, "auroc": a} for i, a in enumerate(e["folds"])]
    pd.DataFrame(rows).to_csv(SRC / "ed3c_probe_folds.csv", index=False)
    rows = []
    for src, tk in (("world model (main analysis)", R["world_model"]["a8"]["exact_primary"]["topk_ci"]),
                    ("comparison experiment", R["foundation_world_model"]["exact_primary"]["topk_ci"])):
        for key, (lo, hi) in tk.items():
            m, k = key.rsplit("@", 1)
            rows.append({"analysis": src, "method": _relabel(m) if src.startswith("comparison") else m, "k": int(k),
                         "ci95_low": lo, "ci95_high": hi})
    pd.DataFrame(rows).to_csv(SRC / "topk_precision_intervals.csv", index=False)


def _relabel(s):
    import re
    s = re.sub(r"hazard_v1_neural", "@@WMR@@", str(s))
    s = re.sub(r"hazard_v1(?![a-z_])", "@@WML@@", s)
    s = re.sub(r"world_model", "foundation_extension", s)
    return s.replace("@@WMR@@", "world_model_reestimated").replace("@@WML@@", "world_model_linear_reestimated")


RELABEL = {"fwm_grid.csv", "fwm_primary_metrics.csv", "fwm_contributions.csv", "fwm_exact_metrics.csv", "fwm_dynamics.csv",
           "fwm_attention_forecast.csv", "fwm_beyond_popularity.csv"}


def main():
    from figstyle import DOMAIN_SHORT
    derived()
    derived_fwm()
    derived_review()
    out = PKG / "source_data_xlsx"
    out.mkdir(exist_ok=True)
    for old in out.glob("SourceData_*.xlsx"):
        if old.stem.replace("SourceData_", "") not in FILES:
            old.unlink()
    for fig, files in FILES.items():
        with pd.ExcelWriter(out / f"SourceData_{fig}.xlsx") as xw:
            for f in files:
                df = pd.read_csv(SRC / f)
                if f in RELABEL:
                    df.columns = [_relabel(c) for c in df.columns]
                    for c in df.columns:
                        if df[c].dtype == object:
                            df[c] = df[c].map(lambda v: _relabel(v) if isinstance(v, str) else v)
                for c in ("domain", "group", "domain_name"):
                    if c in df.columns and df[c].isin(list(DOMAIN_SHORT)).any():
                        df.insert(df.columns.get_loc(c) + 1, f"{c}_in_figure", df[c].map(DOMAIN_SHORT))
                sheet = SHEET.get(f, f.replace(".csv", ""))[:31]
                df.to_excel(xw, sheet_name=sheet, index=False)
        print(fig, "written")


if __name__ == "__main__":
    main()
