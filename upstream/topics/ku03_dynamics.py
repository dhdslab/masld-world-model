import os, sys
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
from ku_common import *

REF = sys.argv[1] if len(sys.argv) > 1 else REF_RUN

if __name__ == "__main__":
    T = pd.read_parquet(os.path.join(ANA, f"{REF}_topics_items.parquet"))
    L = pd.read_parquet(os.path.join(DATA, "literature_included_final.parquet"),
                        columns=["record_id", "year", "nom_nafld", "nom_mafld", "nom_masld", "evidence_type", "evidence_tier", "pub_format"])
    Tl = T[T.stream == "literature"].merge(L, left_on="item_id", right_on="record_id", suffixes=("", "_l"))
    C = pd.crosstab(Tl.topic, Tl.year)
    C.to_csv(os.path.join(ANA, "topic_year_counts.csv"))
    yrs = np.arange(2020, 2026)
    Ny = C[yrs].sum(0).values
    rows = []
    for k, r in C.iterrows():
        y = r[yrs].values.astype(float)
        X = sm.add_constant(yrs - 2020.0)
        try:
            f = sm.GLM(y, X, family=sm.families.Poisson(), offset=np.log(Ny)).fit()
            b, se, p = f.params[1], f.bse[1], f.pvalues[1]
            disp = max(1.0, f.pearson_chi2 / f.df_resid)
            se *= np.sqrt(disp); z = b / se
            from scipy.stats import norm
            p = 2 * norm.sf(abs(z))
        except Exception:
            b, se, p, disp = np.nan, np.nan, np.nan, np.nan
        post = Tl[(Tl.topic == k) & (Tl.year >= 2024)]
        nm, nn = int(post.nom_masld.sum()), len(post)
        lo, hi = proportion_confint(nm, nn, method="wilson") if nn else (np.nan, np.nan)
        rows.append({"topic": k, "n_total": int(r.sum()), "n_2020": int(r.get(2020, 0)), "n_2025": int(r.get(2025, 0)),
                     "n_2026H1": int(r.get(2026, 0)),
                     "share_2020": r.get(2020, 0) / C[2020].sum(), "share_2025": r.get(2025, 0) / C[2025].sum(),
                     "share_2026H1": r.get(2026, 0) / C[2026].sum(),
                     "rgr": np.exp(b), "rgr_lo": np.exp(b - 1.96 * se), "rgr_hi": np.exp(b + 1.96 * se), "p": p, "dispersion": disp,
                     "cagr_2020_2025": (max(r.get(2025, 0), 0.5) / max(r.get(2020, 0), 0.5)) ** (1 / 5) - 1,
                     "masld_term_share_post2024": nm / nn if nn else np.nan, "masld_lo": lo, "masld_hi": hi, "n_post2024": nn})
    D = pd.DataFrame(rows)
    ok = D.p.notna()
    D.loc[ok, "q"] = multipletests(D.loc[ok, "p"], method="fdr_bh")[1]
    D["trend"] = np.where((D.q < 0.05) & (D.rgr > 1), "emerging", np.where((D.q < 0.05) & (D.rgr < 1), "declining", "stable"))
    D.to_csv(os.path.join(ANA, "topic_dynamics.csv"), index=False)
    overall = {"corpus_growth_2020_2025": float(C[2025].sum() / C[2020].sum()),
               "n_emerging": int((D.trend == "emerging").sum()), "n_declining": int((D.trend == "declining").sum()),
               "n_stable": int((D.trend == "stable").sum()),
               "masld_share_post2024_overall": float(Tl[Tl.year >= 2024].nom_masld.mean()),
               "masld_share_topic_range": [float(D.masld_term_share_post2024.min()), float(D.masld_term_share_post2024.max())]}
    import json; json.dump(overall, open(os.path.join(ANA, "dynamics_summary.json"), "w"), indent=1)
    log_event("dynamics.done", **overall)
