from __future__ import annotations

import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

from common import to_group
from common import (ADJACENT_TOPICS, ANA, TOPICS_ANA, TOPICS_EMB, CORPUS_PROC, DATASET_LABEL, GROUP_OF_TYPE,
                    INDIVIDUAL_DATASETS, REF_RUN, SEED, SRC, log, update_results, wilson)

YEARS = list(range(2020, 2026))


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def _split_top(p):
    parts, depth, cur, i = [], 0, "", 0
    while i < len(p):
        ch = p[i]
        if ch == "\\":
            cur += p[i:i + 2]; i += 2; continue
        if ch == "(": depth += 1
        elif ch == ")": depth -= 1
        if ch == "|" and depth == 0:
            parts.append(cur); cur = ""
        else:
            cur += ch
        i += 1
    parts.append(cur)
    return parts


def mixcase(p):
    out = []
    for alt in _split_top(p):
        letters = re.sub(r"\\[a-zA-Z]|s\?", "", alt)
        out.append(f"(?i:{alt})" if re.search(r"[a-z]", letters) else f"(?:{alt})")
    return "|".join(out)


def rx(p):
    return re.compile(mixcase(p))


R = {
    "guideline": rx(r"\bguideline|practice guidance|consensus (statement|recommendation|conference)|position (statement|paper)|"
                    r"expert (consensus|opinion statement)|delphi|clinical practice update|recommendations? (for|from|of the)"),
    "sr_ma": rx(r"meta[\s\-]?analy|systematic(al)?[\s\-]+review|umbrella review|scoping review|network meta"),
    "review_title": rx(r"\breview\b|\boverview\b|state of the art|current (status|perspectives?|concepts?|understanding)|"
                       r"\bupdate on\b|perspectives?\b|narrative|mini[\s\-]?review|\bprimer\b|lessons learned"),
    "protocol": rx(r"study protocol|trial protocol|protocol for a|rationale and design|design and rationale|statistical analysis plan"),
    "case_report": rx(r"case report|a case of|case series|\bcase presentation\b"),
    "randomized": rx(r"randomi[sz]ed|randomly assigned|randomly allocated|placebo[\s\-]controlled"),
    "trial": rx(r"\btrial\b|phase (1|2|3|i{1,3}|2a|2b|iia|iib)\b"),
    "human": rx(r"\bpatients?\b|participants?|\badults?\b|subjects|volunteers|\bmen\b|\bwomen\b|children|adolescents"),
    "nonrand_trial": rx(r"single[\s\-]arm|open[\s\-]label|non[\s\-]?randomi[sz]ed|pilot (study|trial)|phase (1|i|2a|iia)\b|"
                        r"pre[\s\-]?post|before[\s\-]and[\s\-]after|quasi[\s\-]experimental"),
    "animal": rx(r"\bmice\b|\bmouse\b|murine|\brats?\b|zebrafish|hamsters?|rabbits?|\bpigs?\b|minipig|primates?|"
                 r"drosophila|c\. elegans|caenorhabditis|hepatocyte[\s\-]specific|liver[\s\-]specific|transgenic|conditional (knockout|deletion)|"
                 r"\bducks?\b|broilers?|laying hens|\bcattle\b|dairy cows?|largemouth bass|grass carp|tilapia|zebrafish larvae|"
                 r"high[\s\-]fat diet[\s\-]?(fed|induced)|\bHFD\b|knock[\s\-]?out|"
                 r"\bdb/db\b|\bob/ob\b|methionine[\s\-]choline|\bMCD\b diet|CDAHFD|western diet[\s\-]fed"),
    "editorial_title": rx(r"^\s*(editorial|comment|reply|response|letter|correspondence|in memoriam|re:|author'?s? reply)|"
                          r"\b(commentary|editorial|reply to|response to|letter to the editor)\b"),
    "mr": rx(r"mendelian randomi[sz]ation"),
    "invitro": rx(r"in vitro|cell lines?|hepg2|huh[\s\-]?7|l02 cells|aml12|lx[\s\-]?2|primary hepatocytes|organoids?|"
                  r"palmitic acid[\s\-]?(treated|induced)|oleic acid[\s\-]?(treated|induced)|free fatty acid[\s\-]induced"),
    "insilico": rx(r"network pharmacology|molecular docking|bioinformatic|in silico|\bGEO\b database|"
                   r"weighted gene co[\s\-]?expression|\bWGCNA\b|differentially expressed genes|molecular dynamics simulation"),
    "diagnostic": rx(r"diagnostic (accuracy|performance)|\bauroc\b|\bauc\b|area under the (receiver|roc|curve)|"
                     r"sensitivity and specificity|\bc[\s\-]?statistic|validation (cohort|study)|prediction model|"
                     r"nomogram|risk score|machine learning|deep learning|external validation"),
    "cohort": rx(r"\bcohort\b|prospective|retrospective|longitudinal|follow[\s\-]?up|hazard ratio|\bHRs?\b|incidence|"
                 r"person[\s\-]years|registry|nationwide|population[\s\-]based|real[\s\-]world"),
    "case_control": rx(r"case[\s\-]control|matched controls"),
    "xs_explicit": rx(r"cross[\s\-]sectional"),
    "diag_title": rx(r"diagnostic (accuracy|performance|value|utility)|predict(ion|ive) (model|score|value|performance)|nomogram|machine learning|deep learning|risk (score|model)|validation of|validat(ed|ing) (a|the)"),
    "cohort_explicit": rx(r"\bcohort study|prospective(ly)? (cohort|study|follow)|retrospective(ly)? (cohort|study|analy)|longitudinal (study|cohort|analysis)|population[\s\-]based cohort|nationwide cohort|follow[\s\-]?up (of|period|study)"),
    "cross_sectional": rx(r"cross[\s\-]sectional|prevalence|survey|NHANES|KNHANES|odds ratio|\bORs?\b"),
}


def text_rule_full(t, a, kw):
    t, a = t or "", a or ""
    x = t + " \n " + a + " \n " + (kw or "")
    has_abs = len(a) > 100
    if R["guideline"].search(t): return "Guideline/consensus"
    if R["sr_ma"].search(t) or R["sr_ma"].search(a[:800]): return "Systematic review/meta-analysis"
    if R["editorial_title"].search(t) and not has_abs: return "Editorial/comment/letter"
    if R["case_report"].search(t): return "Case report/series"
    if R["protocol"].search(t): return "Trial/study protocol"
    if (R["review_title"].search(t) and not R["randomized"].search(a)
            and not R["cohort"].search(a[:600]) and not R["animal"].search(a)):
        return "Narrative review"
    if R["randomized"].search(x) and R["trial"].search(x) and R["human"].search(x) and not R["animal"].search(x):
        return "Randomized controlled trial"
    if R["nonrand_trial"].search(x) and R["trial"].search(x) and R["human"].search(x):
        return "Non-randomized clinical trial"
    if R["mr"].search(x): return "Mendelian randomization"
    if R["animal"].search(x) or R["invitro"].search(x): return "Basic/experimental (animal or in vitro)"
    if R["insilico"].search(x): return "Basic/experimental (in silico)"
    if not has_abs:
        return "Narrative review" if R["review_title"].search(t) else "Unclassified (title only)"
    if R["case_control"].search(x): return "Case-control"
    if R["diag_title"].search(t): return "Diagnostic/prognostic modelling"
    if R["xs_explicit"].search(x): return "Cross-sectional"
    if R["cohort_explicit"].search(x): return "Cohort"
    if R["diagnostic"].search(x): return "Diagnostic/prognostic modelling"
    if R["cohort"].search(x): return "Cohort"
    if R["cross_sectional"].search(x): return "Cross-sectional"
    return "Other primary study"


def text_rule_7class(title, abstract, keywords):
    t, a = title or "", abstract or ""
    x = t + " \n " + a + " \n " + (keywords or "")
    has_abs = len(a) > 100
    if R["guideline"].search(t):
        return "Guideline"
    if R["sr_ma"].search(t) or R["sr_ma"].search(a[:800]):
        return "SR/MA"
    if R["editorial_title"].search(t) and not has_abs:
        return "Commentary"
    if R["case_report"].search(t):
        return "Case report"
    if R["protocol"].search(t):
        return "Primary/other"
    if R["review_title"].search(t) and not R["randomized"].search(a) and not R["animal"].search(a) \
            and not re.search(mixcase(r"\bcohort\b|prospective|retrospective|longitudinal|follow[\s\-]?up|hazard ratio|\bHRs?\b|incidence|"
                                      r"person[\s\-]years|registry|nationwide|population[\s\-]based|real[\s\-]world"), a[:600]):
        return "Narrative review"
    if R["randomized"].search(x) and R["trial"].search(x) and R["human"].search(x) and not R["animal"].search(x):
        return "RCT"
    if not has_abs and R["review_title"].search(t):
        return "Narrative review"
    return "Primary/other"


def nlm_label(pt):
    pt = str(pt or "")
    if any(s in pt for s in ("Practice Guideline", "Guideline", "Consensus Development Conference")):
        return "Guideline"
    if "Meta-Analysis" in pt or "Systematic Review" in pt:
        return "SR/MA"
    if "Randomized Controlled Trial" in pt:
        return "RCT"
    if "Case Reports" in pt:
        return "Case report"
    if any(s in pt for s in ("Editorial", "Comment", "Letter")):
        return "Commentary"
    if "Review" in pt:
        return "Narrative review"
    return "Primary/other"


def main():
    items = pd.read_parquet(TOPICS_EMB / "items.parquet", columns=["stream", "item_id", "year"])
    T = pd.read_parquet(TOPICS_ANA / f"{REF_RUN}_topics_items.parquet")
    cat = pd.read_csv(TOPICS_ANA / "topic_catalogue_named.csv")
    research = set(cat.topic[~cat.meta_topic])
    dom_name = cat.set_index("topic").domain_name.to_dict()
    lit_idx = np.where(T.stream.values == "literature")[0]
    cols = ["record_id", "year", "evidence_type", "pub_types", "mesh", "in_pubmed", "title", "abstract", "keywords",
            "ai_ml"] + INDIVIDUAL_DATASETS + ["data_gbd"]
    L = pd.read_parquet(CORPUS_PROC / "literature_included_final.parquet", columns=cols)
    L = L.set_index("record_id").loc[items.item_id.values[lit_idx]].reset_index()
    L["topic"] = T.topic.values[lit_idx]
    L["core"] = T.core.values[lit_idx]
    L["research"] = L.topic.isin(research)
    L["domain"] = L.topic.map(dom_name)
    L["adjacent"] = L.topic.isin(ADJACENT_TOPICS)
    L["group"] = to_group(L.evidence_type, L.record_id)
    L["any_dataset"] = L[INDIVIDUAL_DATASETS].any(axis=1)
    L["mr"] = L.evidence_type.eq("Mendelian randomization")
    R_ = L[L.research].copy()

    sizes = R_.topic.value_counts()
    dom_sizes = R_.domain.value_counts()
    structure = {
        "n_topics_total": int(cat.shape[0]), "n_research_topics": int(len(research)),
        "n_meta_topics": int(cat.meta_topic.sum()), "meta_papers": int((~L.research).sum()),
        "research_papers": int(len(R_)), "core_research_papers": int(R_.core.sum()),
        "noise_fraction_literature": float(1 - L.core.mean()),
        "n_domains": int(R_.domain.nunique()),
        "domain_sizes": dom_sizes.to_dict(),
        "domain_core_sizes": R_[R_.core].domain.value_counts().to_dict(),
        "gini_topic_size": gini(sizes.values),
        "top10pct_topics_share": float(sizes.sort_values(ascending=False).head(int(np.ceil(0.1 * len(sizes)))).sum() / sizes.sum()),
        "adjacent_topics": {str(k): v for k, v in ADJACENT_TOPICS.items()},
        "adjacent_papers": int(R_.adjacent.sum()),
        "topic_size_median": float(sizes.median()), "topic_size_range": [int(sizes.min()), int(sizes.max())],
    }
    dy = R_[R_.year.between(2020, 2026)].groupby(["year", "domain"]).size().unstack(fill_value=0)
    dshare = dy.div(dy.sum(1), axis=0)
    dshare.to_csv(SRC / "domain_share_by_year.csv")
    structure["domain_share_2020"] = dshare.loc[2020].to_dict()
    structure["domain_share_2025"] = dshare.loc[2025].to_dict()

    dyn = pd.read_csv(TOPICS_ANA / "topic_dynamics.csv")
    dyn = dyn[dyn.topic.isin(research)].merge(cat[["topic", "topic_label", "domain_name"]], on="topic")
    dyn.to_csv(SRC / "topic_dynamics.csv", index=False)
    structure["dynamics"] = {"emerging": int((dyn.trend == "emerging").sum()),
                             "declining": int((dyn.trend == "declining").sum()),
                             "stable": int((~dyn.trend.isin(["emerging", "declining"])).sum())}
    log("a2.structure", **{k: structure[k] for k in ["n_research_topics", "research_papers", "gini_topic_size"]})

    comp = R_.group.value_counts()
    primary = R_[R_.group.isin(["interventional", "observational", "basic"])].copy()
    primary["interv"] = (primary.group == "interventional").astype(int)
    maturity = {
        "composition_counts": comp.to_dict(),
        "composition_share": (comp / comp.sum()).to_dict(),
        "n_primary_classified": int(len(primary)),
        "interventional_share_primary": float(primary.interv.mean()),
        "noninterventional_per_trial": float((primary.interv == 0).sum() / primary.interv.sum()),
    }
    dom_tab = R_.groupby(["domain", "group"]).size().unstack(fill_value=0)
    dom_tab.to_csv(SRC / "evidence_composition_by_domain.csv")
    pdx = primary.groupby("domain").interv.agg(["sum", "count"])
    maturity["interventional_share_by_domain"] = {d: {"k": int(r["sum"]), "n": int(r["count"]), "share": r["sum"] / r["count"],
                                                     "ci": wilson(r["sum"], r["count"])} for d, r in pdx.iterrows()}
    ptx = primary.groupby("topic").interv.agg(["sum", "count"])
    ptx = ptx.join(cat.set_index("topic")[["topic_label", "domain_name"]])
    ptx.to_csv(SRC / "interventional_by_topic.csv")
    maturity["topics_zero_interventional"] = ptx[ptx["sum"] == 0].topic_label.tolist()

    py = primary[primary.year.between(2020, 2025)].copy()
    py["yc"] = py.year - 2020
    m0 = smf.glm("interv ~ yc", data=py, family=sm.families.Binomial()).fit()
    m1 = smf.glm("interv ~ yc + C(domain)", data=py, family=sm.families.Binomial()).fit()
    m2 = smf.glm("interv ~ yc * C(domain)", data=py, family=sm.families.Binomial()).fit()
    lr = 2 * (m2.llf - m1.llf); df = m2.df_model - m1.df_model
    yearly = py.groupby("year").interv.agg(["sum", "count"])
    trend = {
        "years": YEARS,
        "interventional_k_by_year": yearly["sum"].to_dict(), "primary_n_by_year": yearly["count"].to_dict(),
        "share_by_year": (yearly["sum"] / yearly["count"]).to_dict(),
        "share_ci_by_year": {int(y): wilson(r["sum"], r["count"]) for y, r in yearly.iterrows()},
        "ratio_noninterv_per_trial_by_year": ((yearly["count"] - yearly["sum"]) / yearly["sum"]).to_dict(),
        "or_per_year_crude": float(np.exp(m0.params["yc"])),
        "or_per_year_crude_ci": np.exp(m0.conf_int().loc["yc"]).tolist(), "p_crude": float(m0.pvalues["yc"]),
        "or_per_year_domain_adjusted": float(np.exp(m1.params["yc"])),
        "or_per_year_domain_adjusted_ci": np.exp(m1.conf_int().loc["yc"]).tolist(), "p_adjusted": float(m1.pvalues["yc"]),
        "interaction_lr": float(lr), "interaction_df": int(df), "interaction_p": float(stats.chi2.sf(lr, df)),
    }
    slopes = {}
    for d, g in py.groupby("domain"):
        if g.interv.sum() < 5:
            slopes[d] = None; continue
        mm = smf.glm("interv ~ yc", data=g, family=sm.families.Binomial()).fit()
        slopes[d] = {"or": float(np.exp(mm.params["yc"])), "ci": np.exp(mm.conf_int().loc["yc"]).tolist(),
                     "p": float(mm.pvalues["yc"]), "k": int(g.interv.sum()), "n": int(len(g))}
    trend["or_per_year_by_domain"] = slopes
    py.groupby(["domain", "year"]).interv.agg(["sum", "count"]).to_csv(SRC / "interventional_by_domain_year.csv")
    p26 = primary[primary.year == 2026]
    trend["share_2026H1"] = float(p26.interv.mean()); trend["n_2026H1"] = int(len(p26))
    maturity["trend"] = trend
    sens = {}
    for name, mask in [("core_only", primary.core), ("human_scope", ~primary.adjacent)]:
        g = primary[mask & primary.year.between(2020, 2025)].copy(); g["yc"] = g.year - 2020
        mm = smf.glm("interv ~ yc + C(domain)", data=g, family=sm.families.Binomial()).fit()
        sens[name] = {"share": float(primary[mask].interv.mean()), "n": int(mask.sum()),
                      "or_per_year_adj": float(np.exp(mm.params["yc"])), "ci": np.exp(mm.conf_int().loc["yc"]).tolist(),
                      "p": float(mm.pvalues["yc"])}
    cp = pd.read_parquet(TOPICS_ANA / "classifier_predictions.parquet").set_index("record_id")
    up = cp.loc[R_.record_id]
    upgrade = (~up.nlm_indexed.values) & (up.pred_prob.values >= 0.8) & np.isin(up.pred_label.values, ["RCT", "SR/MA", "Guideline"])
    g2 = R_.group.values.copy()
    g2[upgrade & (up.pred_label.values == "RCT")] = "interventional"
    g2[upgrade & np.isin(up.pred_label.values, ["SR/MA", "Guideline"])] = "synthesis"
    U = R_.assign(group=g2)
    Up = U[U.group.isin(["interventional", "observational", "basic"])].copy(); Up["interv"] = (Up.group == "interventional").astype(int)
    g = Up[Up.year.between(2020, 2025)].copy(); g["yc"] = g.year - 2020
    mm = smf.glm("interv ~ yc + C(domain)", data=g, family=sm.families.Binomial()).fit()
    sens["classifier_upgrade"] = {"records_upgraded": int(upgrade.sum()), "share": float(Up.interv.mean()), "n": int(len(Up)),
                                  "or_per_year_adj": float(np.exp(mm.params["yc"])), "ci": np.exp(mm.conf_int().loc["yc"]).tolist(),
                                  "p": float(mm.pvalues["yc"])}
    R_["indexed"] = (R_.in_pubmed.fillna(False).astype(bool) & (R_.mesh.fillna("").str.len() > 0)).values
    maturity["nlm_indexed_share_by_year"] = R_.groupby("year").indexed.mean().to_dict()
    gi = primary.assign(indexed=R_.set_index("record_id").loc[primary.record_id, "indexed"].values)
    g = gi[gi.indexed & gi.year.between(2020, 2025)].copy(); g["yc"] = g.year - 2020
    mm = smf.glm("interv ~ yc + C(domain)", data=g, family=sm.families.Binomial()).fit()
    sens["nlm_indexed_only"] = {"share": float(gi[gi.indexed].interv.mean()), "n": int(gi.indexed.sum()),
                                "or_per_year_adj": float(np.exp(mm.params["yc"])), "ci": np.exp(mm.conf_int().loc["yc"]).tolist(),
                                "p": float(mm.pvalues["yc"]),
                                "share_by_year": g.groupby("year").interv.mean().to_dict()}
    tt = np.array([text_rule_full(a, b, c) for a, b, c in zip(R_.title, R_.abstract.fillna(""), R_.keywords.fillna(""))])
    TX = R_.assign(group=to_group(tt, R_.record_id))
    TXp = TX[TX.group.isin(["interventional", "observational", "basic"])].copy()
    TXp["interv"] = (TXp.group == "interventional").astype(int)
    g = TXp[TXp.year.between(2020, 2025)].copy(); g["yc"] = g.year - 2020
    mm = smf.glm("interv ~ yc + C(domain)", data=g, family=sm.families.Binomial()).fit()
    sens["text_rules_all_records"] = {"share": float(TXp.interv.mean()), "n": int(len(TXp)),
                                      "or_per_year_adj": float(np.exp(mm.params["yc"])), "ci": np.exp(mm.conf_int().loc["yc"]).tolist(),
                                      "p": float(mm.pvalues["yc"]),
                                      "share_by_year": g.groupby("year").interv.mean().to_dict()}
    maturity["trend_sensitivity"] = sens
    def _fit(formula, data, term="yc"):
        mm_ = smf.glm(formula, data=data, family=sm.families.Binomial()).fit()
        return {"or": float(np.exp(mm_.params[term])), "ci": np.exp(mm_.conf_int().loc[term]).tolist(),
                "p": float(mm_.pvalues[term]), "n": int(len(data))}
    rob = {"period_2020_2022": _fit("interv ~ yc + C(domain)", py[py.year <= 2022]),
           "period_2022_2025": _fit("interv ~ yc + C(domain)", py[py.year >= 2022]),
           "without_trials_domain": _fit("interv ~ yc + C(domain)", py[py.domain != "Clinical trials and interventions"]),
           "crude_without_trials_domain": _fit("interv ~ yc", py[py.domain != "Clinical trials and interventions"])}
    npub = py[~(py.any_dataset.astype(bool) | py.mr.astype(bool))]
    rob["without_public_resource_papers"] = _fit("interv ~ yc + C(domain)", npub)
    rob["without_public_resource_share_by_year"] = {int(y): float(v) for y, v in npub.groupby("year").interv.mean().items()}
    gy = py.assign(indexed=R_.set_index("record_id").loc[py.record_id, "indexed"].values.astype(int))
    mi = smf.glm("interv ~ yc * indexed + C(domain)", data=gy, family=sm.families.Binomial()).fit()
    b, V = mi.params, mi.cov_params()
    se_ix = float(np.sqrt(V.loc["yc", "yc"] + V.loc["yc:indexed", "yc:indexed"] + 2 * V.loc["yc", "yc:indexed"]))
    b_ix = float(b["yc"] + b["yc:indexed"])
    rob["indexing_interaction"] = {
        "or_unindexed": float(np.exp(b["yc"])), "ci_unindexed": np.exp(mi.conf_int().loc["yc"]).tolist(),
        "or_indexed": float(np.exp(b_ix)), "ci_indexed": [float(np.exp(b_ix - 1.96 * se_ix)), float(np.exp(b_ix + 1.96 * se_ix))],
        "p_interaction": float(mi.pvalues["yc:indexed"])}
    import patsy
    agg = py.groupby(["year", "domain"]).interv.agg(["sum", "count"]).reset_index()
    agg["yc"] = agg.year - 2020
    _, X = patsy.dmatrices("sum ~ yc + C(domain)", agg, return_type="dataframe")
    mq = sm.GLM(np.column_stack([agg["sum"], agg["count"] - agg["sum"]]), X, family=sm.families.Binomial()).fit()
    mu_q, n_q, k_q = mq.fittedvalues.values, agg["count"].values, agg["sum"].values
    phi = float(np.sum((k_q - n_q * mu_q) ** 2 / (n_q * mu_q * (1 - mu_q))) / mq.df_resid)
    se_q = float(mq.bse["yc"]) * np.sqrt(max(phi, 1.0))
    b_q = float(mq.params["yc"])
    rob["quasibinomial_year_domain"] = {"or": float(np.exp(b_q)), "ci": [float(np.exp(b_q - 1.96 * se_q)), float(np.exp(b_q + 1.96 * se_q))],
                                        "dispersion": phi}
    mc = smf.glm("interv ~ C(year) + C(domain)", data=py, family=sm.families.Binomial()).fit()
    rob["year_categorical_or_vs_2020"] = {int(y): {"or": float(np.exp(mc.params[f"C(year)[T.{y}]"])),
                                                   "ci": np.exp(mc.conf_int().loc[f"C(year)[T.{y}]"]).tolist()} for y in range(2021, 2026)}
    maturity["trend_robustness"] = rob
    log("a2.maturity", share=round(maturity["interventional_share_primary"], 4),
        or_crude=round(trend["or_per_year_crude"], 3), or_adj=round(trend["or_per_year_domain_adjusted"], 3),
        p_adj=trend["p_adjusted"], inter_p=trend["interaction_p"])

    reuse = {}
    yrs = R_[R_.year.between(2020, 2025)]
    per_year = {DATASET_LABEL[c]: yrs.groupby("year")[c].sum().to_dict() for c in INDIVIDUAL_DATASETS + ["data_gbd"]}
    per_year["Any individual-level dataset"] = yrs.groupby("year").any_dataset.sum().to_dict()
    per_year["Mendelian randomization"] = yrs.groupby("year").mr.sum().to_dict()
    per_year["All research papers"] = yrs.groupby("year").size().to_dict()
    pd.DataFrame(per_year).to_csv(SRC / "dataset_mentions_by_year.csv")
    tot = {k: sum(v.values()) for k, v in per_year.items()}
    top = sorted(((k, v) for k, v in tot.items() if k not in ("All research papers", "Any individual-level dataset",
                                                             "Mendelian randomization", "GBD")), key=lambda kv: -kv[1])
    reuse["dataset_totals_2020_2025"] = dict(top)
    reuse["growth_2025_vs_2020"] = {k: (v.get(2025, 0) / v.get(2020, 0) if v.get(2020, 0) else None) for k, v in per_year.items()}
    obs = R_[(R_.group == "observational") & R_.year.between(2020, 2025)]
    oy = obs.groupby("year").agg(n=("record_id", "size"), ds=("any_dataset", "sum"), mr=("mr", "sum"))
    oy["public"] = obs.assign(pub=(obs.any_dataset | obs.mr)).groupby("year").pub.sum()
    reuse["observational_by_year"] = oy.to_dict()
    d_obs = oy.n[2025] - oy.n[2020]; d_pub = oy.public[2025] - oy.public[2020]
    reuse["observational_increase_2020_2025"] = int(d_obs)
    reuse["public_resource_increase_2020_2025"] = int(d_pub)
    reuse["share_of_observational_increase_from_public_resources"] = float(d_pub / d_obs)
    reuse["public_share_observational_2020"] = float(oy.public[2020] / oy.n[2020])
    reuse["public_share_observational_2025"] = float(oy.public[2025] / oy.n[2025])
    ob = obs.assign(pub=(obs.any_dataset | obs.mr).astype(int), yc=obs.year - 2020)
    mm = smf.glm("pub ~ yc", data=ob, family=sm.families.Binomial()).fit()
    reuse["public_share_or_per_year"] = float(np.exp(mm.params["yc"]))
    reuse["public_share_or_ci"] = np.exp(mm.conf_int().loc["yc"]).tolist(); reuse["public_share_p"] = float(mm.pvalues["yc"])
    dsu = R_[R_.any_dataset]
    shares = pd.Series({DATASET_LABEL[c]: dsu[c].sum() for c in INDIVIDUAL_DATASETS}).sort_values(ascending=False)
    reuse["dataset_paper_mentions_top"] = shares.head(8).to_dict()
    reuse["top3_share_of_dataset_papers"] = float(dsu[[c for c in INDIVIDUAL_DATASETS
                                                       if DATASET_LABEL[c] in shares.index[:3]]].any(axis=1).mean())
    reuse["n_dataset_papers"] = int(len(dsu))
    tp = R_.assign(pub=(R_.any_dataset | R_.mr)).groupby("topic").agg(n=("record_id", "size"), pub=("pub", "mean"))
    tp = tp.join(dyn.set_index("topic")[["rgr", "trend", "topic_label", "domain_name"]])
    tp = tp.join(ptx[["sum", "count"]].rename(columns={"sum": "n_interv", "count": "n_primary"}))
    tp["interv_share"] = tp.n_interv / tp.n_primary
    tp.to_csv(SRC / "topic_public_resource_growth.csv")
    rng = np.random.default_rng(SEED)
    def boot_rho(a, b, B=2000):
        a, b = np.asarray(a), np.asarray(b); bs = []
        for _ in range(B):
            ix = rng.integers(0, len(a), len(a)); bs.append(stats.spearmanr(a[ix], b[ix]).statistic)
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    rho1 = stats.spearmanr(tp.pub, tp.rgr); rho2 = stats.spearmanr(tp.pub, tp.interv_share)
    reuse["topic_rho_public_vs_growth"] = {"rho": float(rho1.statistic), "p": float(rho1.pvalue), "ci": boot_rho(tp.pub, tp.rgr)}
    reuse["topic_rho_public_vs_interventional"] = {"rho": float(rho2.statistic), "p": float(rho2.pvalue),
                                                   "ci": boot_rho(tp.pub, tp.interv_share)}
    reuse["n_topics"] = int(len(tp))
    em = tp[tp.trend == "emerging"]
    reuse["emerging_topics_public_share_median"] = float(em.pub.median())
    reuse["other_topics_public_share_median"] = float(tp[tp.trend != "emerging"].pub.median())
    log("a2.reuse", share_of_increase=round(reuse["share_of_observational_increase_from_public_resources"], 3),
        rho_growth=round(rho1.statistic, 3), rho_interv=round(rho2.statistic, 3))

    idx = (L.in_pubmed.fillna(False).astype(bool) & (L.mesh.fillna("").str.len() > 0)).values
    V = L[idx]
    ref = np.array([nlm_label(p) for p in V.pub_types])
    pred = np.array([text_rule_7class(t, a, k) for t, a, k in zip(V.title, V.abstract.fillna(""), V.keywords.fillna(""))])
    classes = ["Guideline", "SR/MA", "RCT", "Case report", "Commentary", "Narrative review", "Primary/other"]
    p, r, f, s = precision_recall_fscore_support(ref, pred, labels=classes, zero_division=0)
    validation = {"n": int(len(V)), "kappa": float(cohen_kappa_score(ref, pred)), "accuracy": float((ref == pred).mean()),
                  "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i]),
                                    "n_pred": int((pred == c).sum())} for i, c in enumerate(classes)},
                  "note": "Inherited text rules applied without publication types or OpenAlex type, scored against NLM publication types"}
    pd.crosstab(pd.Series(ref, name="NLM"), pd.Series(pred, name="text_rule")).to_csv(SRC / "rule_vs_nlm_confusion.csv")
    log("a2.rule_validation", n=validation["n"], kappa=round(validation["kappa"], 3),
        rct_p=round(validation["per_class"]["RCT"]["precision"], 3), rct_r=round(validation["per_class"]["RCT"]["recall"], 3))

    update_results("structure", structure)
    update_results("maturity", maturity)
    update_results("reuse", reuse)
    update_results("rule_validation", validation)


if __name__ == "__main__":
    main()
