import os, json, re
import numpy as np
import pandas as pd
from common import *

TAB = os.path.join(ROOT, "tables"); os.makedirs(TAB, exist_ok=True)
flow = json.load(open(os.path.join(PROC, "flow_counts_literature.json")))
tables = {}

L = pd.read_parquet(os.path.join(PROC, "literature_included.parquet"))
if os.path.exists(os.path.join(PROC, "fulltext_manifest.parquet")):
    M = pd.read_parquet(os.path.join(PROC, "fulltext_manifest.parquet"))
    L = L.merge(M[["pmcid", "status"]].drop_duplicates("pmcid").rename(columns={"status": "fulltext_status"}), on="pmcid", how="left")
    L.loc[~L.open_access, "fulltext_status"] = None
else:
    L["fulltext_status"] = None
L["fulltext_collected"] = L.fulltext_status.eq("downloaded")
L["content_collected"] = np.where(L.fulltext_collected, "Full text (OA, JATS XML)",
                          np.where(L.abstract.fillna("").str.len() > 100, "Abstract", "Metadata only"))
flow["lit_fulltext_collected"] = int(L.fulltext_collected.sum())
flow["lit_abstract_only"] = int((L.content_collected == "Abstract").sum())
flow["lit_metadata_only"] = int((L.content_collected == "Metadata only").sum())
flow["lit_oa_abstract_only"] = int(((L.content_collected == "Abstract") & L.open_access).sum())
flow["lit_nonoa_abstract_only"] = int(((L.content_collected == "Abstract") & ~L.open_access).sum())
L["year"] = L.year.astype(int)

tables["T01_literature_year_by_evidence_type"] = pd.crosstab(L.evidence_type, L.year, margins=True, margins_name="Total")
tables["T02_literature_evidence_tier_by_year"] = pd.crosstab(L.evidence_tier, L.year, margins=True, margins_name="Total")
src = L.sources.value_counts().rename_axis("source_combination").reset_index(name="n")
tables["T03_literature_source_overlap"] = src
tables["T04_literature_content_collected_by_year"] = pd.crosstab(L.content_collected, L.year, margins=True, margins_name="Total")
tables["T05_literature_open_access_by_year"] = (L.groupby("year").agg(n=("record_id", "size"), open_access=("open_access", "sum"),
                                                fulltext=("fulltext_collected", "sum"), preprints=("pub_format", lambda x: (x == "Preprint").sum()))
                                               .assign(oa_pct=lambda d: (100 * d.open_access / d.n).round(1)))
nom = L.groupby("year")[["nom_nafld", "nom_mafld", "nom_masld"]].mean().mul(100).round(1)
nom.columns = ["NAFLD/NASH_%", "MAFLD_%", "MASLD/MASH/MetALD/SLD_%"]
nq = L.assign(q=pd.to_datetime(L.year.astype(str)) )
tables["T06_nomenclature_share_by_year"] = nom
tcols = [c for c in L.columns if c.startswith("topic_")]
tp = L.groupby("year")[tcols].sum().T
tp["Total"] = tp.sum(axis=1); tp["pct_of_corpus"] = (100 * tp.Total / len(L)).round(1)
tp.index = tp.index.str.replace("topic_", "")
tables["T07_topics_by_year"] = tp.sort_values("Total", ascending=False)
dcols = [c for c in L.columns if c.startswith("data_")]
dt = L.groupby("year")[dcols].sum().T; dt["Total"] = dt.sum(axis=1)
dt.index = dt.index.str.replace("data_", "")
tables["T08_named_data_resources_by_year"] = dt.sort_values("Total", ascending=False)
fc = L.first_author_country.fillna("Unknown").value_counts()
tables["T09_first_author_country"] = fc.rename_axis("country_iso2").reset_index(name="n").assign(pct=lambda d: (100 * d.n / len(L)).round(2))
tables["T10_top_journals"] = L.journal.fillna("Unknown").value_counts().head(50).rename_axis("journal").reset_index(name="n")
ev_c = L.groupby("evidence_type").agg(n=("record_id", "size"), median_citations=("cited_by", "median"),
                                      oa_pct=("open_access", "mean"), with_abstract=("abstract", lambda s: s.notna().sum()))
ev_c["oa_pct"] = (100 * ev_c.oa_pct).round(1)
tables["T11_evidence_type_summary"] = ev_c.sort_values("n", ascending=False)
kr = L[L.first_author_country.eq("KR")]
tables["T12_korea_first_author_by_year_type"] = pd.crosstab(kr.evidence_type, kr.year, margins=True, margins_name="Total")
tables["T13b_publication_format_by_year"] = pd.crosstab(L.pub_format, L.year, margins=True, margins_name="Total")
tables["T13_language"] = L.language.fillna("unknown").str.split(";").str[0].value_counts().rename_axis("language").reset_index(name="n")

CT = pd.read_parquet(os.path.join(PROC, "clinicaltrials.parquet"))
flow["trials_identified"] = len(CT)
CT["relevant"] = (CT.title.fillna("") + " " + CT.conditions.fillna("") + " " + CT.brief_summary.fillna("")).map(is_relevant)
flow["trials_excluded_not_relevant"] = int((~CT.relevant).sum())
CT = CT[CT.relevant].copy()
flow["trials_included"] = len(CT)
CT["year"] = CT.first_submitted.str[:4].astype(int)
DRUG = {"GLP-1/incretin": r"semaglutide|tirzepatide|liraglutide|dulaglutide|survodutide|retatrutide|cotadutide|efinopegdutide|pemvidutide|mazdutide|GLP-1|glucagon",
        "THR-beta agonist": r"resmetirom|VK2809|ASC41|TERN-501|thyroid hormone receptor",
        "FGF21/FGF19 analogue": r"efruxifermin|pegozafermin|aldafermin|FGF21|FGF19|BOS-580",
        "PPAR agonist": r"lanifibranor|elafibranor|saroglitazar|pioglitazone|seladelpar|PPAR",
        "FXR agonist": r"obeticholic|tropifexor|cilofexor|vonafexor|FXR",
        "SGLT2 inhibitor": r"empagliflozin|dapagliflozin|canagliflozin|ertugliflozin|SGLT",
        "Lipogenesis inhibitor (FASN/ACC/DGAT/SCD)": r"denifanstat|firsocostat|TVB-2640|ervogastat|aramchol|DGAT|FASN|ACC inhibitor",
        "Statin/lipid-lowering": r"statin|ezetimibe|omega-3|icosapent",
        "Vitamin E/antioxidant": r"vitamin e|tocopherol|silymarin|antioxidant",
        "Probiotic/microbiome": r"probiotic|synbiotic|prebiotic|fecal microbiota|faecal microbiota|FMT",
        "Lifestyle/diet/exercise": r"diet|exercise|lifestyle|physical activity|fasting|weight loss program|behavio",
        "Bariatric/endoscopic": r"bariatric|sleeve|bypass|endoscopic|intragastric balloon|duodenal",
        "Diagnostic/imaging": r"elastograph|fibroscan|MRI|PDFF|ultrasound|biomarker|screening|diagnos"}
txt = CT.title.fillna("") + " " + CT.interventions.fillna("")
for k, p in DRUG.items():
    CT["iv_" + k] = txt.str.contains(p, case=False, regex=True)
tables["T14_trials_by_year_type_phase"] = pd.crosstab([CT.study_type.fillna("NA"), CT.phases.replace("", "NA").fillna("NA")], CT.year, margins=True, margins_name="Total")
ivc = [c for c in CT.columns if c.startswith("iv_")]
tiv = CT.groupby("year")[ivc].sum().T; tiv["Total"] = tiv.sum(axis=1); tiv.index = tiv.index.str.replace("iv_", "")
tables["T15_trials_intervention_class_by_year"] = tiv.sort_values("Total", ascending=False)
tables["T16_trials_status_sponsor"] = pd.crosstab(CT.status, CT.sponsor_class, margins=True, margins_name="Total")
cc = CT.countries.fillna("").str.split(";").explode().replace("", np.nan).dropna().value_counts().head(40)
tables["T17_trials_countries"] = cc.rename_axis("country").reset_index(name="n_trials")

G = pd.read_parquet(os.path.join(PROC, "nih_reporter.parquet"))
flow["grants_identified"] = len(G)
G["relevant"] = (G.title.fillna("") + " " + G.abstract.fillna("")).map(is_relevant)
flow["grants_excluded_not_relevant"] = int((~G.relevant).sum())
G = G[G.relevant].copy()
flow["grants_included_applications"] = len(G)
flow["grants_included_core_projects"] = int(G.core_project_num.nunique())
tables["T18_nih_grants_by_fiscal_year"] = G.groupby("fiscal_year").agg(
    applications=("appl_id", "size"), unique_core_projects=("core_project_num", "nunique"),
    total_award_usd=("award_amount", "sum")).assign(total_award_usd_millions=lambda d: (d.total_award_usd / 1e6).round(1))
tables["T19_nih_grants_by_activity_code"] = G.activity_code.value_counts().rename_axis("activity_code").reset_index(name="applications")
tables["T20_nih_grants_by_institute"] = G.ic.value_counts().rename_axis("NIH_IC").reset_index(name="applications")

F = pd.read_parquet(os.path.join(PROC, "openfda_labels.parquet"))
flow["labels_identified"] = len(F)
flow["labels_with_disease_indication"] = int(F.indication_mentions_disease.sum()) if len(F) else 0
tables["T21_fda_labels"] = F[["effective_time", "brand_name", "generic_name", "manufacturer", "indication_mentions_disease"]].sort_values("effective_time")

N = pd.read_parquet(os.path.join(PROC, "news.parquet")) if os.path.exists(os.path.join(PROC, "news.parquet")) else pd.DataFrame()
if len(N):
    flow["news_after_window_and_dedup"] = len(N)
    N["relevant"] = [is_relevant(t, korean=(e == "KR")) for t, e in zip(N.title, N.edition)]
    flow["news_excluded_not_relevant"] = int((~N.relevant).sum())
    N = N[N.relevant].copy()
    flow["news_included"] = len(N)
    flow["news_included_google"] = int((N.stream == "google_news").sum()); flow["news_included_gdelt"] = int((N.stream == "gdelt").sum())
    N["year"] = N.date.dt.year
    NT = {"Drug approval/regulatory": r"FDA|EMA|approv|approval|승인|허가",
          "Resmetirom (Rezdiffra)": r"resmetirom|rezdiffra|레즈디프라",
          "GLP-1/incretin (semaglutide etc.)": r"semaglutide|wegovy|ozempic|tirzepatide|mounjaro|zepbound|GLP-1|위고비|세마글루타이드",
          "Nomenclature (MASLD rename)": r"\bMASLD\b|\bMASH\b|renam|new name|명칭",
          "Epidemiology/burden": r"prevalence|million|rising|epidemic|burden|cases|유병|증가",
          "Diet/lifestyle": r"diet|food|coffee|exercise|sugar|alcohol|식단|운동|음식|술",
          "Cancer/HCC": r"cancer|carcinoma|암",
          "Children": r"child|kids|teen|adolescen|소아|어린이|청소년",
          "Business/pipeline": r"stock|shares|acquisition|acquire|deal|trial results|phase 3|phase 2|biotech|market"}
    for k, p in NT.items():
        N["nt_" + k] = N.title.fillna("").str.contains(p, case=False, regex=True)
    tables["T22_news_by_year_stream_edition"] = pd.crosstab([N.stream, N.edition.fillna("NA")], N.year, margins=True, margins_name="Total")
    ntc = [c for c in N.columns if c.startswith("nt_")]
    tn = N.groupby("year")[ntc].sum().T; tn["Total"] = tn.sum(axis=1); tn.index = tn.index.str.replace("nt_", "")
    tables["T23_news_themes_by_year"] = tn.sort_values("Total", ascending=False)
    tables["T24_news_top_outlets"] = N.outlet.fillna("Unknown").value_counts().head(50).rename_axis("outlet").reset_index(name="n")
    N.to_parquet(os.path.join(PROC, "news_included.parquet"), index=False)

CT.to_parquet(os.path.join(PROC, "clinicaltrials_included.parquet"), index=False)
G.to_parquet(os.path.join(PROC, "nih_reporter_included.parquet"), index=False)
L.to_parquet(os.path.join(PROC, "literature_included_final.parquet"), index=False)

def url_lit(r):
    if isinstance(r.doi, str): return "https://doi.org/" + r.doi
    if isinstance(r.pmid, str): return "https://pubmed.ncbi.nlm.nih.gov/" + r.pmid
    return None
parts = [pd.DataFrame({"stream": np.where(L.pub_format.eq("Preprint"), "Preprint", "Literature (journal, conference, thesis, book)"),
                       "item_id": L.record_id, "year": L.year, "title": L.title, "class": L.evidence_type,
                       "content_collected": L.content_collected, "url": [url_lit(r) for r in L.itertuples()]}),
         pd.DataFrame({"stream": "Clinical trial registration", "item_id": CT.nct_id, "year": CT.year, "title": CT.title,
                       "class": CT.study_type.fillna("") + " " + CT.phases.fillna(""), "content_collected": "Registry record",
                       "url": "https://clinicaltrials.gov/study/" + CT.nct_id}),
         pd.DataFrame({"stream": "Research funding (NIH)", "item_id": G.appl_id.astype(str), "year": G.fiscal_year, "title": G.title,
                       "class": G.activity_code, "content_collected": "Project abstract",
                       "url": "https://reporter.nih.gov/project-details/" + G.appl_id.astype(str)}),
         pd.DataFrame({"stream": "Regulatory label (FDA)", "item_id": F.set_id, "year": pd.to_numeric(F.effective_time.str[:4], errors="coerce"),
                       "title": F.brand_name + " (" + F.generic_name + ")", "class": np.where(F.indication_mentions_disease, "Disease indication", "Mention in label"),
                       "content_collected": "Label excerpt", "url": "https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid=" + F.set_id.astype(str)})]
if len(N):
    parts.append(pd.DataFrame({"stream": "News", "item_id": [f"MASLD-N{i:06d}" for i in range(1, len(N) + 1)], "year": N.year,
                               "title": N.title, "class": N.stream + ":" + N.edition.fillna(""), "content_collected": "Headline + link",
                               "url": N.url}))
MI = pd.concat(parts, ignore_index=True)
MI.to_parquet(os.path.join(PROC, "evidence_master_index.parquet"), index=False)
MI.to_csv(os.path.join(PROC, "evidence_master_index.csv.gz"), index=False, encoding="utf-8-sig")
tables["T00_all_streams_by_year"] = pd.crosstab(MI.stream, MI.year, margins=True, margins_name="Total")
flow["master_index_items"] = len(MI)

for k, t in tables.items():
    t.to_csv(os.path.join(TAB, k + ".csv"), encoding="utf-8-sig")
try:
    with pd.ExcelWriter(os.path.join(TAB, "MASLD_evidence_tables.xlsx")) as xw:
        for k, t in sorted(tables.items()):
            t.to_excel(xw, sheet_name=k[:31])
except ImportError:
    pass
json.dump(flow, open(os.path.join(PROC, "flow_counts_all.json"), "w"), indent=1)
log_event("quantify.done", **flow)
