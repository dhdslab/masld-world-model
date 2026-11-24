import os, re, json
import numpy as np
import pandas as pd
from common import *

P = pd.read_parquet(os.path.join(PROC, "pubmed.parquet")).assign(src="pubmed")
E = pd.read_parquet(os.path.join(PROC, "europepmc.parquet")).assign(src="europepmc")
O = pd.read_parquet(os.path.join(PROC, "openalex.parquet")).assign(src="openalex")
counts = {"identified_pubmed": len(P), "identified_europepmc": len(E), "identified_openalex": len(O)}
counts["identified_total"] = sum(counts.values())

def norm_title(t):
    t = re.sub(r"<[^>]+>", " ", str(t or "")).lower()
    return re.sub(r"[^a-z0-9]+", "", t)

for df in (P, E, O):
    for c in ("pmid", "doi", "pmcid"):
        if c not in df: df[c] = None
    df["pmid"] = df.pmid.astype("string").str.strip()
    df["doi"] = df.doi.astype("string").str.lower().str.strip().str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True)
    df["pmcid"] = df.pmcid.astype("string").str.upper().str.strip()
    df["tnorm"] = df.title.map(norm_title)

E["year"] = pd.to_numeric(E.pub_year, errors="coerce")
P["year"] = pd.to_numeric(P.pub_year, errors="coerce")
O["year"] = pd.to_numeric(O.pub_year, errors="coerce")

ALL = pd.concat([P, E, O], ignore_index=True, sort=False)
parent = np.arange(len(ALL))

def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[max(ra, rb)] = min(ra, rb)

keymap = {}
for i, r in enumerate(ALL[["pmid", "doi", "pmcid", "tnorm", "year"]].itertuples(index=False)):
    keys = []
    if isinstance(r.pmid, str) and r.pmid: keys.append("p:" + r.pmid)
    if isinstance(r.doi, str) and len(r.doi) > 6: keys.append("d:" + r.doi)
    if isinstance(r.pmcid, str) and r.pmcid.startswith("PMC"): keys.append("c:" + r.pmcid)
    if len(r.tnorm) >= 40 and not pd.isna(r.year): keys.append(f"t:{r.tnorm}|{int(r.year)}")
    for k in keys:
        if k in keymap: union(i, keymap[k])
        else: keymap[k] = i
ALL["cluster"] = [find(i) for i in range(len(ALL))]
counts["duplicates_removed"] = int(len(ALL) - ALL.cluster.nunique())
counts["unique_records"] = int(ALL.cluster.nunique())

PRI = {"pubmed": 0, "europepmc": 1, "openalex": 2}
ALL["pri"] = ALL.src.map(PRI)
ALL = ALL.sort_values(["cluster", "pri"])

def first_valid(s):
    s = s.dropna()
    s = s[s.astype(str).str.len() > 0] if len(s) else s
    return s.iloc[0] if len(s) else None

g = ALL.groupby("cluster", sort=False)
cols_pri = ["pmid", "doi", "pmcid", "title", "abstract", "journal", "pub_types", "mesh", "keywords",
            "language", "first_author_aff", "last_author_aff", "all_affiliations", "grants", "n_authors"]
U = pd.DataFrame({c: g[c].agg(first_valid) for c in cols_pri if c in ALL})
U["year"] = g["year"].agg(lambda s: s.dropna().min() if s.notna().any() else np.nan)
U["sources"] = g["src"].agg(lambda s: ";".join(sorted(set(s))))
U["in_pubmed"] = U.sources.str.contains("pubmed"); U["in_europepmc"] = U.sources.str.contains("europepmc")
U["in_openalex"] = U.sources.str.contains("openalex")
Ee = ALL[ALL.src == "europepmc"].groupby("cluster")
Oo = ALL[ALL.src == "openalex"].groupby("cluster")
U["epmc_source"] = Ee["epmc_source"].first()
U["epmc_id"] = Ee["epmc_id"].first()
U["is_preprint"] = Ee["epmc_source"].agg(lambda s: (s == "PPR").all()).reindex(U.index).fillna(False).astype(bool)
U["epmc_oa"] = Ee["is_open_access"].agg(lambda s: (s == "Y").any()).reindex(U.index)
U["epmc_in_epmc_fulltext"] = Ee["in_epmc"].agg(lambda s: (s == "Y").any()).reindex(U.index)
U["license"] = Ee["license"].agg(first_valid).reindex(U.index)
U["openalex_id"] = Oo["openalex_id"].first()
U["openalex_type"] = Oo["oa_type"].first()
U["oa_is_oa"] = Oo["is_oa"].agg(lambda s: bool(s.fillna(False).any())).reindex(U.index)
U["oa_status"] = Oo["oa_status"].first()
U["oa_url"] = Oo["oa_url"].first()
U["is_retracted_openalex"] = Oo["is_retracted"].agg(lambda s: bool(s.fillna(False).any())).reindex(U.index)
U["countries_all"] = Oo["countries"].first()
U["first_author_country_oa"] = Oo["first_author_countries"].first()
U["last_author_country_oa"] = Oo["last_author_countries"].first()
U["oa_topic"] = Oo["topic"].first(); U["oa_field"] = Oo["field"].first()
U["cited_by"] = Oo["cited_by"].max().reindex(U.index).fillna(Ee["cited_by"].max().reindex(U.index))
U["fwci"] = Oo["fwci"].max()
U["open_access"] = U.epmc_oa.fillna(False).astype(bool) | U.oa_is_oa.fillna(False).astype(bool)
U = U.reset_index(drop=True)
U.insert(0, "record_id", [f"MASLD-L{i:06d}" for i in range(1, len(U) + 1)])

U["text"] = (U.title.fillna("") + " \n " + U.abstract.fillna("") + " \n " + U.keywords.fillna(""))
U["text_l"] = U.text.str.lower()


def relevant(row):
    t = NASH_NOISE.sub(" ", row.text)
    if DISEASE_CI.search(t) or DISEASE_CS.search(t):
        return True
    m = str(row.mesh or "")
    return bool(re.search(r"Fatty Liver|Steatohepatitis", m))

U["excl"] = None
U.loc[U.title.fillna("").str.strip() == "", "excl"] = "E1_no_title"
U.loc[U.excl.isna() & ~U.year.between(2020, 2026), "excl"] = "E2_outside_window"
rel = U.apply(relevant, axis=1)
U.loc[U.excl.isna() & ~rel, "excl"] = "E3_not_steatotic_liver_disease"
pt = U.pub_types.fillna("")
erratum = (pt.str.contains(r"Published Erratum|Retraction of Publication|Retracted Publication|Expression of Concern", regex=True)
           | U.openalex_type.fillna("").isin(["erratum", "retraction", "paratext"])
           | U.is_retracted_openalex.fillna(False).astype(bool)
           | U.title.fillna("").str.match(r"^\s*(erratum|corrigendum|correction|retraction|retracted|notice of retraction)\b", case=False))
U.loc[U.excl.isna() & erratum, "excl"] = "E4_erratum_or_retraction"
NONRESEARCH_TYPES = {"peer-review", "dataset", "component", "supplementary-materials", "other", "grant",
                     "standard", "reference-entry", "libguides", "report-component"}
nonres = ((U.openalex_type.fillna("").isin(NONRESEARCH_TYPES) & ~U.in_pubmed & ~U.in_europepmc)
          | U.title.fillna("").str.match(r"^\s*(additional file|supplementary (file|material|table|figure)|decision letter|author response|"
                                         r"reviewer #?\d|review for \"|figure \d|table \d|data from|dataset)", case=False))
U.loc[U.excl.isna() & nonres, "excl"] = "E5_non_research_object"
for k, v in U.excl.value_counts().items():
    counts[f"excluded_{k}"] = int(v)
counts["excluded_E3_records_without_abstract"] = int(((U.excl == "E3_not_steatotic_liver_disease") & U.abstract.isna()).sum())
I = U[U.excl.isna()].copy()
counts["included_literature"] = len(I)

CONF_TITLE = re.compile(r"^\s*((MON|TUE|WED|THU|FRI|SAT|SUN|Mo|Tu|We|Th|Fr|Sa|Su|PO|OP|OS|LB|LBA|LBP|PS|AB|ABS|EP|OC|PP)[\s\-‐]?\d{1,5}[A-Z]?|\d{2,5}[\s\-‐](OR|LB|P|PUB|LBA))\b[\s:.\-‐]")
ot = I.openalex_type.fillna(""); ptp = I.pub_types.fillna("")
I["pub_format"] = np.select(
    [I.is_preprint | ot.eq("preprint") | I.epmc_source.eq("PPR"),
     ot.isin(["conference-abstract"]) | I.title.fillna("").str.match(CONF_TITLE) | ptp.str.contains("Congress|Meeting Abstract", regex=True),
     ot.isin(["conference-paper"]),
     ot.isin(["dissertation"]),
     ot.isin(["book-chapter", "book"]) | ptp.str.contains("Book", regex=False)],
    ["Preprint", "Conference abstract", "Conference paper", "Dissertation/thesis", "Book/chapter"], "Journal article")
for k, v in I.pub_format.value_counts().items():
    counts["format_" + k] = int(v)

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


def rx(p): return re.compile(mixcase(p))
tl, ab = I.title.fillna(""), I.abstract.fillna("")
T = I.text
PT = I.pub_types.fillna("")
OT = I.openalex_type.fillna("")
has_abs = ab.str.len() > 100

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
 "mr": rx(r"mendelian randomi[sz]ation"),
 "animal": rx(r"\bmice\b|\bmouse\b|murine|\brats?\b|zebrafish|hamsters?|rabbits?|\bpigs?\b|minipig|primates?|"
              r"drosophila|c\. elegans|caenorhabditis|hepatocyte[\s\-]specific|liver[\s\-]specific|transgenic|conditional (knockout|deletion)|"
              r"\bducks?\b|broilers?|laying hens|\bcattle\b|dairy cows?|largemouth bass|grass carp|tilapia|zebrafish larvae|"
              r"high[\s\-]fat diet[\s\-]?(fed|induced)|\bHFD\b|knock[\s\-]?out|"
              r"\bdb/db\b|\bob/ob\b|methionine[\s\-]choline|\bMCD\b diet|CDAHFD|western diet[\s\-]fed"),
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
 "editorial_title": rx(r"^\s*(editorial|comment|reply|response|letter|correspondence|in memoriam|re:|author'?s? reply)|"
                       r"\b(commentary|editorial|reply to|response to|letter to the editor)\b"),
}

def classify(i):
    t, a, p, o = tl.at[i], ab.at[i], PT.at[i], OT.at[i]
    x = T.at[i]
    if re.search(r"Guideline|Practice Guideline|Consensus Development Conference", p) or R["guideline"].search(t):
        return "Guideline/consensus"
    if re.search(r"Meta-Analysis|Systematic Review", p) or R["sr_ma"].search(t) or R["sr_ma"].search(a[:800]):
        return "Systematic review/meta-analysis"
    if re.search(r"Editorial|Comment|Letter|News|Biography|Interview|Portrait", p) and not re.search(r"Randomized|Clinical Trial", p):
        return "Editorial/comment/letter"
    if R["editorial_title"].search(t) and not has_abs.at[i]:
        return "Editorial/comment/letter"
    if re.search(r"Case Reports", p) or R["case_report"].search(t):
        return "Case report/series"
    if re.search(r"Clinical Trial Protocol", p) or R["protocol"].search(t):
        return "Trial/study protocol"
    if re.search(r"\bReview\b", p) and not re.search(r"Randomized|Clinical Trial", p):
        return "Narrative review"
    if (R["review_title"].search(t) and not R["randomized"].search(a)
            and not R["cohort"].search(a[:600]) and not R["animal"].search(a)):
        return "Narrative review"
    if re.search(r"Randomized Controlled Trial", p) or (R["randomized"].search(x) and R["trial"].search(x) and R["human"].search(x)
                                                        and not R["animal"].search(x)):
        return "Randomized controlled trial"
    if re.search(r"Clinical Trial|Controlled Clinical Trial", p) or (R["nonrand_trial"].search(x) and R["trial"].search(x) and R["human"].search(x)):
        return "Non-randomized clinical trial"
    if R["mr"].search(x):
        return "Mendelian randomization"
    if R["animal"].search(x) or R["invitro"].search(x):
        return "Basic/experimental (animal or in vitro)"
    if R["insilico"].search(x):
        return "Basic/experimental (in silico)"
    if not has_abs.at[i]:
        if o in ("editorial", "letter"): return "Editorial/comment/letter"
        if o == "review" or R["review_title"].search(t): return "Narrative review"
        if o == "preprint" or I.is_preprint.at[i]: return "Unclassified (title only)"
        return "Unclassified (title only)"
    if R["case_control"].search(x): return "Case-control"
    if R["diag_title"].search(t): return "Diagnostic/prognostic modelling"
    if R["xs_explicit"].search(x): return "Cross-sectional"
    if R["cohort_explicit"].search(x): return "Cohort"
    if R["diagnostic"].search(x): return "Diagnostic/prognostic modelling"
    if R["cohort"].search(x): return "Cohort"
    if R["cross_sectional"].search(x): return "Cross-sectional"
    return "Other primary study"

I["evidence_type"] = [classify(i) for i in I.index]
TIER = {"Guideline/consensus": 1, "Systematic review/meta-analysis": 1, "Randomized controlled trial": 2,
        "Non-randomized clinical trial": 3, "Mendelian randomization": 3, "Cohort": 3,
        "Case-control": 4, "Cross-sectional": 4, "Diagnostic/prognostic modelling": 4,
        "Case report/series": 5, "Basic/experimental (animal or in vitro)": 5, "Basic/experimental (in silico)": 5,
        "Narrative review": 5, "Editorial/comment/letter": 5, "Trial/study protocol": 0,
        "Other primary study": 0, "Unclassified (title only)": 0}
I["evidence_tier"] = I.evidence_type.map(TIER)

TOPICS = {
 "Epidemiology/burden": r"prevalence|incidence|burden|global burden|\bGBD\b|epidemiolog|trends?\b|mortality rate|disability[\s\-]adjusted",
 "Risk factors/exposures": r"risk factor|association (of|between)|associated with|exposure|lifestyle factor|smoking|sleep|socioeconomic|air pollution",
 "Genetics/omics": r"\bPNPLA3\b|\bTM6SF2\b|\bHSD17B13\b|\bMBOAT7\b|\bGCKR\b|polygenic|\bGWAS\b|genome[\s\-]wide|transcriptom|proteom|metabolom|lipidom|epigen|methylation|single[\s\-]cell|multi[\s\-]?omics|exome",
 "Non-invasive diagnosis/biomarkers": r"\bFIB[\s\-]?4\b|\bNFS\b|fibrosis score|\bELF\b|enhanced liver fibrosis|biomarker|\bAPRI\b|\bFAST score|\bAGILE|non[\s\-]?invasive test|\bNITs?\b|\bCK[\s\-]?18\b|\bPRO[\s\-]?C3\b",
 "Imaging/elastography": r"elastograph|fibroscan|transient elastography|\bVCTE\b|\bLSM\b|\bCAP\b|controlled attenuation|MRI[\s\-]?PDFF|proton density fat|magnetic resonance elastograph|\bMRE\b|ultrasound|ultrasonograph|computed tomography|\bCT\b",
 "Histopathology/fibrosis": r"biops|histolog|patholog|\bNAS\b|activity score|fibrosis stage|\bF[2-4]\b|cirrhosis|ballooning|collagen",
 "Mechanisms/pathophysiology": r"lipotoxic|insulin resistance|de novo lipogenesis|inflammat|oxidative stress|mitochondri|endoplasmic reticulum stress|ferroptosis|autophagy|pyroptosis|hepatic stellate|macrophage|kupffer|signaling pathway|\bNLRP3\b|\bAMPK\b|\bSREBP|\bPPAR",
 "Gut microbiome": r"microbio|microbiota|gut[\s\-]liver axis|dysbiosis|short[\s\-]chain fatty|bile acid",
 "Pharmacotherapy": r"resmetirom|semaglutide|tirzepatide|liraglutide|dulaglutide|survodutide|retatrutide|efruxifermin|pegozafermin|lanifibranor|obeticholic|elafibranor|saroglitazar|pioglitazone|vitamin e|\bSGLT[\s\-]?2|empagliflozin|dapagliflozin|statin|metformin|\bGLP[\s\-]?1|\bFGF[\s\-]?21|\bTHR[\s\-]?β|thyroid hormone receptor|\bFXR\b|aramchol|denifanstat|\bFASN\b|drug treatment|pharmacotherap",
 "Lifestyle/diet/exercise": r"lifestyle intervention|weight loss|\bdiet(ary)?\b|mediterranean|exercise|physical activity|caloric restriction|intermittent fasting|coffee|alcohol consumption|nutrition",
 "Bariatric/metabolic surgery": r"bariatric|metabolic surgery|sleeve gastrectomy|gastric bypass|roux[\s\-]en[\s\-]y",
 "Liver cancer (HCC)": r"hepatocellular carcinoma|\bHCC\b|liver cancer|hepatocarcinogen",
 "Cardiovascular disease": r"cardiovascular|coronary|atherosclero|myocardial infarction|heart failure|stroke|arterial stiffness|atrial fibrillation|carotid",
 "Diabetes/obesity/metabolic": r"diabetes|\bT2DM\b|\bT2D\b|obesity|overweight|metabolic syndrome|insulin resistance|\bBMI\b|adipos",
 "Kidney disease": r"chronic kidney|\bCKD\b|renal function|albuminuria|\beGFR\b",
 "Extrahepatic cancers/other outcomes": r"colorectal cancer|extrahepatic cancer|breast cancer|cancer incidence|dementia|depression|osteoporosis|sarcopenia",
 "Pediatric/adolescent": r"\bchild|pediatric|paediatric|adolescen|infant|youth|school[\s\-]age",
 "Pregnancy/women's health": r"pregnan|gestational|polycystic ovary|\bPCOS\b|menopaus|maternal",
 "Lean/non-obese": r"\blean\b|non[\s\-]?obese",
 "Alcohol overlap (MetALD/ALD)": r"\bMetALD\b|alcohol[\s\-]related liver|alcoholic liver|alcohol intake|alcohol use",
 "Liver transplantation": r"liver transplant",
 "Cirrhosis/decompensation/outcomes": r"cirrhosis|decompensat|liver[\s\-]related (events|outcomes|mortality)|portal hypertension|varice",
 "AI/machine learning": r"machine learning|deep learning|artificial intelligence|neural network|random forest|xgboost|\bLLM\b|large language model|radiomics|convolutional",
 "Nomenclature/definition": r"nomenclature|new definition|redefin|renam|MAFLD versus NAFLD|MASLD versus NAFLD|terminolog|diagnostic criteria",
 "Health economics/policy": r"cost[\s\-]effective|economic|health care cost|quality[\s\-]adjusted|\bQALY|policy|awareness|health literacy",
 "Traditional/herbal medicine": r"traditional chinese|herbal|\bTCM\b|decoction|phytochemical|natural product|polyphenol|flavonoid|extract of",
 "COVID-19": r"covid|sars[\s\-]cov[\s\-]?2",
}
for k, p in TOPICS.items():
    I["topic_" + re.sub(r"[^a-z0-9]+", "_", k.lower()).strip("_")] = T.str.contains(mixcase(p), regex=True)

NOMEN = {"NAFLD/NASH": r"\bNAFLD\b|\bNAFL\b|\bNASH\b|non[\s\-]?alcoholic (fatty|steatohep)",
         "MAFLD": r"\bMAFLD\b|metabolic[\s\-](dysfunction[\s\-])?associated fatty liver",
         "MASLD/MASH/MetALD/SLD": r"\bMASLD\b|\bMASH\b|\bMetALD\b|steatotic liver|metabolic dysfunction[\s\-]associated steatohepatitis"}
for k, p in NOMEN.items():
    I["nom_" + k.split("/")[0].lower()] = T.str.contains(mixcase(p), regex=True)

DATA = {"NHANES": r"\bNHANES\b|national health and nutrition examination", "UK Biobank": r"uk biobank|\bUKB\b",
        "Korean NHIS": r"\bNHIS\b|national health insurance service|national health screening", "KNHANES": r"\bKNHANES\b|korea national health and nutrition",
        "JMDC": r"\bJMDC\b", "GBD": r"global burden of disease|\bGBD\b", "FinnGen": r"finngen", "Biobank Japan": r"biobank japan",
        "China Kadoorie": r"kadoorie", "MIMIC": r"\bMIMIC\b", "Framingham": r"framingham", "Rotterdam Study": r"rotterdam study",
        "ARIC": r"\bARIC\b|atherosclerosis risk in communities", "MESA": r"\bMESA\b|multi[\s\-]ethnic study of atherosclerosis",
        "Kangbuk Samsung": r"kangbuk samsung", "Swedish registers (ESPRESSO etc.)": r"espresso|swedish (national|patient) regist",
        "NASH CRN/NIDDK": r"\bNASH CRN\b|clinical research network", "LITMUS": r"\bLITMUS\b", "All of Us": r"all of us research",
        "TriNetX": r"trinetx", "SEER": r"\bSEER\b", "CPRD": r"\bCPRD\b|clinical practice research datalink",
        "Taiwan NHIRD": r"\bNHIRD\b|taiwan.{0,30}national health insurance research"}
for k, p in DATA.items():
    I["data_" + re.sub(r"[^a-z0-9]+", "_", k.lower()).strip("_")] = T.str.contains(mixcase(p), regex=True)

I["ai_ml"] = I["topic_ai_machine_learning"]

CN = {"china": "CN", "p.r. china": "CN", "pr china": "CN", "united states": "US", "usa": "US", "u.s.a": "US",
      "korea": "KR", "japan": "JP", "italy": "IT", "germany": "DE", "united kingdom": "GB", "uk": "GB", "england": "GB",
      "scotland": "GB", "france": "FR", "spain": "ES", "india": "IN", "iran": "IR", "brazil": "BR", "canada": "CA",
      "australia": "AU", "taiwan": "TW", "egypt": "EG", "turkey": "TR", "türkiye": "TR", "netherlands": "NL",
      "sweden": "SE", "switzerland": "CH", "belgium": "BE", "mexico": "MX", "poland": "PL", "israel": "IL",
      "saudi arabia": "SA", "pakistan": "PK", "greece": "GR", "denmark": "DK", "austria": "AT", "portugal": "PT",
      "finland": "FI", "norway": "NO", "singapore": "SG", "hong kong": "HK", "thailand": "TH", "malaysia": "MY",
      "indonesia": "ID", "vietnam": "VN", "russia": "RU", "romania": "RO", "argentina": "AR", "chile": "CL",
      "colombia": "CO", "nigeria": "NG", "south africa": "ZA", "ireland": "IE", "czech": "CZ", "hungary": "HU",
      "new zealand": "NZ", "qatar": "QA", "united arab emirates": "AE", "lebanon": "LB", "jordan": "JO",
      "bangladesh": "BD", "sri lanka": "LK", "philippines": "PH", "serbia": "RS", "croatia": "HR", "tunisia": "TN",
      "morocco": "MA", "iraq": "IQ", "kuwait": "KW", "peru": "PE", "ukraine": "UA", "slovenia": "SI", "slovakia": "SK",
      "bulgaria": "BG", "lithuania": "LT", "cuba": "CU", "ethiopia": "ET", "kenya": "KE", "nepal": "NP"}
US_STATES = r"\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\s+\d{5}"
def aff_country(a):
    if not isinstance(a, str) or not a: return None
    seg = a.lower().split(";")[0]
    tail = seg.split(",")[-3:]
    for part in reversed(tail):
        part = re.sub(r"[^a-z\s\.ü]", " ", part).strip()
        part = re.sub(r"\s+", " ", part)
        for name in sorted(CN, key=len, reverse=True):
            if re.search(r"\b" + re.escape(name) + r"\b", part):
                return CN[name]
    if re.search(US_STATES, a): return "US"
    return None
fa = I.first_author_country_oa.fillna("").str.split(";").str[0].replace("", None)
I["first_author_country"] = fa.where(fa.notna(), I.first_author_aff.map(aff_country))
la = I.last_author_country_oa.fillna("").str.split(";").str[0].replace("", None)
I["last_author_country"] = la.where(la.notna(), I.last_author_aff.map(aff_country))

I["access_level"] = np.where(I.open_access & I.pmcid.notna(), "OA (PMC full text eligible)",
                    np.where(I.open_access, "OA (publisher/repository; abstract + OA link)",
                    np.where(I.abstract.fillna("").str.len() > 100, "Non-OA (abstract only)", "Non-OA (metadata only)")))

keep = [c for c in I.columns if c not in ("text", "text_l", "excl")]
I[keep].to_parquet(os.path.join(PROC, "literature_included.parquet"), index=False)
U.drop(columns=["text", "text_l"]).to_parquet(os.path.join(PROC, "literature_all_unique_with_screening.parquet"), index=False)
counts.update({"included_with_abstract": int((I.abstract.fillna("").str.len() > 100).sum()),
               "included_open_access": int(I.open_access.sum()),
               "included_oa_with_pmcid": int((I.open_access & I.pmcid.notna()).sum()),
               "included_preprints": int(I.is_preprint.sum())})
for k, v in I.access_level.value_counts().items(): counts["access_" + k] = int(v)
for k, v in I.evidence_type.value_counts().items(): counts["type_" + k] = int(v)
json.dump(counts, open(os.path.join(PROC, "flow_counts_literature.json"), "w"), indent=1)
log_event("merge_screen_classify.done", **{k: v for k, v in counts.items()})
