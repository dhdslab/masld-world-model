from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

SEED = 20261108

REPO = Path(__file__).resolve().parents[1]
PKG = Path(os.environ.get("MASLD_OUTPUT_DIR", REPO)).resolve()
PROTOCOL = REPO / "protocol"
CORPUS_DIR = Path(os.environ.get("MASLD_CORPUS_DIR", REPO / "data" / "corpus")).resolve()
CORPUS_PROC = CORPUS_DIR / "processed"
TOPICS_DIR = Path(os.environ.get("MASLD_TOPICS_DIR", REPO / "data" / "topics")).resolve()
TOPICS_ANA = TOPICS_DIR / "analysis"
TOPICS_EMB = TOPICS_DIR / "embeddings"
ANA = PKG / "analysis"
FIG = PKG / "figures"
SRC = PKG / "source_data"
LOG = PKG / "logs"
for _d in (ANA, FIG, SRC, LOG):
    _d.mkdir(parents=True, exist_ok=True)

REF_RUN = "pubmedbert_neutral_mcs150_s2026"
ORIG_RUN = "pubmedbert_mcs150_s2026"
RESULTS = ANA / "results.json"

ADJACENT_TOPICS = {
    77: "veterinary (laying hens)",
    78: "veterinary (dairy cows)",
    24: "acute fatty liver of pregnancy",
    11: "pancreatic steatosis",
    18: "viral hepatitis with steatosis",
    20: "autoimmune and drug-induced liver injury",
    2: "lipid-droplet fluorescent probes (chemistry)",
}

GROUP_OF_TYPE = {
    "Randomized controlled trial": "interventional",
    "Non-randomized clinical trial": "interventional",
    "Systematic review/meta-analysis": "synthesis",
    "Guideline/consensus": "synthesis",
    "Cohort": "observational",
    "Case-control": "observational",
    "Cross-sectional": "observational",
    "Mendelian randomization": "observational",
    "Diagnostic/prognostic modelling": "observational",
    "Basic/experimental (animal or in vitro)": "basic",
    "Basic/experimental (in silico)": "basic",
    "Narrative review": "review_opinion",
    "Editorial/comment/letter": "review_opinion",
    "Case report/series": "case",
    "Trial/study protocol": "protocol",
    "Other primary study": "other",
    "Unclassified (title only)": "unclassified",
}

_VET_IDS = None


def veterinary_record_ids() -> set:
    global _VET_IDS
    if _VET_IDS is None:
        import pandas as pd
        d = pd.read_parquet(CORPUS_PROC / "literature_included_final.parquet", columns=["record_id", "pub_types"])
        _VET_IDS = set(d.record_id[d.pub_types.fillna("").astype(str).str.contains("Veterinary")])
    return _VET_IDS


def to_group(evidence_types, record_ids):
    import pandas as pd
    g = pd.Series(list(evidence_types), dtype=object).map(GROUP_OF_TYPE)
    vet = pd.Series(list(record_ids)).isin(veterinary_record_ids()).values
    g[vet & g.eq("interventional").values] = "basic"
    return g.values

INDIVIDUAL_DATASETS = [
    "data_nhanes", "data_uk_biobank", "data_korean_nhis", "data_knhanes", "data_jmdc", "data_finngen",
    "data_biobank_japan", "data_china_kadoorie", "data_mimic", "data_rotterdam_study",
    "data_aric", "data_mesa", "data_kangbuk_samsung", "data_swedish_registers_espresso_etc",
    "data_litmus", "data_all_of_us", "data_trinetx", "data_seer", "data_cprd",
    "data_taiwan_nhird",
]
EXCLUDED_DATASET_FLAGS = ["data_framingham", "data_nash_crn_niddk"]
DATASET_LABEL = {
    "data_nhanes": "NHANES", "data_uk_biobank": "UK Biobank", "data_korean_nhis": "NHIS/health screening",
    "data_knhanes": "KNHANES", "data_jmdc": "JMDC", "data_finngen": "FinnGen", "data_biobank_japan": "Biobank Japan",
    "data_china_kadoorie": "China Kadoorie", "data_mimic": "MIMIC", "data_framingham": "Framingham",
    "data_rotterdam_study": "Rotterdam Study", "data_aric": "ARIC", "data_mesa": "MESA",
    "data_kangbuk_samsung": "Kangbuk Samsung", "data_swedish_registers_espresso_etc": "Swedish registers",
    "data_nash_crn_niddk": "NASH CRN/NIDDK", "data_litmus": "LITMUS", "data_all_of_us": "All of Us",
    "data_korean_nhis_note": "NHIS pattern also matches national health screening and the US National Health Interview Survey",
    "data_trinetx": "TriNetX", "data_seer": "SEER", "data_cprd": "CPRD", "data_taiwan_nhird": "Taiwan NHIRD",
    "data_gbd": "GBD",
}

CAT = ["#3C5488", "#DC4E38", "#00A087", "#C99A2E", "#4DBBD5", "#8E6BB8"]
BLUE_RAMP = {100: "#edf0f6", 150: "#dce1ee", 200: "#cbd3e5", 250: "#b9c3dc", 300: "#a7b5d4", 350: "#96a6cc",
             400: "#8193bd", 450: "#6a7eab", 500: "#53699a", 550: "#3c5487", 600: "#314675", 650: "#273962",
             700: "#1d2b4f"}
DIV_BLUE = "#3C5488"
DIV_RED = "#DC4E38"
DIV_MID = "#F4F4F4"
INK = "#1A1A1A"
INK2 = INK
MUTED = "#969696"
GRID = "#E3E3E3"
AXIS = "#1A1A1A"
SURFACE = "#ffffff"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def update_results(section: str, payload: dict):
    data = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    data[section] = _clean(payload)
    data.setdefault("_meta", {})[section] = now()
    RESULTS.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def load_results() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def log(step: str, **kw):
    rec = {"utc": now(), "step": step, **_clean(kw)}
    with open(LOG / "pipeline_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{rec['utc']}] {step}: " + ", ".join(f"{k}={v}" for k, v in kw.items())[:400], flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)
