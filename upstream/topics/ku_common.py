import os, json, datetime, random
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.abspath(os.environ.get("MASLD_TOPICS_DIR", os.path.join(_REPO, "data", "topics")))
DATA = os.path.join(os.path.abspath(os.environ.get("MASLD_CORPUS_DIR", os.path.join(_REPO, "data", "corpus"))), "processed")
EMB = os.path.join(RES, "embeddings"); ANA = os.path.join(RES, "analysis")
FIG = os.path.join(RES, "figures"); TAB = os.path.join(RES, "tables"); LOG = os.path.join(RES, "logs")
MODELS = os.path.join(RES, "models"); HF_CACHE = os.path.join(MODELS, "hf_cache")
for d in (EMB, ANA, FIG, TAB, LOG):
    os.makedirs(d, exist_ok=True)

SEED = 2026
PRIMARY = ("NeuML/pubmedbert-base-embeddings", "b79526d6ef3645e0df4530322e266f24c829f5ef")
SECONDARY = ("BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a")
MAX_SEQ = 256


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s)
    try:
        import torch; torch.manual_seed(s)
    except ImportError:
        pass


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def log_event(step, **kw):
    rec = {"utc": now(), "step": step, **kw}
    with open(os.path.join(LOG, "ku_pipeline_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(f"[{rec['utc']}] {step}: " + ", ".join(f"{k}={v}" for k, v in kw.items())[:600], flush=True)

REF_RUN = "pubmedbert_neutral_mcs150_s2026"
