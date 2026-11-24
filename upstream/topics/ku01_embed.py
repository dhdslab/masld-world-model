import os, sys, re, time
import numpy as np, pandas as pd
from ku_common import *

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
seed_all()


def build_items():
    L = pd.read_parquet(os.path.join(DATA, "literature_included_final.parquet"))
    CT = pd.read_parquet(os.path.join(DATA, "clinicaltrials_included.parquet"))
    G = pd.read_parquet(os.path.join(DATA, "nih_reporter_included.parquet"))
    N = pd.read_parquet(os.path.join(DATA, "news_included.parquet"))
    clean = lambda s: re.sub(r"<[^>]+>|\s+", " ", str(s or "")).strip()
    parts = [
        pd.DataFrame({"stream": "literature", "item_id": L.record_id, "year": L.year.astype(int),
                      "text": [clean(t) + ". " + clean(a) if isinstance(a, str) else clean(t) for t, a in zip(L.title, L.abstract)]}),
        pd.DataFrame({"stream": "trial", "item_id": CT.nct_id, "year": CT.year.astype(int),
                      "text": [clean(t) + ". " + clean(s) for t, s in zip(CT.title, CT.brief_summary)]}),
        pd.DataFrame({"stream": "nih_grant", "item_id": G.appl_id.astype(str), "year": G.fiscal_year.astype(int),
                      "text": [clean(t) + ". " + clean(a) for t, a in zip(G.title, G.abstract)]}),
        pd.DataFrame({"stream": "news", "item_id": [f"N{i:06d}" for i in range(len(N))], "year": N.year.astype(int),
                      "text": [re.sub(r"\s+-\s+[^-]{2,60}$", "", clean(t)) for t in N.title]}),
    ]
    I = pd.concat(parts, ignore_index=True)
    I["n_chars"] = I.text.str.len()
    I.to_parquet(os.path.join(EMB, "items.parquet"), index=False)
    log_event("embed.items", **I.stream.value_counts().to_dict(), total=len(I))
    return I


NEUTRAL = [
    (r"(?i)\bnon[\s\-]?alcoholic[\s\-]+fatty[\s\-]+liver[\s\-]+diseases?\b", "steatotic liver disease"),
    (r"(?i)\bmetabolic[\s\-]+(?:dysfunction[\s\-]+)?(?:associated|related)[\s\-]+(?:fatty|steatotic)[\s\-]+liver[\s\-]+diseases?\b", "steatotic liver disease"),
    (r"(?i)\bnon[\s\-]?alcoholic[\s\-]+steatohepatitis\b", "steatohepatitis"),
    (r"(?i)\bmetabolic[\s\-]+dysfunction[\s\-]+(?:associated|related)[\s\-]+steatohepatitis\b", "steatohepatitis"),
    (r"(?i)\bnon[\s\-]?alcoholic[\s\-]+fatty[\s\-]+liver\b", "steatotic liver"),
    (r"\bMetALDs?\b", "SLD with alcohol"),
    (r"\b(?:NAFLD|MAFLD|MASLD|NAFL)s?\b", "SLD"),
    (r"\b(?:NASH|MASH)\b", "SH"),
]


def neutralise(t):
    for p, r in NEUTRAL:
        t = re.sub(p, r, t)
    return t


def encode(model_id, revision, texts, tag):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(os.cpu_count())
    m = SentenceTransformer(model_id, revision=revision, cache_folder=HF_CACHE, device="cpu")
    m.max_seq_length = MAX_SEQ
    out = os.path.join(EMB, f"{tag}.npy")
    t = time.time()
    chunks, step = [], 4096
    for i in range(0, len(texts), step):
        part = os.path.join(EMB, f"{tag}_part{i // step:03d}.npy")
        if not os.path.exists(part):
            e = m.encode(texts[i:i + step], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            np.save(part, e.astype(np.float16))
        chunks.append(part)
        print(tag, min(i + step, len(texts)), "/", len(texts), f"{time.time() - t:.0f}s", flush=True)
    E = np.concatenate([np.load(p) for p in chunks])
    np.save(out, E)
    for p in chunks: os.remove(p)
    log_event("embed.done", model=model_id, revision=revision, n=len(E), dim=E.shape[1], seconds=round(time.time() - t))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    neutral = "--neutral" in sys.argv
    I = build_items() if not os.path.exists(os.path.join(EMB, "items.parquet")) else pd.read_parquet(os.path.join(EMB, "items.parquet"))
    texts = I.text.tolist()
    suf = ""
    if neutral:
        texts = [neutralise(t) for t in texts]; suf = "_neutral"
        pd.DataFrame({"text_neutral": texts}).to_parquet(os.path.join(EMB, "items_text_neutral.parquet"), index=False)
        log_event("embed.neutralised", changed=int(sum(a != b for a, b in zip(texts, I.text))))
    if which in ("primary", "both"):
        encode(*PRIMARY, texts, "pubmedbert" + suf)
    if which in ("secondary", "both"):
        encode(*SECONDARY, texts, "bge_small" + suf)
