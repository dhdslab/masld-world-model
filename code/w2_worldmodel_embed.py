from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import PKG, TOPICS_DIR

OUT = PKG / "analysis" / "world_model"
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


def neutralise(t: str) -> str:
    for p, r in NEUTRAL:
        t = re.sub(p, r, t)
    return t


def model_path(encoder: str) -> Path:
    if encoder in ("biomedbert", "random"):
        return OUT / "models" / "biomedbert_abstract"
    snap = TOPICS_DIR / "models" / "hf_cache" / "models--NeuML--pubmedbert-base-embeddings" / "snapshots"
    return sorted(snap.iterdir())[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="biomedbert", choices=["biomedbert", "neuml", "random"])
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    import torch
    try:
        import transformers.utils.import_utils as iu
        iu._torchvision_available = False
    except Exception:
        pass
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = pd.read_parquet(OUT / "docs.parquet", columns=["pmid", "title", "abstract"])
    text = [neutralise((t or "").strip() + ". " + (b or "").strip()) for t, b in zip(D.title, D.abstract)]
    mp = model_path(a.encoder)
    tok = AutoTokenizer.from_pretrained(mp)
    if a.encoder == "random":
        torch.manual_seed(20261117)
        model = AutoModel.from_config(AutoConfig.from_pretrained(mp))
    else:
        model = AutoModel.from_pretrained(mp)
    model = model.to(dev).eval()
    lens = np.array([len(t) for t in text])
    order = np.argsort(lens, kind="stable")
    E = np.zeros((len(text), model.config.hidden_size), dtype=np.float16)
    t0 = time.time()
    with torch.inference_mode():
        for k in range(0, len(order), a.batch):
            ix = order[k:k + a.batch]
            b = tok([text[i] for i in ix], padding=True, truncation=True, max_length=a.max_length, return_tensors="pt").to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
                h = model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            E[ix] = ((h * m).sum(1) / m.sum(1)).float().cpu().numpy().astype(np.float16)
            if (k // a.batch) % 200 == 0:
                print(f"{k}/{len(order)} {k / max(time.time() - t0, 1e-9):.1f} docs/s", flush=True)
    np.save(OUT / f"emb_{a.encoder}.npy", E)
    meta = {"encoder": a.encoder, "path": str(mp), "device": dev, "n": len(text), "max_length": a.max_length,
            "seconds": round(time.time() - t0, 1), "pooling": "mean of final-layer token states over the attention mask",
            "text": "neutralized title + abstract", "torch": torch.__version__}
    (OUT / f"emb_{a.encoder}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
