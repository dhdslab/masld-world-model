from __future__ import annotations

import json
from pathlib import Path

from common import PKG, TOPICS_DIR

BIOMEDBERT = ("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract", "d673b8835373c6fa116d6d8006b33d48734e305d")
BIOMEDBERT_FILES = ["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.txt"]
NEUML = ("NeuML/pubmedbert-base-embeddings", "b79526d6ef3645e0df4530322e266f24c829f5ef")


def download_biomedbert(out: Path, files=BIOMEDBERT_FILES) -> Path:
    from huggingface_hub import hf_hub_download
    out.mkdir(parents=True, exist_ok=True)
    repo, rev = BIOMEDBERT
    for f in files:
        if not (out / f).exists():
            hf_hub_download(repo, f, revision=rev, local_dir=out)
    (out / "SOURCE.json").write_text(json.dumps({"repo": repo, "revision": rev, "files": list(files)}, indent=1))
    return out


def download_neuml(cache: Path) -> Path:
    from huggingface_hub import snapshot_download
    repo, rev = NEUML
    return Path(snapshot_download(repo, revision=rev, cache_dir=cache))


def main():
    p1 = download_biomedbert(PKG / "analysis" / "world_model" / "models" / "biomedbert_abstract")
    p2 = download_neuml(TOPICS_DIR / "models" / "hf_cache")
    print(json.dumps({"biomedbert": str(p1), "neuml": str(p2)}, indent=1))


if __name__ == "__main__":
    main()
