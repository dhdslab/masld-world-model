"""Run the analysis pipeline of the paper, stage by stage.

  python run_pipeline.py corpus   stage 1 - literature, trial, grant, news and drug-label records (upstream/corpus)
  python run_pipeline.py topics   stage 2 - embeddings and topic model of the companion corpus (upstream/topics)
  python run_pipeline.py study    stage 3 - world model, trial landscape, figures and source data (code/)
  python run_pipeline.py all      stages 1-3 in order

Options
  --list                          print the steps of the chosen stage(s) and exit
  --from SCRIPT                   resume at the first step that runs SCRIPT (earlier outputs must exist)
  --encoder-python PATH           interpreter for encoder steps (w0, w2)
  --upstream-python PATH          interpreter for corpus and topic steps
                                  both default to the interpreter running this script
  --refresh-world-model-corpus    re-download the PubMed/PubTator3 corpus of the world model (w1) even if it exists
  --refresh-registries            re-query ClinicalTrials.gov and PubMed for the trial landscape (a10)

Folders are set with the environment variables MASLD_CORPUS_DIR, MASLD_TOPICS_DIR and MASLD_OUTPUT_DIR (see README).
Every step runs in its own process, with the script's folder as working directory; the run stops at the first failure.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUTPUT = Path(os.environ.get("MASLD_OUTPUT_DIR", REPO)).resolve()
WM = OUTPUT / "analysis" / "world_model"

FIGURES = ["fig_architecture.py", "fig1_design.py", "fig_atlas.py", "fig4_forecast.py", "fig_fwm.py", "fig_fwm_extended.py",
           "fig2_nomenclature.py", "fig_topics_ed.py", "fig_wm_extended.py", "fig_trials_ed.py"]


def steps(a):
    c, t, s = "upstream/corpus", "upstream/topics", "code"
    corpus = [
        ("corpus", c, "01_pubmed.py", [], False, None),
        ("corpus", c, "02_europepmc.py", [], False, None),
        ("corpus", c, "03_openalex.py", [], False, None),
        ("corpus", c, "04_registries.py", [], False, None),
        ("corpus", c, "05_news.py", ["--gdelt-download-only"], False, None),
        ("corpus", c, "05_news.py", ["--gdelt-cache-only"], False, None),
        ("corpus", c, "06_merge_screen_classify.py", [], False, None),
        ("corpus", c, "07_fulltext_oa.py", [], False, None),
        ("corpus", c, "08_quantify.py", [], False, None),
    ]
    topics = [
        ("topics", t, "ku01_embed.py", ["primary"], False, None),
        ("topics", t, "ku01_embed.py", ["primary", "--neutral"], False, None),
        ("topics", t, "ku02_universe.py", ["pubmedbert_neutral", "150", "2026", "1.0"], False, None),
        ("topics", t, "ku02_universe.py", ["pubmedbert", "150", "2026", "1.0"], False, None),
        ("topics", t, "ku02b_domains_and_labels.py", [], False, None),
        ("topics", t, "ku06_classifier.py", [], False, None),
        ("topics", t, "ku03_dynamics.py", [], False, None),
    ]
    need_corpus = a.refresh_world_model_corpus or not (WM / "docs.parquet").exists()
    study = [
        ("study", s, "a1_corpus_nomenclature.py", [], False, None),
        ("study", s, "a2_structure_maturity.py", [], False, None),
        ("study", s, "a10_trial_landscape.py", ["--refresh"] if a.refresh_registries else [], False, None),
        ("study", s, "w1_worldmodel_corpus.py", [], False, (need_corpus, "corpus already downloaded")),
        ("study", s, "w0_download_encoders.py", [], True,
         (not (WM / "models" / "biomedbert_abstract" / "pytorch_model.bin").exists(), "encoders already downloaded")),
    ] + [("study", s, "w2_worldmodel_embed.py", ["--encoder", e], True,
          (not (WM / f"emb_{e}.npy").exists(), "embeddings already computed")) for e in ("biomedbert", "neuml", "random")] + [
        ("study", s, "w3_world_model.py", ["--encoder", "biomedbert"], False, None),
        ("study", s, "w3_world_model.py", ["--encoder", "biomedbert", "--species-specific", "--primary-only"], False, None),
        ("study", s, "w6_foundation_world_model.py", [], False, None),
        ("study", s, "w10_semantic_atlas.py", [], False, None),
        ("study", s, "w4_run_history.py", [], False, None),
    ] + [("study", s, f, [], False, None) for f in FIGURES] + [
        ("study", s, "build_source_data.py", [], False, None),
    ]
    return {"corpus": corpus, "topics": topics, "study": study}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["corpus", "topics", "study", "all"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--from", dest="start", default="")
    ap.add_argument("--encoder-python", default=sys.executable)
    ap.add_argument("--upstream-python", default=sys.executable)
    ap.add_argument("--refresh-world-model-corpus", action="store_true")
    ap.add_argument("--refresh-registries", action="store_true")
    a = ap.parse_args()
    S = steps(a)
    todo = [x for st in (["corpus", "topics", "study"] if a.stage == "all" else [a.stage]) for x in S[st]]
    if a.start:
        names = [x[2] for x in todo]
        if a.start not in names:
            sys.exit(f"--from {a.start}: no such step; steps are {', '.join(dict.fromkeys(names))}")
        todo = todo[names.index(a.start):]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore")
    if any(x[2] == "03_openalex.py" for x in todo) and not a.list and "OPENALEX_API_KEY" not in env:
        sys.exit("03_openalex.py needs an OpenAlex API key in the environment variable OPENALEX_API_KEY")
    t0 = time.time()
    for stage, folder, script, args, enc, cond in todo:
        label = f"[{stage}] {folder}/{script} {' '.join(args)}".rstrip()
        if cond is not None and not cond[0]:
            print(f"-- skip {label} ({cond[1]})", flush=True)
            continue
        if a.list:
            print(label + ("   (encoder environment)" if enc else ""))
            continue
        print(f"== {label}", flush=True)
        t = time.time()
        python = {
            "corpus": a.upstream_python,
            "topics": a.upstream_python,
            "study": a.encoder_python if enc else sys.executable,
        }[stage]
        r = subprocess.run([python, script, *args], cwd=REPO / folder, env=env)
        if r.returncode != 0:
            sys.exit(f"Failed: {label} (exit code {r.returncode})")
        print(f"   done in {(time.time() - t) / 60:.1f} min", flush=True)
    if not a.list:
        print(f"All steps completed in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
