# A world model of MASLD research

Analysis code for *Mapping the Future of Metabolic Dysfunction-associated Steatotic Liver Disease*.

The model forecasts which pairs of biomedical entities will first be reported together in the MASLD literature. At each
publication cutoff, a frozen PubMedBERT encoder represents genes, chemicals, diseases and variants. The model combines
these representations and their changes over time with publication counts and the co-mention graph. A discrete-time
hazard model estimates how likely a pair is to be reported together after the cutoff.

Eligible pairs had never appeared together in a sentence through the cutoff. A pair counts as a new association if it
appears together in a sentence in at least two distinct papers during the evaluation window. We used temporal hold-outs
and compared the world model with an extension adapted from three foundation-model designs. See [Data](#data) for the
inputs needed to reproduce the analyses.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## Repository layout

| Path | Contents |
|---|---|
| `upstream/corpus/` | Stage 1. Retrieves and screens the companion corpus (PubMed, Europe PMC, OpenAlex, 2020 to June 2026) and the trial, grant, news and drug-label records (`01`–`08`). |
| `upstream/topics/` | Stage 2. Sentence embeddings of the companion corpus, topic models (PCA, UMAP, HDBSCAN), research domains, a study-design classifier and topic dynamics (`ku01`–`ku06`). |
| `code/` | Stage 3, the study. It builds the world-model corpus (PubMed and PubTator3, 2000 to June 2026), encoders and embeddings (`w0`–`w2`) and the world model and its evaluation (`w3`). It also covers the foundation-model extension (`w6`), the semantic atlas (`w10`) and the run history (`w4`), as well as the corpus and renaming analyses (`a1`, `a2`), the trial landscape (`a10`), the figures (`fig*`) and Source Data (`build_source_data.py`). |
| `protocol/` | Version 1.0 of the analysis plan and the dated amendments A1–A12, described in Supplementary Note 1. |
| `demo/` | A synthetic-data demo of the model and its evaluation. |
| `docs/RESULTS_MAP.md` | Where each figure, table and note of the paper comes from. |
| `run_pipeline.py` | Runs the three stages in order. |

---

## System requirements

**Operating system.** The reported runs used Windows 11 Enterprise (build 26300). Linux and macOS are untested.

**Hardware used for the paper.**

- Study pipeline: AMD Ryzen 5 7600 (6 cores, 12 threads) with 128 GB RAM, and an NVIDIA GeForce RTX 4060 Ti for the
  encoder step `w2`. The world model (`w3`) and the extension (`w6`) run on the CPU. Smaller memory has not been tested.
- Upstream pipelines: a Windows workstation, CPU only, with 32 cores and 128 GB RAM.

**Software.** Three Python environments were used. Their versions are pinned in the files below.

| Environment | Used for | Python | Key packages | File |
|---|---|---|---|---|
| analysis | `code/` (except `w0`, `w2`) and `demo/` | 3.12.7 | NumPy 2.0.2, pandas 2.3.3, SciPy 1.16.2, scikit-learn 1.7.2, statsmodels 0.14.6, PyTorch 2.8.0 (CPU), TimesFM 3.0.2, Matplotlib 3.10.7, PyMuPDF 1.25.0 | `requirements.txt` |
| encoder | `code/w0_download_encoders.py`, `code/w2_worldmodel_embed.py` | 3.9.21 | PyTorch 2.8.0 (CUDA 12.6), transformers 4.46.3 | `requirements-encoder.txt` |
| upstream | `upstream/` | 3.11.18 | sentence-transformers 6.2.0, umap-learn 0.5.12, hdbscan 0.8.44, scikit-learn 1.9.1 | `upstream/requirements.txt`; full list in `upstream/requirements-lock.txt` |

**Non-standard requirements.**

- An NVIDIA GPU is optional; `w2` falls back to the CPU.
- Fig. 1 and Extended Data Figs. 1 and 5 require [draw.io Desktop](https://www.drawio.com/). The scripts look for
  its command-line interface on `PATH` and in the default Windows installation directories.
- Figures use the Arial font.
- Retrieval steps need internet access. `upstream/corpus/03_openalex.py` needs an OpenAlex API key in the environment
  variable `OPENALEX_API_KEY`.

---

## Installation

```bash
git clone https://github.com/dhdslab/masld-world-model.git
cd masld-world-model

# analysis environment (enough for the demo and for stage 3 except w0/w2)
conda create -n masld python=3.12.7 -y && conda activate masld
pip install -r requirements.txt

# encoder environment (for w0 and w2)
conda create -n masld-encoder python=3.9 -y && conda activate masld-encoder
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-encoder.txt

# upstream environment (stages 1 and 2)
conda create -n masld-upstream python=3.11 -y && conda activate masld-upstream
pip install -r upstream/requirements.txt
```

Each environment typically installs in a few minutes, most of which are spent downloading PyTorch. The scripts
download model weights from Hugging Face on first use. The encoder scripts (`code/w0_download_encoders.py` and
`upstream/topics/ku01_embed.py`) specify model revisions. The foundation-model extension (`w6`) downloads the
available revision of TimesFM 2.5 (`google/timesfm-2.5-200m-pytorch`).

---

## Demo

The demo generates a synthetic literature with latent research topics, unequal publication counts and a known
hazard of new associations. It trains at one cutoff and evaluates at a later cutoff. It uses the mathematical and
neural models (`Hazard`), evaluation metrics (`evaluate`) and node-level bootstrap (`node_bootstrap`) from
`code/w3_world_model.py`. The comparators are degree product and resource allocation.

```bash
conda activate masld
python demo/demo_synthetic.py
```

Expected output, which took about 15 s on the computer described above:

```text
Synthetic field: 600 entities, 7,357 reported associations at the test cutoff, 172,343 candidate pairs, 2,679 new associations within 3 years (training cutoff: 2,272).

                          AUROC  popularity-adjusted AUROC  average precision AUROC 95% CI
model                                                                                     
degree product            0.823                      0.590              0.094  0.806-0.840
resource allocation       0.845                      0.717              0.135  0.830-0.862
mathematical world model  0.869                      0.756              0.154  0.857-0.883
neural world model        0.868                      0.754              0.153  0.856-0.883

Paired differences in AUROC (node bootstrap, 100 draws, 95% interval):
  neural world model minus resource allocation: 0.017 to 0.028
  mathematical world model minus resource allocation: 0.017 to 0.028

Standardized coefficients of the mathematical world model (per standard deviation):
  semantic_cosine          +0.48
  resource_allocation      +0.02
  log_degree_product       +0.17
  log_common_neighbours    +0.75
  attention_low            +0.20
  attention_high           +0.38

Finished in 12 s; results written to demo/output/demo_results.json
```

The demo writes `demo/output/demo_results.json`. In the example run, both world models had higher AUROCs than
resource allocation, and both paired intervals excluded zero. Values can differ slightly across platforms.

---

## Instructions for use

### Folders

Outputs are written to three folders, which are set with environment variables:

| Variable | Default | Contents |
|---|---|---|
| `MASLD_CORPUS_DIR` | `data/corpus` | Stage 1 output: `raw/`, `processed/`, `fulltext/`, `logs/` |
| `MASLD_TOPICS_DIR` | `data/topics` | Stage 2 output: `embeddings/`, `analysis/`, `models/` (Hugging Face cache) |
| `MASLD_OUTPUT_DIR` | repository root | Stage 3 output: `analysis/` (`results.json`, `world_model/`), `figures/`, `source_data/`, `source_data_xlsx/`, `logs/` |

All of these folders are excluded from version control.

### Running the pipeline

```bash
python run_pipeline.py corpus                       # stage 1 (network-bound; GDELT news retrieval can take many hours)
python run_pipeline.py topics                       # stage 2
python run_pipeline.py study --encoder-python /path/to/masld-encoder/python
python run_pipeline.py study --list                 # show the steps without running them
python run_pipeline.py study --from w6_foundation_world_model.py   # resume
```

Each stage runs its scripts in order, each in its own process, and stops at the first failure.

- **Cached downloads.** The world-model corpus (`w1`), the encoders (`w0`) and the embeddings (`w2`) are skipped if
  they already exist. Use `--refresh-world-model-corpus` to download the corpus again. `a10` re-queries
  ClinicalTrials.gov and PubMed only with `--refresh-registries`.
- **Running one script.** Scripts can also be run one at a time from their folder, for example
  `cd code && python w3_world_model.py --encoder biomedbert`.

Run times of the longest steps of stage 3 on the computer described above:

| Step | Run time |
|---|---|
| `w3_world_model.py`, primary run | about 2.2 hours |
| `w6_foundation_world_model.py` | about 1.5 hours on 12 threads |
| `w2_worldmodel_embed.py` | about 3 minutes per encoder on the GPU |

Every other step takes minutes. `w1` and stage 1 are limited by the rate limits of the public services they query.

### Outputs

- `analysis/results.json` contains the numerical analysis results.
- `figures/` contains the figures in vector PDF, SVG and 600 dpi PNG.
- `source_data/` and `source_data_xlsx/` contain the plotted values.

[`docs/RESULTS_MAP.md`](docs/RESULTS_MAP.md) links the paper's figures, tables and notes to their scripts and outputs.

---

## Data

The pipeline retrieves public records and downloads model weights from the sources below.

| Source | Used for | Retrieved by |
|---|---|---|
| [PubMed / NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/) | world-model corpus; companion corpus; trial results papers | `code/w1`, `upstream/corpus/01`, `code/a10` |
| [PubTator3](https://www.ncbi.nlm.nih.gov/research/pubtator3/) | entity annotations of the world-model corpus | `code/w1` |
| [Europe PMC](https://europepmc.org/RestfulWebService), [OpenAlex](https://openalex.org) | companion corpus; open-access full text | `upstream/corpus/02`, `03`, `07` |
| [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api) | trial registrations | `upstream/corpus/04`, `code/a10` |
| [NIH RePORTER](https://reporter.nih.gov), [openFDA](https://open.fda.gov) | grant and drug-label records of the companion corpus | `upstream/corpus/04` |
| [GDELT](https://www.gdeltproject.org), Google News RSS | news records of the companion corpus | `upstream/corpus/05` |
| [Hugging Face](https://huggingface.co) | BiomedBERT, PubMedBERT sentence embeddings, TimesFM 2.5 weights | `code/w0`, `code/w6`, `upstream/topics/ku01` |

Titles and abstracts are subject to the copyright of their publishers and are not redistributed.

**Derived data.** The pipeline generates the world-model entity tables, states, candidate pairs, scores and run history.

**Physician review of trial registrations.** A physician independently classified the disease stage and primary endpoint
for a stratified sample of trial registrations. The agreement analysis in `code/a10_trial_landscape.py` computes
Cohen’s κ by comparing the physician’s assessments with the rule-based classifications.

**Exact reproduction.** PubMed, PubTator3 and the other sources change over time, so a new download may differ from the
study corpus. Exact reproduction requires the records used for the reported analyses. The world-model corpus was
retrieved on 17 November 2026 and the trial records on 12 November 2026.

---

## Analysis plan

Version 1.0 of the analysis plan specified the primary task, model and metric before any forecasting outcome was
computed. The 12 dated amendments in `protocol/` record subsequent changes and indicate which decisions were made after
reviewing results. Supplementary Note 1 describes the plan and its amendments.

The pipeline logs analysis runs, including development and superseded runs, to `logs/pipeline_log.jsonl`.
`code/w4_run_history.py` compiles the run history for Supplementary Table 2.

---

## Reproducibility notes

- **Random seeds** are fixed:
  - `SEED = 20261108` in `code/common.py`;
  - `WM_SEED = 20261117` in `code/w3_world_model.py` (separate generators are derived from it for later amendments);
  - `SEED = 2026` in `upstream/topics/ku_common.py`.
- **Embeddings** are stored as float16. Results computed on other GPUs or on the CPU may differ in the last digit.
- **Run record.** `results.json` records the outputs of every step, with the time each section was written.
- **Earlier exploratory analyses.** Supplementary Note 5 documents the exploratory work that preceded the reported
  analyses. This repository covers the analyses reported in the paper.

## Citation

If you use this code, please cite the paper and repository. Repository citation details are in
[`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE).

## Questions

Open a GitHub issue for questions about the code.
