# A world model of MASLD research

Analysis code for *Mapping the Future of Metabolic Dysfunction-associated Steatotic Liver Disease*.

The world model ranks gene, chemical, disease and variant pairs by the likelihood of a future co-mention in the
MASLD literature. Candidate pairs had never appeared together in a sentence by the cutoff. The outcome was a
sentence co-mention in at least two distinct papers during the evaluation window.

The primary evaluation used the state at the end of 2022 to rank pairs reported in 2023–2025. The model uses
frozen PubMedBERT embeddings, publication counts and the co-mention graph in a discrete-time hazard model.
A separate comparison evaluated an extension adapted from three foundation-model designs.

## Repository layout

| Path | Contents |
|---|---|
| `upstream/corpus/` | Stage 1. Retrieves and screens the companion corpus (PubMed, Europe PMC, OpenAlex, 2020 to June 2026) and the trial, grant, news and drug-label records (`01`–`08`). |
| `upstream/topics/` | Stage 2. Sentence embeddings of the companion corpus, topic models (PCA, UMAP, HDBSCAN), research domains, a study-design classifier and topic dynamics (`ku01`–`ku06`). |
| `code/` | Stage 3. Forecasting corpus and embeddings (`w0`–`w2`), primary evaluation (`w3`), foundation-model comparison (`w6`), semantic atlas (`w10`), trial analysis (`a10`), figure scripts and Source Data. See [the results map](docs/RESULTS_MAP.md) for individual outputs. |
| `protocol/` | Version 1.0 of the analysis plan and the dated amendments A1–A12, described in Supplementary Note 1. |
| `demo/` | A synthetic-data demo of the model and its evaluation. |
| `docs/RESULTS_MAP.md` | Scripts and outputs for each figure, table and note. |
| `run_pipeline.py` | Runs the three stages in order. |

## System requirements

**Operating system.** The reported runs used Windows 11 Enterprise (build 26300). Linux and macOS are untested.

**Hardware used for the paper.**

- Study pipeline: AMD Ryzen 5 7600 (6 cores, 12 threads) with 128 GB RAM, and an NVIDIA GeForce RTX 4060 Ti for the
  encoder step `w2`. The world model (`w3`) and the extension (`w6`) run on the CPU. Smaller memory has not been tested.
- Upstream pipelines: a Windows workstation, CPU only, with 32 cores and 128 GB RAM.

The reported runs used three Python environments.

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

## Installation

```bash
git clone https://github.com/dhdslab/masld-world-model.git
cd masld-world-model

# analysis environment (enough for the demo and for stage 3 except w0/w2)
conda create -n masld python=3.12.7 -y && conda activate masld
pip install -r requirements.txt

# encoder environment (for w0 and w2)
conda create -n masld-encoder python=3.9.21 -y && conda activate masld-encoder
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-encoder.txt

# upstream environment (stages 1 and 2)
conda create -n masld-upstream python=3.11.18 -y && conda activate masld-upstream
pip install -r upstream/requirements.txt
```

The scripts download model weights from Hugging Face on first use. The encoder scripts (`code/w0_download_encoders.py` and
`upstream/topics/ku01_embed.py`) specify model revisions. The foundation-model extension (`w6`) downloads the
available revision of TimesFM 2.5 (`google/timesfm-2.5-200m-pytorch`).

## Demo

The demo generates a synthetic literature with latent research topics, unequal publication counts and a known
hazard of new associations. It trains at one cutoff and evaluates at a later cutoff. It uses the mathematical and
neural models (`Hazard`), evaluation metrics (`evaluate`) and node-level bootstrap (`node_bootstrap`) from
`code/w3_world_model.py`. The comparators are degree product and resource allocation.

```bash
conda activate masld
python demo/demo_synthetic.py
```

Example output from the synthetic demo

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

Demo results can vary slightly across platforms.

## Instructions for use

### Folders

Use these environment variables to set the output paths.

| Variable | Default | Contents |
|---|---|---|
| `MASLD_CORPUS_DIR` | `data/corpus` | Stage 1 output: `raw/`, `processed/`, `fulltext/`, `logs/` |
| `MASLD_TOPICS_DIR` | `data/topics` | Stage 2 output: `embeddings/`, `analysis/`, `models/` (Hugging Face cache) |
| `MASLD_OUTPUT_DIR` | repository root | Stage 3 output: `analysis/` (`results.json`, `world_model/`), `figures/`, `source_data/`, `source_data_xlsx/`, `logs/` |

All of these folders are excluded from version control.

### Running the pipeline

```bash
conda activate masld-upstream
python run_pipeline.py corpus
python run_pipeline.py topics

conda activate masld
python run_pipeline.py study --encoder-python /path/to/masld-encoder/python
```

Use `study --list` to show the steps. Resume with `study --from w6_foundation_world_model.py` and
the same `--encoder-python` option. Stages stop at the first failed command. To run both upstream
stages and the study in one command, pass `--upstream-python` and `--encoder-python` to `all`.

The runner reuses downloaded corpora, model weights and embeddings at their expected paths. To
rebuild the corpus and derived arrays together, start with a fresh `MASLD_OUTPUT_DIR` and retain the
original output folder if it contains a reported run. The trial script reuses its ClinicalTrials.gov
and PubMed caches where available. Use `--refresh-registries` to refresh both.

Run times of the longest steps of stage 3 on the computer described above:

| Step | Run time |
|---|---|
| `w3_world_model.py`, primary run | about 2.2 hours |
| `w6_foundation_world_model.py` | about 1.5 hours on 12 threads |
| `w2_worldmodel_embed.py` | about 3 minutes per encoder on the GPU |

Retrieval times depend on the public services and their rate limits.

### Outputs

- `analysis/results.json` contains the numerical analysis results.
- `figures/` contains `Figure1`–`Figure6` and `ExtendedDataFig1`–`ExtendedDataFig6` as PDF, SVG and 600 dpi PNG.
- `source_data/` holds plotted values. The workbook for Extended Data Figs. 1–6 is `source_data_xlsx/SourceData_ExtendedDataFigures.xlsx`.

[`docs/RESULTS_MAP.md`](docs/RESULTS_MAP.md) links the paper's figures, tables and notes to their scripts and outputs.

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

### Data snapshots

The reported world-model corpus was retrieved on 17 November 2026 and the trial records on 12 November 2026.
The PubMed search used publication-year windows and subdivided windows that met the search limit.
The repository contains the retrieval and analysis scripts. Reproducing the reported numbers also requires
the original source snapshots and cached inputs because the public databases change over time.

### Clinical trial publication and physician review

The publication analysis includes MASLD-targeted trials recorded as completed or terminated. Their actual
primary completion date had to be on or after 1 January 2010 and at least 365 days before registry retrieval.
Follow-up ended at retrieval. PubMed links identify results papers.

A physician independently rated the disease stage of 200 sampled registrations. To recompute the reported
agreement, use the original registry snapshot, the original sample IDs in
`validation/registry_stage_verification_ids.txt` and the original physician ratings in a CSV with
`nct_id` and `verified_stage` columns. Run:

```bash
python code/a10_trial_landscape.py --physician-review /path/to/physician_ratings.csv
```

The script writes Cohen’s κ for exact stage and for advanced versus other stages to
`trial_landscape.physician_stage_agreement` in `analysis/results.json`. Supply one rating from the seven
study categories for each sampled NCT ID. The original review files are not distributed here.

### Forecast evaluation

Graph states and training labels are restricted by cutoff. The reported pipeline merges entities with the
same name, identifies corpus-defining diseases and screens for abrupt changes in annotation rates using
the full corpus before it constructs those states. Supplementary Note 2 reports cutoff-specific
comparisons for the same-name and corpus-defining disease rules.

## Analysis plan and run records

[Version 1.0 and amendments A1–A12](protocol/World_Model_Analysis_Plan.md) record the original plan and
subsequent decisions (Supplementary Note 1).
Supplementary Table 2 uses the historical `logs/pipeline_log.jsonl` file. The script
`code/w4_run_history.py` can compile logs from a new run, but the historical log is required for the
published run record. Supplementary Note 5 documents the audit of earlier analyses.

### Seeds and numerical precision

- `code/common.py` uses seed `20261108`.
- `code/w3_world_model.py` uses seed `20261117` and derives separate generators for the later analyses.
- `upstream/topics/ku_common.py` uses seed `2026`.
- Embeddings are stored as float16. Platform differences can affect the last digit of reported metrics.

## Citation

If you use this code, please cite the paper and repository. Repository citation details are in
[`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE).

## Questions

Open a GitHub issue for questions about the code.
