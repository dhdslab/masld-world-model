# Where each result of the paper comes from

Analysis results are recorded in `analysis/results.json` and the CSV files used to prepare the figures and tables.
Paths are relative to `MASLD_OUTPUT_DIR` (the repository root by default) except for source files under `code/`,
`upstream/` and `protocol/`.

## Main figures

| Figure | Drawn by | Results from | `results.json` section |
|---|---|---|---|
| Fig. 1 World model: temporal design and architecture | `code/fig_architecture.py` (draw.io) | `code/w3_world_model.py` | `world_model` |
| Fig. 2 Research topics and domains of the companion corpus | `code/fig_topics_ed.py` | `upstream/topics/ku02_universe.py`, `ku02b_domains_and_labels.py`, `ku03_dynamics.py`; `code/a1_corpus_nomenclature.py`, `code/a2_structure_maturity.py` | `corpus`, `structure` |
| Fig. 3 Semantic atlas by research domain | `code/fig_atlas.py` | `code/w10_semantic_atlas.py` | `semantic_atlas` |
| Fig. 4 Forecasting new associations | `code/fig4_forecast.py` | `code/w3_world_model.py` | `world_model` |
| Fig. 5 Which associations are anticipated | `code/fig4_forecast.py` | `code/w3_world_model.py` | `world_model` |
| Fig. 6 Comparison experiment (foundation-model extension) | `code/fig_fwm.py` | `code/w6_foundation_world_model.py` | `foundation_world_model` |

## Extended Data figures

| Figure | Drawn by | Results from |
|---|---|---|
| Extended Data Fig. 1 Study design | `code/fig1_design.py` (draw.io) | `code/w3_world_model.py` |
| Extended Data Fig. 2 Corpus, entity graphs, tasks, extension encoder | `code/fig_fwm_extended.py` | `code/w3_world_model.py`, `code/w6_foundation_world_model.py` |
| Extended Data Fig. 3 Disease renaming and transformer embeddings | `code/fig2_nomenclature.py` | `code/a1_corpus_nomenclature.py` (embeddings and topic models of `upstream/topics`) |
| Extended Data Fig. 4 Robustness and diagnostics | `code/fig_wm_extended.py` | `code/w3_world_model.py` |
| Extended Data Fig. 5 Architecture of the foundation-model extension | `code/fig_architecture.py` (draw.io) | `code/w6_foundation_world_model.py` |
| Extended Data Fig. 6 Trial landscape | `code/fig_trials_ed.py` | `code/a10_trial_landscape.py` |

## Supplementary Tables

| Table | Content | Produced by |
|---|---|---|
| 1a–c | Forecasting tasks; world model and comparison experiment for every task | `code/w3_world_model.py` (1a, 1b), `code/w6_foundation_world_model.py` (1c) |
| 2 | Analysis runs for the world model and its extension | `code/w4_run_history.py` (requires the archived `logs/pipeline_log.jsonl` for the reported runs) |
| 3 | REFORMS checklist | reporting checklist (not computed) |
| 4 | Neutralization of disease names | `NEUTRAL` in `code/w2_worldmodel_embed.py` (rules copied from `upstream/topics/ku01_embed.py`) |
| 5 | Evidence-type rules | `upstream/corpus/06_merge_screen_classify.py` (evidence type); `GROUP_OF_TYPE` and `to_group` in `code/common.py` |
| 6 | Settings of the foundation-model extension | constants of `code/w6_foundation_world_model.py` |
| 7 | Trial landscape (stage, endpoint, publication) | `code/a10_trial_landscape.py` |

## Supplementary Notes

| Note | Content | Source |
|---|---|---|
| 1 | Analysis plan with amendments (forecast, checksum and review-note passages omitted, marked […]) | `protocol/` |
| 2 | Additional results of the world model | `code/w3_world_model.py` (secondary analyses, amendment A8), `code/w6_foundation_world_model.py` |
| 3 | PubMed query of the world-model corpus | `code/w1_worldmodel_corpus.py` |
| 4 | Classification of ClinicalTrials.gov registrations | `code/a10_trial_landscape.py` |
| 5 | Audit of earlier automated analyses | historical analyses documented in Supplementary Note 5 |

## Source Data

The analysis scripts export plotted values to `source_data/`. `code/build_source_data.py` writes one
workbook per main figure and `SourceData_ExtendedDataFigures.xlsx` for Extended Data Figs. 1–6 to
`source_data_xlsx/`.
