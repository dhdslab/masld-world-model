# MASLD world model: analysis plan and dated amendments

Version 1.0 of the analysis plan was fixed on 17 November 2026 before any forecasting outcomes were computed. This note summarises the plan and records the subsequent amendments in chronological order.

## 1. Research question

Using only the MASLD literature available by a given cutoff, can the model identify pairs of biomedical entities that will first be reported together after that cutoff?

An association is a sentence-level co-mention of two entities, such as genes, chemicals or drugs, diseases or phenotypes, or variants, in a paper from the MASLD corpus. The outcome is subsequent reporting in the literature. It does not establish a biological association.

## 2. Corpus

- **PubMed records.** We included papers published from 2000 through June 2026 that used one of 17 historical, interim or current disease terms in the title or abstract, or were indexed under the MeSH heading Non-alcoholic Fatty Liver Disease. The query excluded six publication types in all years: Published Erratum, Retraction of Publication, Comment, Editorial, Letter and News.
- **Text and entities.** Titles, abstracts and normalised entity annotations came from PubTator3. We used the Gene, Chemical, Disease and Variant categories and omitted species and cell-line annotations. The same eligibility rules applied in every year, independently of the 2020–2026 main corpus.
- **Disease terminology.** Before embedding the text, we applied the same replacements as in the main analysis: NAFLD, NAFL, MAFLD and MASLD became “SLD”, and NASH and MASH became “steatohepatitis”. We applied these replacements to limit the time signal carried by changes in disease terminology.
- **Excluded entities.** We omitted MASLD-family disease entities used to define the corpus. These comprised identifiers for which at least half of the annotated mentions matched a query term and the two MeSH headings Non-alcoholic Fatty Liver Disease and Fatty Liver. Entities without a normalised identifier were also excluded.

## 3. Forecasting task

| Element | Definition |
|---|---|
| Cutoff T | End of a publication year. |
| Node set V_T | Entities mentioned in at least 3 papers published through T. |
| Candidate pairs | Unordered pairs of entities in V_T that had never appeared together in a sentence through T. |
| Label Y_ij(T, Δ) | 1 if the pair appears together in a sentence in at least 2 distinct papers published in (T, T + Δ]. |
| Test tasks | T = 2020, 2021, 2022 or 2023 and Δ = 1, 2 or 3 years, subject to T + Δ ≤ 2025. |
| Primary task | T = 2022 and Δ = 3, covering associations first reported in 2023–2025. |
| Near-prospective task | T = 2025 and Δ = 0.5. The model uses training data through 2025 and is evaluated on papers published from January through June 2026. |
| Training tasks | For a test task (T, Δ), the eligible training cutoffs T' satisfy T' + Δ ≤ T. This keeps all training labels within the test cutoff. |

The first test cutoff was 2020, so evaluation labels came from papers published in 2021 or later. The encoder was pretrained on PubMed abstracts collected in 2020 (Gu et al., arXiv:2007.15779).

## 4. World model

**Encoder E.** We used frozen PubMedBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract). We encoded the neutralised title and abstract, retaining the first 256 word pieces, and averaged the final-layer token states to obtain e_d = E(x_d).

**Whitening.** At each cutoff T, we estimated the mean μ_T and the leading 128 principal axes from papers published through T. We centred, projected and scaled each embedding as ẽ_d = Λ_T^{-1/2} U_T^T (e_d − μ_T).

**Entity state and dynamics.** The semantic state h_i(T) was the unit-normalised sum of the embeddings of papers that mentioned entity i through T, with an exponential weight for publication age:

- h_i(T) = Σ_d κ(T − t_d) 1[i ∈ d, t_d ≤ T] ẽ_d, normalised to unit length.
- κ(a) = exp(−a/τ), with τ = 3 years.
- Two-year semantic velocity was v_i(T) = h_i(T) − h_i(T − 2).
- The projected state was ĥ_i(T + Δ) = normalize(h_i(T) + (Δ/2) v_i(T)).

**Field drift.** The drift vector u_T was the difference between the mean ẽ_d over the latest two years and the mean over the preceding two years. Pair alignment with this drift was ((h_i + h_j)/2) · u_T.

**Research attention.** Let n_i(T) be the number of papers mentioning i through T. Recent growth was g_i(T) = log[(n_i(T) − n_i(T − 2) + 1) / (n_i(T − 2) − n_i(T − 4) + 1)].

**Graph features.** From the sentence co-mention graph A_T on V_T, we computed common neighbours, resource allocation, Adamic–Adar, Jaccard and degree product for each pair.

**Pair features x_ij(T).**

- Current semantic cosine h_i·h_j and projected cosine ĥ_i·ĥ_j.
- Semantic convergence, defined as the cosine at T minus the cosine at T − 2.
- Semantic triadic closure, defined as the maximum cosine between i and a graph neighbour of j, averaged over both directions.
- Alignment with field drift.
- The two entities’ log attention and growth, sorted to make the pair representation order invariant.
- The five graph features after log1p transformation, and indicators for entity-type pairs.

**Transition model (mathematical world model).** We used a discrete-time proportional-hazards model for association formation:

Pr(Y_ij(T, Δ) = 1 | S_T) = 1 − exp{−Δ exp(η_ij(T))}, with η_ij(T) = β_0 + βᵀ x_ij(T).

This is a complementary log–log model with offset log Δ. We fitted it by L2-penalised maximum likelihood (penalty 1/C, C = 1) using standardised features from the training tasks.

**Neural world model (primary model in version 1.0).** We added a learned interaction of the entity states to the linear predictor, η_ij = βᵀ x_ij + g_θ(z_i, z_j, x_ij). The vector z_i contained the first 32 whitened coordinates of h_i. The pair inputs were symmetric: z_i ⊙ z_j and |z_i − z_j|. The interaction g_θ was a two-layer perceptron fitted with the same likelihood (64 units, ReLU, dropout 0.1, weight decay 1e-4, Adam, 30 epochs and a batch size of 4,096). For each training task, we used all positives and 20 randomly sampled negatives per positive, with case–control weights.

## 5. Comparators in version 1.0

1. Degree product (preferential attachment).
2. Resource allocation, the best-performing topological score in the earlier analysis.
3. Static semantic similarity h_i·h_j, following the word-embedding approach of Tshitoyan et al.
4. A logistic model with resource allocation and static semantic similarity as predictors.
5. The mathematical world model, the hazard model fitted to x_ij.

## 6. Outcomes and statistical analysis

The primary endpoint was the neural world model’s AUROC on the primary task. We evaluated all positive pairs and a uniform random sample of up to 1,000,000 negative pairs. Uniform sampling of negatives yields an unbiased AUROC estimate. For average precision and precision among the top k pairs, we used inverse sampling weights.

The primary contrast was the difference in AUROC between the neural world model and resource allocation. We obtained 95% percentile intervals from 500 node-level bootstrap draws. Each draw assigned a Poisson multiplier to each entity and weighted each pair by the product of its two multipliers.

**Secondary analyses.** We reported AUROC for each cutoff and horizon, by entity-type pair, and for pairs without common neighbours (“non-obvious” associations). We also examined pairs involving advanced liver disease. This group was defined before evaluation using MeSH identifiers for cirrhosis and its complications, liver cancer, liver failure and liver transplantation (see A8 for the implemented definition). Additional analyses stratified AUROC by the evidence type of the earliest papers reporting a pair (trial, observational, basic or review), compared observed and predicted risk by decile, assessed time-to-report concordance and evaluated the most recent hold-out.

**Ablations.** We removed each feature block from x_ij in turn and refitted the model: semantics, dynamics, field drift, graph features and attention.

**Sensitivity analyses.**

- Require co-mentions in at least 1 or at least 3 papers.
- Restrict the corpus to primary research.
- Set τ = ∞ so that all publication years receive equal weight.
- Replace the encoder with NeuML/pubmedbert-base-embeddings to probe possible leakage from fine-tuning text published after the cutoffs.
- Replace the encoder with a randomly initialised BERT of the same architecture as a negative control.
- Shuffle the publication years within the training data as a negative control.

**Evaluation design.** To assess how the evaluation design affected AUROC, we applied the same scores to a random 20% edge hold-out with never-mentioned pairs as negatives.

**Random seed.** 20261117.

## 7. Interpretation

We described a forecast model as outperforming a comparator when the 95% interval for the AUROC difference lay above zero. Performance measures how well the model ranks pairs that are subsequently reported together. Causality and clinical validity were outside the scope of this evaluation. We reported results for every test task.

## Amendments

**A1 (17 November 2026, before any forecasting outcome was computed).** We replaced the maximum-cosine definition of semantic triadic closure with the cosine between h_i and the unit-normalised centroid of the states of j’s graph neighbours, averaged over both directions. Computing the maximum would have required a max-plus product over millions of pairs.

**A2 (17 November 2026, before any forecasting outcome was computed).** Training cutoffs started in 2010, giving every training state at least 10 years of publication history, and ended at the latest year allowed under section 3. Each test horizon used training tasks with the same Δ. The Δ = 0.5 hold-out used the Δ = 1 model, as the log-Δ offset does not affect the ranking. The PubTator3 gene identifiers were species specific. We indexed variants by dbSNP identifier when available and by HGVS and gene otherwise.

**A3 (17 November 2026, after a test on the partial corpus and before analysis of the full corpus).** Under inverse sampling, weights for positive examples were of the order of 10^-3. Neural model training was unstable, and performance was below chance at one horizon. We switched to balanced class weights for training on the case–control sample, then recalibrated the intercept by weighted maximum likelihood using the inverse sampling weights. Recalibration shifts the intercept without changing rankings or AUROC. The mathematical world model retained the original weighting because its objective is convex. A8 records the partial-corpus outcomes available at this point.

**A4 (17 November 2026, after the same partial-corpus test and before analysis of the full corpus).** The uniform sample contained many negatives involving rarely studied entities, allowing degree product alone to discriminate well. We added a secondary measure of popularity-adjusted AUROC for every task and model. We defined degree-product deciles from the negative sample, computed AUROC within each decile and averaged the estimates weighted by the number of positives. We excluded deciles with fewer than 10 positives. The original primary endpoint still applied.

**A5 (17 November 2026, before analysis of the full corpus).** The PubTator3 BioC-JSON export returned incomplete annotations or an HTTP 400 error for records with PMIDs above about 39.1 million. The BioC-JSON and plain-text exports agreed for older records, so we used the plain-text export for all records.

**A6 (17 November 2026, after the pre-specified outcomes had been computed).** Inspection of the model output identified frequent pairings of different identifiers for the same gene or concept. PubTator3 uses different identifiers for human and mouse orthologs, and entries such as IL-6 with IL-6 and MCP-1 with CCL2 could therefore be counted as new associations. We mapped genes to human orthologs where available using NCBI gene_orthologs (downloaded 17 November 2026). We also excluded pairs whose most frequent mentions were identical after lowercasing and removing non-alphanumeric characters. We retained the species-specific results from the original plan alongside the revised ones (keys ending in _prespecified in results.json).

**A7 (17 November 2026, after A6).** The negative sampler truncated a sorted array of pair keys, so it under-sampled pairs with high identifiers and included almost no gene–gene pairs. The sampler for never-mentioned negatives in the evaluation-design comparison had the same error. We corrected both samplers to shuffle the arrays before truncation and recomputed all reported results, including the species-specific analysis.

**A8 (18 November 2026, after all outcomes had been computed). Implementation corrections, deviations from the plan and additional analyses.**

- **Implementation.** The PubTator3 plain-text export used the variant labels “SNP”, “DNAMutation” and “ProteinMutation”. The parser did not recognise them, so the earlier analysis contained no variants. The corrected parser includes these labels and indexes variants by dbSNP identifier, by HGVS and gene if a dbSNP identifier is unavailable, and by the tmVar normalised form and gene if neither is available. The concordance calculation now uses all positive pairs, replacing the first 4,001 in key order. We also changed the combined logistic comparator (resource allocation and semantic similarity) to balanced class weights, as in A3. With inverse sampling weights, it had performed worse than one of its predictors alone. We recomputed every result.
- **Deviations present in the initial implementation.** The code evaluated an additional test task (T = 2024, Δ = 1). We report it separately and use the 11 tasks from section 3 for summaries across tasks. The advanced-liver-disease set contained no liver-transplantation identifier because PubTator3 does not annotate the procedure as a disease.
- **Timing of A3 and A4.** Both changes followed evaluation on 64,051 of the final 82,084 papers. In that subset, the original neural model achieved an AUROC of 0.36 for Δ = 1. At the primary horizon, its AUROC was 0.957 versus 0.967 for the hazard model. Intercept recalibration preserves the ranking. Changing the training weights can change it.
- **Additional secondary analyses.** These were added after the original plan and used a separate random-number generator (seed + 8) to preserve the earlier draws:
  - Supervised comparators based on attention alone, on attention plus graph topology (hazard and neural models), and on gradient-boosted trees with and without language-model features.
  - Randomly initialised and sentence-tuned encoder probes evaluated on the same candidate pairs as the primary analysis.
  - Node-bootstrap intervals for popularity-adjusted AUROC, the difference from resource allocation on each task, subgroups, ablations, the most recent hold-out and the evaluation-design comparison.
  - Evaluation of all candidate pairs, including precision and recall at 100, 1,000 and 10,000, average precision, ROC curves and precision–recall curves.
  - Per-entity AUROC, hits among the 10 highest-ranked partners and mean reciprocal rank.
  - AUROC stratified by the degree twentiles of both entities.
  - An audit of entity normalisation, with a sensitivity analysis excluding ambiguous entities or pairs. We excluded a node if its most frequent mention appeared in at least three times as many papers as were annotated with that node or if another node shared the same mention. We excluded a pair if either name contained the other. All rules used text published through the cutoff.
  - Cutoff-specific versions of the shared-name exclusion (A6) and the MASLD-family exclusion.
  - Tests of the model’s ability to predict its own state. For the semantic state, we compared the projection in equation (10) with persistence. For research attention, we compared a Poisson transition estimated at earlier cutoffs with persistence of recent activity.
  - Calibration slope and exact Poisson intervals for observed rates.
- We renamed the “near-prospective” task the “most recent temporal hold-out” because the outcomes already existed when we scored the task.
- The original plan is archived as World_Model_Analysis_Plan_v1.0.md.

**A9 (18 November 2026, after all A8 outcomes had been computed). Measurement corrections.**

PubTator3 assigned the mention “fat” mainly to a disease identifier through 2023 and to a chemical identifier from 2024, creating the appearance that the latter identifier had acquired many new associations. PubTator3 also annotated “metabolic dysfunction” within the new disease name as a separate disease. The terminology change therefore generated new co-mentions. Both entities were frequent among the positive pairs.

We applied three rules in all years and analyses using annotation and text information without reference to individual pair outcomes:

1. We removed annotations within the full names for NAFLD, NASH, MAFLD, MASLD, MASH and MetALD, and within “steatotic liver disease”. The encoder already used neutralised disease names. This correction applied the corresponding rule to the entity graph.
2. We merged identifiers whose most frequent mentions were identical after normalisation. This extended A6 from pair exclusions to the entity definitions. The species-specific analysis from the original plan retained one entity per identifier.
3. We defined an entity’s annotation rate as the number of papers in which PubTator3 annotated it divided by the number of papers containing its most frequent mention. We excluded entities whose rate changed by a factor of 3 or more between adjacent years if both years had at least 20 papers containing the mention and at least one year had 20 or more annotated papers. We treated such jumps as changes in annotation.

Further changes to the record and outputs:

- The run-history table includes the A8 development run of 17 November 2026, which covered a subset of cutoffs.
- The rule check reports how the cutoff-specific MASLD-family exclusion would affect the positive pairs.
- We also report precision intervals for the hazard model.

**A10 (18 November 2026, 05:15 UTC, after the A1–A9 outcomes were available and before the new model was run on any test task). Development of the foundation-model extension.**

At the authors’ request, we built a model drawing on three Google Research systems to examine their application to literature forecasting:

- The Population Dynamics Foundation Model (PDFM; arXiv:2411.07207).
- SensorWM, the latent world model developed for wearable cardiovascular–kidney–metabolic profiling (arXiv:2611.08568 and arXiv:2603.27787).
- The Earth AI planetary geospatial foundation models for public health (arXiv:2610.05699).

We reimplemented the relevant architectural components for biomedical entities from the published descriptions. TimesFM 2.5 was the only model for which we used the pretrained weights directly (200 million parameters, Apache-2.0).

The new model was implemented in code/w6_foundation_world_model.py and was designated the primary model at this stage. It had four components.

1. **Graph encoder, adapted from PDFM.** We constructed annual entity graphs from 2005 onwards using the entities in V_t. Edges represented sentence co-mentions weighted by paper counts or each node’s 10 nearest neighbours in the language representation. Node inputs were the whitened PubMedBERT state with the basis fixed at 2019 (language), annual paper counts for the preceding five years, cumulative count, growth and time since first mention (attention), and entity type, degree, weighted degree and the type composition of neighbours (graph context). The encoder used one GELU layer, two GraphSAGE layers with pooling aggregators and a linear projection to 64 dimensions. The language, attention and context blocks had 32, 16 and 16 dimensions, respectively. Each block reconstructed its own input using Huber loss, with the three losses combined in a balanced sum. Inputs were masked for a random 30% of nodes at each step. We trained the encoder with Adam and a cosine schedule for 40 passes over the 2005–2019 snapshots, then froze it.
2. **Latent dynamics, adapted from SensorWM.** We used F_t = g(z_t, a_t), Δẑ_{t+1} = F_t Δz_t and ẑ_{t+1} = z_t + Δẑ_{t+1}. The function g used a multiplicative action gate and spectral normalisation. The actions a_t were the log paper count in the step year and its change. The loss combined rollout MSE with twice the path MSE over three steps. At year t, we fitted the dynamics model on an expanding window of transitions ending by t.
3. **Attention forecasts, adapted from PDFM and Earth AI.** Zero-shot TimesFM 2.5 forecast log(1 + papers per year) for three years. A two-hidden-layer multilayer perceptron took the forecast, the embedding and the last observation as inputs. We trained this adapter on data through t and used its adjusted forecasts as actions for the rollout.
4. **Prediction heads, adapted from PDFM and Earth AI.** The heads used gradient-boosted trees with the A8 baseline settings. Their inputs were historical features (graph topology, attention and entity-type pair), elementwise products and absolute differences of the two entity embeddings, blockwise cosines of the current and rolled-out embeddings, and each entity’s forecast attention.

The primary contrast compared the new model with resource allocation on the primary task (2022, Δ = 3). Secondary comparisons included the history-only head, the head with embeddings but without dynamics or forecasts, and the original hazard world model refitted on the same training tasks. We ablated each feature block (language, attention, graph context, rollout and TimesFM) and reran the pipeline with a random encoder. We compared attention forecasts with the last observation and with TimesFM alone using mean absolute error. For the latent dynamics, we used three references: the last observation carried forward, TimesFM applied to each latent dimension, and rollouts using observed activity. Secondary evaluations covered the full task grid, all candidate pairs and the most recent hold-out. We used node-bootstrap intervals with a separate generator (seed + 10).

We first checked the implementation on one development task with cutoff 2016 and Δ = 3, using training cutoffs from 2010 through 2013 and labels from 2017–2019. The development run was intended to check the code. Any subsequent changes to settings were restricted to corrections and recorded in this amendment. The authors had already seen the A1–A9 outcomes for the same tasks when the new model was specified. All retrospective comparisons of the new model were therefore exploratory.

**Corrections after the development run (18 November 2026, 05:45 UTC, before any test task was run).**

1. The TimesFM adapter overfitted when trained for a fixed 300 epochs. On the development task, its mean absolute error was 0.525, compared with 0.439 for TimesFM alone. We added early stopping on a 20% held-out subset of entities, following the 20% split used for PDFM tuning, and used weight decay of 10^-3. After this correction, the adapter performed at least as well as TimesFM alone in the 2013 and 2016 development years.
2. We simplified the formula used for the exact Poisson intervals in the calibration table. The estimates were unchanged.
3. We added a projection of the 2025 embeddings onto two principal components for the figures and a development flag in the run log.

The other settings were retained. The development AUROCs for link forecasting (0.975 for the new model and 0.977 for the refitted hazard model) did not inform any changes.

**Addition after the full A10 run (18 November 2026, 07:10 UTC, after the outcomes were available).** On the primary task, the new model had a slightly lower AUROC than the refitted hazard world model (difference −0.003, 95% CI −0.005 to −0.001).

**A11 (18 November 2026, about 08:00 UTC, after all A10 outcomes were available).**

We added the following exploratory analyses, each with its own random-number generator to preserve the draws used in earlier results:

1. Refit the neural hazard model specified in version 1.0 on each test task and calculate paired intervals for AUROC and popularity-adjusted AUROC.
2. Calculate paired intervals for popularity-adjusted AUROC against resource allocation and the history-only head on each task.
3. Add a head that uses historical covariates and the first 32 axes of the whitened PubMedBERT state. This comparison assesses the contribution of the graph encoder beyond the language state.
4. Compare the latent dynamics with drift (the state plus the mean three-year change over the training transitions), ridge regression of the change on the state and its last change, a transition model without actions, and rollouts with actions shuffled across entities. Calculate intervals for each embedding block.
5. Calculate reconstruction loss when each input is predicted by its mean, as a reference for the encoder.
6. Evaluate the hazard model on the most recent hold-out.

**A12 (18 November 2026, about 12:00 UTC, after all A1–A11 outcomes were available).**

At the authors’ request, we restored the neural model specified in version 1.0 (sections 4 and 5, as amended in A1–A9) as the primary model and referred to it as the “world model”. We retained the A10 architecture as the “foundation-model extension” to assess whether the additional architectural complexity improved forecasts.

The world model had the highest point estimates for AUROC and popularity-adjusted AUROC on the primary task. The two values were 0.976 and 0.923 in the reported A9 run and 0.976 and 0.924 when refitted in the A10–A11 comparison. The corresponding values for the foundation-model extension were 0.973 and 0.914. The comparison with boosted trees using all of the world model’s features did not establish an AUROC difference. The linear form of the world model had a slightly higher point estimate for precision among its 1,000 highest-ranked pairs.

The main results came from the A9 run. The paired comparison with the refitted world model and the results for the foundation-model extension came from the A10–A11 run. This amendment changed the designation of the primary model, with no refitting, setting changes or new outcome calculations. The designation was made after comparing the results, so the retrospective comparisons remained exploratory.

**A12 addendum (19 November 2026, after all outcomes were available). Descriptive semantic atlas.**

At the authors’ request, we added a descriptive figure showing how the world-model state was organised by research domain, presented in the final paper as Fig. 3.

We linked the entities in the 2022 graph to the companion topic model via PubMed identifiers. The topic model used disease-name-neutralised PubMedBERT representations of papers published from 2020 through June 2026 and identified 91 topics across 10 domains. We assigned each entity with at least three linked papers through 2022 to the domain with the largest Pearson residual for its linked papers. We first tried assignment by the most frequent domain, which placed most genes and chemicals in the largest domain (experimental research). We chose the Pearson residual rule after seeing that distribution.

We report standardised summaries of the entity state by domain. For the primary task, we report the AUROC of each pair feature evaluated alone and of the world model, both across all candidate pairs and for pairs involving each domain. We refitted the world model using the training tasks and random-number generator from the comparison run and reproduced that run’s AUROC.

The model, its settings and the comparisons were unchanged. The code is in code/w10_semantic_atlas.py and code/fig_atlas.py.

- **19 November 2026, later the same day.** At the authors’ request, the study design moved to Fig. 1, the architecture to Extended Data Fig. 1 and the semantic atlas to Fig. 2. These were figure changes only.
- **21 November 2026.** Fig. 1 was assigned to the world model’s temporal design and architecture, Fig. 2 to the research topics and domains, and Fig. 3 to the semantic atlas. The study design moved to Extended Data Fig. 1 and the foundation-model extension architecture to Extended Data Fig. 5. These were figure changes only.
- **22 November 2026.** Extended Data Figs. 1–6 became Supplementary Figs. 1–6. Extended Data Tables 1 and 2 became Supplementary Tables 1a and 3. Supplementary Information 1 contains the supplementary figures and tables. Supplementary Information 2 contains the notes, renumbered by first citation. The analysis plan appears as Supplementary Note 1.
