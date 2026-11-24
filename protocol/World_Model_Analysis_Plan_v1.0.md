# Prespecified analysis plan for a PubMedBERT world model of MASLD associations

Version 1.0 was finalised on 17 November 2026, before any forecasting outcomes were calculated. Later amendments are documented in the dated amendment log.

## 1. Research question

Can a model trained on MASLD literature available through a given cutoff identify pairs of biomedical entities that will first be reported together after that cutoff?

We define an association as the co-mention of two entities in the same sentence of a paper in the MASLD corpus. Eligible entities include genes, chemicals or drugs, diseases or phenotypes, and variants. The model forecasts future reporting in the literature. The validity of the underlying biological associations is outside the scope of this analysis.

## 2. Corpus

- **Records.** We include PubMed records published from 2000 through June 2026 that contain any of the 17 historical, interim or current disease terms in the title or abstract, or carry the MeSH heading Non-alcoholic Fatty Liver Disease. The query excludes six publication types across all years: Published Erratum, Retraction of Publication, Comment, Editorial, Letter and News.
- **Text and entities.** We use titles and abstracts from PubTator3 and its normalised annotations for the Gene, Chemical, Disease and Variant categories. Species and cell-line annotations are excluded. The same criteria apply in every year. The world-model corpus is assembled independently of the main 2020–2026 corpus.
- **Disease terminology.** Before embedding, we replace NAFLD, NAFL, MAFLD and MASLD with “SLD” and NASH and MASH with “steatohepatitis”, following the rules used in the main analysis. These substitutions limit the extent to which changes in terminology can signal the publication period.
- **Excluded entities.** The graph excludes the MASLD-family disease entities that define the corpus. These include identifiers for which at least half of the annotated mentions match a query term and the MeSH headings Non-alcoholic Fatty Liver Disease and Fatty Liver. We also exclude entities without a normalised identifier.

## 3. Forecasting task

| Element | Definition |
|---|---|
| Cutoff T | End of a publication year. |
| Node set V_T | Entities mentioned in at least 3 papers published through T. |
| Candidate pairs | Unordered pairs of entities in V_T that have never appeared together in a sentence through T. |
| Label Y_ij(T, Δ) | The label is 1 if the pair appears together in a sentence in at least 2 distinct papers published in (T, T + Δ]. |
| Test tasks | T = 2020, 2021, 2022 or 2023, with Δ = 1, 2 or 3 years, subject to T + Δ ≤ 2025. |
| Primary task | T = 2022 and Δ = 3, covering associations first reported in 2023–2025. |
| Near-prospective task | T = 2025 and Δ = 0.5. The model uses training data through 2025 and is evaluated on papers published from January through June 2026. |
| Training tasks | For each test task (T, Δ), eligible training cutoffs T' satisfy T' + Δ ≤ T. All training labels therefore come from papers published by the test cutoff. |

The earliest test cutoff is 2020. This places the evaluation labels in 2021 or later, after the PubMed abstracts collected in 2020 for encoder pretraining (Gu et al., arXiv:2007.15779).

## 4. World model

**Encoder E.** We use frozen PubMedBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract) to encode the first 256 word pieces of the title and abstract after applying the terminology substitutions. We average the final-layer token states to obtain the embedding e_d = E(x_d).

**Whitening.** For each cutoff T, we estimate the mean μ_T and the leading 128 principal axes using papers published through T. We centre, project and scale each embedding as ẽ_d = Λ_T^{-1/2} U_T^T (e_d − μ_T).

**Entity state and dynamics.** The semantic state h_i(T) is the weighted sum of the embeddings of papers that mention entity i through T, normalised to unit length:

- h_i(T) = Σ_d κ(T − t_d) 1[i ∈ d, t_d ≤ T] ẽ_d, normalised to unit length.
- The weight for a paper of age a is κ(a) = exp(−a/τ), with τ = 3 years.
- Semantic velocity over two years is v_i(T) = h_i(T) − h_i(T − 2).
- The projected state is ĥ_i(T + Δ) = normalize(h_i(T) + (Δ/2) v_i(T)).

**Field drift.** We define u_T as the difference between the mean of ẽ_d over the latest two years and the mean over the preceding two years. The alignment of an entity pair with that drift is ((h_i + h_j)/2) · u_T.

**Research attention.** The count n_i(T) is the number of papers mentioning entity i through T. Recent growth is g_i(T) = log[(n_i(T) − n_i(T − 2) + 1) / (n_i(T − 2) − n_i(T − 4) + 1)].

**Graph state.** From A_T, the sentence co-mention graph on V_T, we compute five pairwise features: common neighbours, resource allocation, Adamic–Adar, Jaccard and degree product.

**Pair features x_ij(T).**

- Current semantic cosine h_i·h_j and projected semantic cosine ĥ_i·ĥ_j.
- Semantic convergence, calculated as the cosine at T minus the cosine at T − 2.
- Semantic triadic closure, calculated as the largest cosine between i and a graph neighbour of j, averaged over the two directions.
- Alignment with field drift.
- Log attention and recent growth for both entities, sorted to make the representation independent of pair order.
- The five graph features after log1p transformation and indicators for entity-type pairs.

**Mathematical world model.** We model association formation with a discrete-time proportional-hazards model:

Pr(Y_ij(T, Δ) = 1 | S_T) = 1 − exp{−Δ exp(η_ij(T))}, where η_ij(T) = β_0 + βᵀ x_ij(T).

This is a complementary log–log model with offset log Δ. We fit it by L2-penalised maximum likelihood (penalty 1/C, C = 1) using standardised features from the training tasks.

**Neural world model (primary model).** The linear predictor includes a learned interaction between entity states, η_ij = βᵀ x_ij + g_θ(z_i, z_j, x_ij). The vector z_i contains the first 32 whitened coordinates of h_i. We represent the two entities symmetrically using z_i ⊙ z_j and |z_i − z_j|. The interaction g_θ is a two-layer perceptron trained with the same likelihood (64 units, ReLU, dropout 0.1, weight decay 1e-4, Adam, 30 epochs and a batch size of 4,096). Within each training task, we use all positives and 20 randomly sampled negatives per positive, with case–control weights.

## 5. Prespecified comparators

1. Degree product (preferential attachment).
2. Resource allocation, the best-performing topological score in the earlier analysis.
3. Static semantic similarity h_i·h_j, following the word-embedding approach of Tshitoyan et al.
4. A logistic model with resource allocation and static semantic similarity as predictors.
5. The mathematical world model, the hazard model fitted to x_ij.

## 6. Outcomes and statistical analysis

**Primary endpoint and contrast.** The primary endpoint is the neural world model’s AUROC on the primary task. We evaluate all positive pairs and a uniform random sample of 1,000,000 negative pairs, using all negatives if fewer are available. Uniform sampling of negatives preserves unbiased estimation of AUROC. We use inverse sampling weights for average precision and precision among the top k pairs. The primary contrast is the AUROC difference between the neural world model and resource allocation.

**Uncertainty.** We obtain 95% percentile intervals from 500 node-level bootstrap draws. In each draw, each entity receives a Poisson multiplier, and each pair is weighted by the product of its two multipliers.

**Secondary analyses.**

- AUROC for every test task (cutoff × horizon) and by entity-type pair.
- AUROC for pairs without common neighbours.
- AUROC for pairs involving advanced liver disease. The analysis uses MeSH identifiers fixed in the code before evaluation for cirrhosis and its complications, liver cancer, liver failure and liver transplantation.
- AUROC by the evidence type of the first papers reporting the pair (trial, observational, basic or review).
- Calibration by decile of predicted risk, time-to-report concordance and performance on the near-prospective task.

**Ablations.** We remove each feature block from x_ij in turn and refit the model: semantics, dynamics, field drift, graph features and attention.

**Sensitivity analyses.**

- Change the label threshold to at least 1 or at least 3 papers.
- Restrict the analysis to primary research.
- Set τ = ∞ to remove temporal decay.
- Replace PubMedBERT with NeuML/pubmedbert-base-embeddings to assess potential leakage from fine-tuning texts published after the cutoffs.
- Replace the encoder with a randomly initialised BERT of the same architecture as a negative control.
- Shuffle publication years within the training data as a negative control.

**Evaluation-design comparison.** To assess the effect of a non-temporal design on AUROC, we evaluate the same scores using a random 20% edge hold-out with never-mentioned pairs as negatives.

**Random seed.** 20261117.

## 7. Interpretation

We conclude that the forecast model outperforms a comparator when the 95% interval for the AUROC difference lies entirely above zero. Discrimination measures how well the model ranks pairs that are subsequently reported together. Causal and clinical validity are outside the scope of the evaluation. We report results for every test task.

## Amendments

None in version 1.0.
