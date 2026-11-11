# Methodology

This document specifies the task, estimators, evaluation protocol, and statistical analysis used by EngageGraph 2.0. It is written to be sufficient for an independent researcher to reproduce every reported number and to audit every design choice.

## 1. Problem statement

Let \(U\) be a set of panelists and \(I\) a catalogue of apps. For a calendar month \(m\), the observed interaction log is a set of triples \((u, i, w_{u,i}^{(m)})\), where \(w_{u,i}^{(m)} \ge 0\) is an engagement weight (foreground duration). The January log is the **training month**; the February log is the **label month**. We study the following task:

> **Next-month new-app adoption ranking.** Given a panelist \(u\) and their January history \(H_u = \{i : (u,i) \in \mathcal{E}_{\text{Jan}}\}\), rank all apps in the January catalogue \(I_{\text{Jan}}\) that are **not** in \(H_u\). A recommendation is relevant if \(u\) interacts with that app in February.

Formally, for each eligible user we produce a ranking \(\pi_u\) of \(I_{\text{Jan}} \setminus H_u\), and evaluate it against the binary relevance set

\[
R_u = \{ i \in I_{\text{Jan}} : (u,i) \in \mathcal{E}_{\text{Feb}},\; i \notin H_u \}.
\]

This is a *future adoption* task, not repeat-usage prediction and not next-session prediction. The distinction matters: repeat usage is dominated by a user's existing repertoire, whereas new-app adoption is the operationally interesting and much harder decision.

### Why a temporal split

Random interaction splits leak future co-usage into training and inflate ranking metrics. A temporal split also matches deployment: a model fitted on last month's data ranks candidates for next month. We therefore fit **all** parameters — mappings, catalogue, popularity, item similarities, latent factors, graph embeddings — on January only, and touch February once, to construct \(R_u\).

### Candidate catalogue

The catalogue is fixed from January: apps used by at least 10 distinct January panelists, capped at the 10,000 most-supported apps. Apps that appear only in February are out of scope and are counted in the data card. This is deliberately conservative: adding February-only apps would leak the future catalogue.

### Eligibility

- A user is eligible for training if they used at least 3 distinct catalogue apps in January.
- A user is eligible for ranking evaluation if they have at least one February adoption in the catalogue that they did not use in January.
- February users with no January history are a **cold-start slice**: they cannot be personalised from January and are reported separately. They are never silently dropped from the data card.

## 2. Data preparation

### 2.1 Cleaning

Raw rows are dropped when either identifier is missing or when the weight is missing, non-numeric, or negative. App labels are Unicode NFKC-normalised, stripped of control and bidirectional-override characters, and whitespace-collapsed. Duplicate `(user, app)` rows are summed. The cleaner returns a `DataQualityReport` recording every removal; the report is stored in `manifest.json`.

The weight unit is **inferred** to be milliseconds from the distribution (median ≈ 4.2 × 10⁵; 90th percentile ≈ 5.6 × 10⁷). No codebook ships with the exports, so this remains an inference; all reported quantities are therefore based on **relative** weights, and the benchmark metrics are invariant to a global unit rescaling.

### 2.2 Weight transform

Raw durations span nine orders of magnitude and include implausible outliers (maximum ≈ 1.1 × 10¹¹). We clip at the 99.9th percentile of January catalogue weights and apply

\[
r_{u,i} = \log(1 + \min(w_{u,i}, q_{0.999})).
\]

`log1p` compresses heavy tails while preserving ordering. The transform is a configuration option (`log1p`, `binary`, `share`, `raw`); the primary benchmark uses `log1p`, and the sensitivity analysis reports `binary`.

Crucially, the transform is fitted on **January only** and applied unchanged to February. February weights are used only to define binary relevance, not to fit any model.

### 2.3 Matrices

Let \(X \in \mathbb{R}^{|U| \times |I|}\) be the sparse January matrix with \(X_{u,i} = r_{u,i}\). Let \(Y \in \{0,1\}^{|U| \times |I|}\) be the February relevance matrix restricted to catalogue items and eligible users. By construction, \(X_{u,i} > 0 \Rightarrow Y_{u,i} = 0\): an app used in January is never a February adoption label. This invariant is asserted in the pipeline and covered by tests.

## 3. Estimators

All models are fitted on \(X\) only. They expose a score vector over the full catalogue; masking seen items is performed by the evaluator and serving layer, never by the model, so every model is compared on exactly the same candidate set.

### 3.1 Popularity

\[
s_u(i) = \log(1 + |\{u' : X_{u',i} > 0\}|).
\]

Popularity is a strong adjacent-month baseline because catalogue demand is persistent. It is also the cold-start fallback. Any personalised model that does not beat popularity has no demonstrated user-specific signal.

### 3.2 Item-kNN

Item vectors are the columns of \(X\) (binary by default). Cosine similarity is

\[
\mathrm{sim}(i,j) = \frac{\langle x_i, x_j \rangle}{\lVert x_i \rVert_2 \lVert x_j \rVert_2}.
\]

For each item we retain the top 100 neighbours, producing a sparse item–item matrix \(S\). A user's score for item \(j\) is

\[
s_u(j) = \sum_{i \in H_u} X_{u,i} \, S_{i,j}.
\]

The top-\(k\) truncation keeps the artefact small and makes explanations exact: the neighbour graph is the reason. This is the standard item-based CF formulation (Sarwar et al., 2001).

### 3.3 Implicit ALS

We use the implicit-feedback ALS objective of Hu, Koren & Volinsky (2008). With confidence \(c_{u,i} = 1 + \alpha X_{u,i}\) and \(\alpha = 40\), we solve

\[
\min_{P,Q} \sum_{u,i} c_{u,i} \left( X_{u,i} - p_u^\top q_i \right)^2 + \lambda \left( \lVert P \rVert_F^2 + \lVert Q \rVert_F^2 \right),
\]

with 64 factors, \(\lambda = 0.05\), and 15 alternating least-squares sweeps. Scores are \(s_u(i) = p_u^\top q_i\). The solver is the well-tested `implicit` implementation; the random seed is fixed.

### 3.4 MetaPath2Vec

The January matrix defines a bipartite graph \(G = (U \cup I, E)\) with an edge for every observed interaction. We learn 64-dimensional embeddings with the metapath2vec objective (Dong, Chawla & Swami, 2017):

\[
\max_{\theta} \sum_{u \in U}\sum_{i \in I} \sum_{c \in C(i)} \log \sigma\!\left( \mathbf{e}_c^\top \mathbf{e}_u \right) + \sum_{j \sim P_n(i)} \log \sigma\!\left( -\mathbf{e}_j^\top \mathbf{e}_u \right),
\]

where \(C(i)\) are context nodes sampled by meta-path-guided random walks and \(P_n(i)\) is the noise distribution. We use the meta-path `app → user → app`, walk length 20, context size 5, 5 walks per node, 5 negative samples, and 3 epochs. Scores are dot products of the learned user and app embeddings.

This is the canonical objective. The original notebook's read-out applied an elementwise "cosine" and averaged it, producing the constant 0.0156 observed in its output; that read-out is not part of metapath2vec and is not used here.

### 3.5 Hyperparameter policy

The primary configurations — popularity, item-kNN with `k=100`, and ALS — were fixed before the February evaluation from literature defaults and January-only reasoning. No February metric was used to select the headline model or the primary k.

Two exploratory adjustments were made after inspecting February diagnostics, and are disclosed rather than presented as pre-registered:

1. **MetaPath2Vec read-out.** The canonical user-vector dot product produced near-random top-10 rankings. We replaced it with a profile read-out (engagement-weighted mean of the user's January item embeddings) and increased the epoch count. The reported MetaPath2Vec result is therefore **exploratory**, not a pre-registered comparison. It is included because omitting the model that motivated the original repository would be misleading.
2. **Item-kNN sweep.** We evaluated `k ∈ {50, 100, 200, 300}`, a non-binary variant, and a popularity blend on February as a sensitivity check. The pre-specified `k=100` was retained because it was the best or statistically indistinguishable from the best. The sweep did not change the primary configuration.

The item-kNN vs. popularity comparison, which is the paper's main positive result, does not depend on either adjustment. Readers who require a strictly pre-registered comparison should treat the MetaPath2Vec row as descriptive.

## 4. Evaluation

### 4.1 Ranking metrics

For a user with relevance set \(R_u\) and a ranked list \(\pi_u^{(K)}\) of length \(K\):

- **Recall@K** \(= |\pi_u^{(K)} \cap R_u| / |R_u|\)
- **Precision@K** \(= |\pi_u^{(K)} \cap R_u| / K\)
- **HitRate@K** \(= \mathbf{1}[\pi_u^{(K)} \cap R_u \neq \emptyset]\)
- **NDCG@K** \(= \mathrm{DCG}@K / \mathrm{IDCG}@K\), with \(\mathrm{DCG}@K = \sum_{r=1}^{K} \frac{\mathbf{1}[\pi_{u,r} \in R_u]}{\log_2(r+1)}\) and \(\mathrm{IDCG}@K = \sum_{r=1}^{\min(|R_u|,K)} \frac{1}{\log_2(r+1)}\)
- **MAP@K** \(= \frac{1}{\min(|R_u|,K)} \sum_{r=1}^{K} \mathbf{1}[\pi_{u,r} \in R_u] \cdot \frac{|\pi_u^{(r)} \cap R_u|}{r}\)
- **MRR@K** \(= 1/r^\*\) if the first relevant item is at rank \(r^\* \le K\), else 0.

Metrics are computed per user and macro-averaged, so a user with 60 adoptions does not dominate a user with one. The **primary metric is NDCG@10**, chosen because it rewards early placement and is normalised for users with different numbers of relevant apps.

### 4.2 Beyond accuracy

- **Coverage@10**: fraction of catalogue items that appear in at least one user's top 10. Low coverage signals a filter bubble.
- **Gini@10**: inequality of recommendation exposure across items.
- **Novelty@10**: mean \(-\log_2 p(i)\) where \(p(i)\) is the January user support share. Higher is less obvious.

These are reported alongside accuracy because a model can win NDCG by recommending the same popular apps to everyone.

### 4.3 Statistical analysis

Users are the resampling unit. For each pair of models we compute the per-user difference in NDCG@10 and run a **paired bootstrap** with 1,000 resamples, reporting the mean difference and a 95% percentile interval. The two-sided bootstrap p-value uses the standard \(+1\) correction. We additionally report a Wilcoxon signed-rank test as a distribution-free check and apply **Holm–Bonferroni** correction across the family of model comparisons against the baseline. Intervals and p-values are descriptive of this panel and month; they are not a claim about all app markets.

### 4.4 Cold-start slice

February users with no January history cannot be scored by a January-fitted personalised model. We report their count and evaluate the popularity fallback on them as a separate, clearly labelled slice. Combining warm and cold users into one headline number would hide the population for whom the product has no personal signal.

### 4.5 What is *not* the primary evaluation

A balanced link-prediction AUC is not the primary metric. It answers a different question ("can the model separate a sampled positive edge from a sampled non-edge?") and is highly sensitive to negative construction. The original notebook's 0.9257 AUC trained and tested on February edges and selected negatives by similarity. We reproduce and quantify that sensitivity in [`reports/legacy-audit.md`](../reports/legacy-audit.md) rather than presenting it as recommendation quality.

## 5. Leakage controls

| Risk | Control |
| --- | --- |
| Future catalogue | Catalogue built from January only |
| Future popularity | Popularity, item norms, IDF, clipping thresholds from January only |
| Future users | Mappings fitted on January; February-only users are cold-start, not imputed silently |
| Seen-item leakage | All January items masked at scoring time |
| Target leakage | Test labels never passed to `fit`; evaluator reads only `Y` |
| Hyperparameter leakage | Primary configs fixed before February; post-hoc MetaPath2Vec/item-kNN diagnostics disclosed in §3.5 |
| Duplicate leakage | `(user, item)` pairs aggregated; train/test disjointness asserted |
| Weight leakage | February weights used only for binary relevance, never for model fitting |

## 6. Limitations

1. **One panel, two months.** The benchmark measures January → February 2024 generalisation for one panel. It is not evidence of long-horizon or cross-market generalisation.
2. **No content or demographics.** App names are opaque labels; there is no category taxonomy, developer, price, or user demographic. Explanations are co-usage, not causal.
3. **Panel selection.** Panelists opt into measurement; their behaviour may not represent the population. The data card records this.
4. **Weight ambiguity.** The unit is inferred. Relative comparisons are safe; absolute "hours saved" claims are not.
5. **Label noise.** An app absent from February may reflect churn from the panel, not disinterest. The binary label cannot distinguish these.
6. **Binary relevance.** We do not claim graded engagement prediction; the weight transform sensitivity analysis is reported separately.
7. **Static evaluation.** The benchmark is offline. Online feedback, position bias, and interference are out of scope for this release.

## 7. Ethical considerations

The exports contain pseudonymous panelist identifiers and app-usage durations. They do not contain names, demographics, or content. Nevertheless, behavioural telemetry can be sensitive. We do not attempt re-identification, do not join external data, and recommend hashing identifiers at ingest in any deployment. The demo API accepts panelist IDs because it runs locally against the research export; a public deployment should put it behind authentication and replace raw IDs with opaque tokens.

## 8. References

- Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. *ICDM*.
- Dong, Y., Chawla, N. V., & Swami, A. (2017). metapath2vec: Scalable representation learning for heterogeneous networks. *KDD*.
- Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001). Item-based collaborative filtering recommendation algorithms. *WWW*.
- Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). BPR: Bayesian personalized ranking from implicit feedback. *UAI*.
- Steck, H. (2019). Embarrassingly shallow autoencoders for sparse data. *WWW*.
- Sculley, D., et al. (2015). Hidden technical debt in machine learning systems. *NeurIPS*.
