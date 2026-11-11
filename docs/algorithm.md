# Algorithm notes

This note states the ranking framework, the confidence signal, and the decision
policy used by the serving layer. It is explicit about what is *not* being
claimed: raw scores are not probabilities, and co-usage reasons are not causal
effects.

## Framework at a glance

For a training matrix \(X\) and a user \(u\) with history
\(H_u = \{i : X_{u,i} > 0\}\), every model in this repository implements

\[
f_\theta : U \times I \to \mathbb{R}, \qquad
s_{u,i} = f_\theta(u,i),
\]

and the candidate set is \(C_u = I \setminus H_u\). The framework is the
composition of five independent stages:

1. **Fit** \(\theta\) on the training period only.
2. **Score** the full catalogue (or an explicit candidate subset).
3. **Mask** \(H_u\); this is performed by the evaluator and the service, not
   by the model.
4. **Rank** by \(s_{u,i}\), optionally apply a percentile threshold and/or an
   MMR diversity rule.
5. **Evaluate** against the next-period adoption labels.

Domain configuration supplies vocabulary and an optional serving blocklist. It
does not change \(X\), \(C_u\), \(f_\theta\), or any metric. This separation
is tested in `tests/test_math_verification.py`.

## 1. Item-kNN scoring

Let \(X \in \mathbb{R}^{|U| \times |I|}\) be the sparse training matrix, with
\(X_{u,i} = \log(1 + \min(w_{u,i}, q_{0.999}))\). Let \(x_i\) be the \(i\)-th
column. Cosine similarity is

\[
S_{i,j} = \frac{\langle x_i, x_j \rangle}{\lVert x_i \rVert_2 \, \lVert x_j \rVert_2},
\qquad S_{i,i} = 0.
\]

For each item we retain the top \(K=100\) neighbours, yielding a sparse matrix
\(\tilde{S}\). For a user with history \(H_u\) and unseen candidate set
\(C_u = I \setminus H_u\), the score is

\[
s_{u,i} = \sum_{j \in H_u} X_{u,j} \, \tilde{S}_{j,i}.
\]

The truncation is a sparsity/regularisation choice: it keeps the artefact small
and makes each reason exact, because the retained neighbour graph *is* the
explanation. The score has units of similarity mass and is not comparable
across users with different history lengths.

## 2. Within-user percentile

Raw scores are monotone within a user but not calibrated across users. We expose

\[
q_{u,i} = \frac{\left|\{ j \in C_u : s_{u,j} \le s_{u,i} \}\right|}{|C_u|}
\in (0, 1].
\]

Properties:

- **Distribution-free.** It makes no Gaussian or logistic assumption.
- **User-comparable.** A value of 0.99 means "top 1% of this user's unseen
  candidates", independent of history length.
- **Monotone invariant.** Any strictly increasing transform of \(s_{u,\cdot}\)
  leaves \(q\) unchanged.
- **No future labels.** It uses only the current score vector.

The API reports this as `percentile`; the dashboard renders `TOP x%` as
\(100(1-q)\). It is a **relative confidence**, not \(P(\text{adopt})\).

## 3. Decision policy

The API exposes this as `min_percentile = q`. For a requested user, let
\(Q_q\) be the empirical \(q\)-quantile of \(\{s_{u,i}: i \in C_u\}\) and

\[
C_u(q) = \{ i \in C_u : s_{u,i} \ge Q_q \}.
\]

The slate is the top-\(k\) by \(s_{u,i}\) within \(C_u(q)\). If \(C_u(q)\) is
empty the service returns no items with strategy `threshold_filtered`; it does
not silently substitute an unrelated popularity list. The percentile shown on a
returned item is computed against the **original** unseen set \(C_u\), not the
filtered set, so the number remains comparable across thresholds.

The default is \(q=0\) (no abstention). Because \(q\) is a rank transform, the
threshold has a direct interpretation: \(q=0.99\) means "only the top 1% of
this user's unseen candidates". Increasing \(q\) can only shrink the slate; it
does not change the relative order of items that remain.

For exploratory surfaces, use `diversify=true`. The MMR objective is

\[
\max_{i \in R} \; (1-\lambda)\,\tilde{s}_{u,i}
- \lambda \max_{j \in S} \cos(e_i, e_j),
\]

where \(S\) is the selected set, \(\tilde{s}\) is the score normalised by the
best candidate, and \(e_i\) are item factors when the model exposes them. MMR
trades relevance for redundancy; it is a product choice, not a metric
improvement.

## 4. What we do not claim

- \(s_{u,i}\) is **not** a probability. A score of 200 does not mean "20% chance".
- \(q_{u,i}\) is **not** a probability either; it is a percentile.
- Explanation reasons are **co-usage**, not causal effects. A user who used
  \(j\) and later adopted \(i\) may share a latent preference; the data cannot
  identify why.
- The benchmark is offline. No position bias, interference, or long-horizon
  effect is modelled.

If a deployment needs calibrated probabilities, fit a calibrator on a
*separate* time window and report reliability/Brier/log-loss on a further
holdout. We deliberately do not ship an unvalidated probability.

## 5. Complexity

Let \(n\) be the number of users, \(m\) the number of items, \(K\) the
neighbour count, and \(h_u = |H_u|\).

- **Item-kNN fitting:** sparse item-block products over \(m\) items, with
  \(O(\sum_i \text{nnz}(x_i))\) work per block and top-\(K\) selection.
- **Serving one user:** \(O(h_u K + m \log m)\) for scoring, masking, and the
  full sort; the top-\(k\) selection can use `argpartition` in \(O(m)\).
- **Artefact size:** \(O(mK)\) similarity entries plus the user history.
- **Batch evaluation:** chunks of `eval_batch_size` users, bounded memory.

For catalogues above ~100k items, replace the full-catalogue sort with an ANN
candidate generator and keep the same evaluation contract. The current
reference catalogue (9,848 items) does not need ANN.

## 6. Reproducibility

The model is deterministic given the January matrix and `K`. The dataset
fingerprint and artefact digest are stored in the manifest. The evaluator
masks \(H_u\) centrally, so the serving layer and offline metric use the same
candidate set. See [`methodology.md`](methodology.md) and
[`evaluation.md`](evaluation.md).
