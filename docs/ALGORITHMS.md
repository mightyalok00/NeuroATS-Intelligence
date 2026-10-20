# Algorithm Notes

This document describes the main algorithm families used by NeuroATS Intelligence and how to interpret their outputs. The equations are written as ordinary Markdown code or text so they remain readable on GitHub mobile and other Markdown renderers.

> **Scope:** These notes explain algorithm concepts and the intended role of the components. They are not a substitute for implementation inspection or empirical evaluation. Exact behavior depends on the current code and configuration.

## 1. Resume-quality rubric

The ATS-style score combines configured dimensions into a weighted score. In general form:

```text
score(resume) = sum(weight_i * dimension_score_i)
```

Dimensions may include contact details, document structure, action verbs, quantified impact, technical skills, readability, and length. The weights and normalization rules should be treated as application configuration, not as a universal industry standard.

**Interpretation:** Use the score to identify areas for editing. It is not a hiring probability, a candidate-quality measurement, or a reproduction of a commercial ATS score.

## 2. BM25 lexical relevance

BM25 ranks documents against a query using query-term frequency, inverse document frequency, and document-length normalization. A common form is:

```text
BM25(D, Q) = sum over q in Q of:
  IDF(q) * [ f(q,D) * (k1 + 1) ] /
           [ f(q,D) + k1 * (1 - b + b * |D| / avgdl) ]
```

Where:
- `D` is a document and `Q` is a query.
- `f(q,D)` is the frequency of term `q` in document `D`.
- `|D|` is the document length and `avgdl` is average document length.
- `k1` controls term-frequency saturation and `b` controls length normalization.
- `IDF(q)` weights terms based on their distribution across the document collection.

A commonly used inverse document frequency expression is:

```text
IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))
```

Here `N` is the number of documents and `n(q)` is the number of documents containing term `q`.

**Use in NeuroATS:** lexical relevance can help retrieve resume passages that share terms with a query. BM25 alone does not understand all synonyms or the context of a qualification, so results should be reviewed.

## 3. TF-IDF and cosine similarity

TF-IDF represents text using term-weighted vectors. A common document-frequency weighting is:

```text
tfidf(term, document) = tf(term, document) * idf(term)
```

For two vectors, cosine similarity is:

```text
cosine_similarity(A, B) = (A dot B) / (||A|| * ||B||)
```

This measures the angle between vector representations. It is useful for comparing text, but similar wording does not guarantee equivalent experience, and different wording can describe the same skill.

## 4. TextRank sentence ranking

TextRank builds a graph whose nodes represent sentences and whose edges represent sentence similarity. A PageRank-style update is:

```text
rank(u) = (1 - d) / |V|
          + d * sum over v linked to u of
            [ weight(v,u) / sum_k weight(v,k) ] * rank(v)
```

Where `V` is the sentence set, `d` is a damping factor, and edge weights represent sentence similarity. Higher-ranked sentences may be useful for extractive summaries; ranking is not a factuality guarantee.

## 5. Dense embeddings and LSA fallback

Dense sentence embeddings can represent semantic similarity beyond exact word overlap. Latent Semantic Analysis (LSA) uses a low-rank decomposition of a term-document matrix to form compact text representations. These approaches have different trade-offs:

- Dense embeddings may capture paraphrases more effectively but can require downloading and caching a pretrained model.
- LSA is based on the data and vocabulary used to build its representation and may behave differently from a pretrained embedding model.
- Similarity scores are ranking signals, not calibrated probabilities.

## 6. Evaluation requirements

A credible algorithm claim needs an evaluation design that matches the task. Before reporting results:

1. Define the target label (for example, extraction correctness, relevance, or ranking quality).
2. Use permitted data with documented provenance.
3. Split data by candidate before fitting preprocessing or models to avoid leakage.
4. Compare with simple baselines.
5. Report task-appropriate metrics, sample counts, and failure analysis.
6. Record configuration, random seeds, dependencies, and commands.
7. Publish results only after running the evaluation.

For the project's fairness, privacy, and release checklist, see [Evaluation, Fairness, and Limitations](EVALUATION_AND_FAIRNESS.md).
