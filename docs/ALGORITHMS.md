# Algorithm Notes

This document explains the main algorithm families used by NeuroATS Intelligence and how their outputs should be interpreted. These are general algorithm descriptions; exact behavior depends on the implementation and configuration.

> **Important:** A formula or passing unit test does not demonstrate real-world ranking accuracy, fairness, or hiring validity. See [Evaluation, Fairness, and Limitations](EVALUATION_AND_FAIRNESS.md).

## 1. Resume-quality rubric

The configurable rubric combines dimension scores using weights:

```text
score(resume) = sum(weight_i * dimension_score_i)
```

Dimensions may include contact details, document structure, action verbs, quantified impact, technical skills, readability, and length. The result is resume-improvement feedback—not a hiring probability or an estimate of a proprietary ATS score.

## 2. BM25 lexical relevance

BM25 ranks documents against a query using term frequency, inverse document frequency, and document-length normalization:

```text
BM25(D, Q) = sum over q in Q of:
  IDF(q) * [ f(q,D) * (k1 + 1) ] /
           [ f(q,D) + k1 * (1 - b + b * |D| / avgdl) ]
```

Where `D` is a document, `Q` a query, `f(q,D)` term frequency, `|D|` document length, and `avgdl` average document length. `k1` controls term-frequency saturation; `b` controls length normalization.

A commonly used IDF expression is:

```text
IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))
```

Here `N` is the corpus size and `n(q)` is the number of documents containing the term. BM25 helps retrieve text with matching terminology but does not fully understand context or synonyms.

## 3. TF-IDF and cosine similarity

TF-IDF weights terms by their occurrence in a document and their rarity across a corpus. Cosine similarity between vectors `A` and `B` is:

```text
cosine_similarity(A, B) = (A dot B) / (||A|| * ||B||)
```

This is a similarity signal, not a probability. Similar wording does not guarantee equivalent experience, and different wording can describe the same skill.

## 4. TextRank sentence ranking

TextRank builds a graph in which nodes represent sentences and edges represent sentence similarity. A PageRank-style update is:

```text
rank(u) = (1 - d) / |V|
          + d * sum over v linked to u of
            [ weight(v,u) / sum_k weight(v,k) ] * rank(v)
```

`V` is the sentence set, `d` is a damping factor, and edge weights encode sentence similarity. Top-ranked sentences may help with extractive summaries; ranking is not a factuality guarantee.

## 5. Dense embeddings and LSA fallback

Dense sentence embeddings can represent semantic relationships beyond exact word overlap. Latent Semantic Analysis (LSA) uses a low-rank decomposition of a term-document matrix to form compact representations.

- Dense embeddings may better capture paraphrases but can require a pretrained model download on first use.
- LSA depends on the vocabulary and data used to construct its representation and may behave differently from a pretrained model.
- Similarity values should not be interpreted as calibrated probabilities.

## 6. Reproducible evaluation

The repository includes a small BM25 ranking smoke benchmark at `benchmarks/evaluate_ranking.py`. It uses a tiny, manually labelled illustrative corpus to verify metric calculations and provide a reproducible example—not a real-world resume-ranking performance claim.

A credible evaluation requires permitted representative data, candidate-level split controls, relevance labels, simple baselines, task-appropriate metrics such as Recall@k and NDCG@k, failure analysis, and documented configuration. Publish real-world claims only after running that evaluation.
