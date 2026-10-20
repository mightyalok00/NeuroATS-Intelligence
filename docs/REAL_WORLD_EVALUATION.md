# Real-World Resume/Job Evaluation Protocol

## Status

**Not yet validated on a real-world dataset.** The repository's checked-in ranking fixture is synthetic and its relevance labels are not independently adjudicated. This document defines the requirements for a future external evaluation; it does not claim that a dataset has been acquired, licensed, reviewed, or evaluated.

## Dataset admission checklist

Before importing data, record all of the following in a dataset card:

- Dataset name, publisher, version, retrieval date, canonical source URL, and checksum.
- License / terms of use and a written determination that resume and job-description evaluation is permitted.
- Consent, lawful basis, provenance, and any redistribution restrictions.
- Which fields are included, excluded, transformed, or redacted.
- Sampling frame, languages, countries/regions, job families, career levels, and known coverage gaps.
- Label definition, annotator instructions, number of annotators, disagreement resolution, and label date.
- Known duplicate, near-duplicate, source, and temporal leakage risks.

Do not commit raw resumes, contact details, credentials, private labels, or restricted data to this repository. Keep data outside Git and CI unless its license explicitly permits that use. CI should run only on synthetic fixtures or a sanitized, distributable subset with documented permission.

## Evaluation design

1. **Define the task before scoring.** Separate parsing quality, skill extraction, job-related relevance, ranking, and any downstream hiring outcome. Do not use one as a proxy for another.
2. **Freeze a split by candidate/person first.** All documents from the same candidate must remain in one split. Where appropriate, additionally separate by job family, source, employer, or time. Deduplicate before splitting.
3. **Prevent leakage.** Fit vocabulary, IDF weights, feature selection, calibration, and thresholds using development data only. The held-out test set is for final evaluation only.
4. **Compare baselines.** Include keyword overlap, BM25, TF-IDF, and the project's actual production scoring path where applicable. Ensure all methods receive the same documents and queries.
5. **Predeclare metrics.** For ranking, report MRR, Recall@k, and NDCG@k at declared cutoffs. For extraction/classification, report precision, recall, macro-F1, and a confusion matrix as appropriate.
6. **Report uncertainty and failures.** Include query/candidate counts, confidence intervals where defensible, results by job family, and representative false positives and false negatives. Do not report only a macro average.
7. **Keep an audit trail.** Record code commit, dataset version/checksum, split seed, dependency versions, label version, exact command, metric definitions, and all exclusions.
8. **Separate development from confirmation.** Tune only on development data. Do not repeatedly tune against the held-out test set.

## Human review requirements

Use at least two reviewers for relevance labels, blinded to system rankings and the original labels. Define a job-related rubric in advance; distinguish relevant evidence from generic keyword overlap. Report raw agreement and Cohen's kappa when appropriate, with limitations for small or imbalanced samples. Adjudicate disagreements and version the final labels. Reviewers and the data owner must approve use of the dataset before analysis.

## Acceptance gate

A report must not call the system "accurate", "fair", "validated", or "production-ready" solely because this evaluation script runs. Those claims require results from the appropriate permitted data, documented human review, a frozen test set, and a scoped claim supported by evidence.
