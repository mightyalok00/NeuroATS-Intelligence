# Evaluation, Fairness, and Limitations

## Purpose

NeuroATS-Intelligence is a resume-analysis and job-description matching aid. Its scores are heuristic indicators of document structure and keyword alignment; they are **not** validated probabilities of hiring success, job performance, or candidate quality.

Do not use the output as the sole basis for hiring, rejection, interview selection, or compensation decisions. A human reviewer must assess the candidate's actual qualifications and relevant evidence.

## Evaluation protocol

Before describing a scoring change as an improvement, maintain a versioned benchmark with:

1. **Written task definition:** specify whether each label measures resume parsing correctness, skill extraction, job relevance, or a separate outcome. Do not mix these targets.
2. **Document provenance and permission:** use consented, licensed, or public benchmark data whose terms permit the intended use. Remove direct identifiers where practical.
3. **Stable splits:** split by candidate (and by source or time where applicable) before fitting vectorizers, classifiers, feature selection, or thresholds. Do not let resumes from the same candidate leak across train and test.
4. **Baselines:** compare against simple keyword overlap and a majority-class baseline where applicable.
5. **Metrics:** report precision, recall, macro-F1, and a confusion matrix for classification; report Recall@k, NDCG@k, or another predeclared ranking metric for candidate-job ranking when relevance labels exist.
6. **Uncertainty and failure analysis:** report sample counts, confidence intervals where feasible, performance by relevant job family, and representative false positives and false negatives.
7. **Reproducibility:** record dataset version, split seed, dependency versions, model artifact version, scoring rubric, and exact evaluation command.
8. **Honest reporting:** publish results only after the benchmark has been run. Never infer accuracy from a hand-picked demo, synthetic example, or deterministic output alone.

Keep a held-out test set untouched until model and scoring choices are finalized. If the dataset is too small to support a stable test set, clearly label results exploratory and use cross-validation within the development data.

## Fairness checks

Resume features can reflect unequal access to education, networks, career opportunities, language support, and document templates. Before any consequential use:

- Prefer job-related evidence over proxy signals such as name, photograph, age, address, graduation year, or unexplained employment gaps.
- Do not treat writing style, readability, prestigious employers, or uninterrupted career history as proof of competence.
- Test parsing and ranking errors across resume formats, career stages, job families, and language varieties where the data and legal basis permit.
- Where lawful, appropriate, and statistically meaningful, examine subgroup error rates using carefully governed data. Do not infer sensitive attributes from names or resumes.
- Document limitations, allow human override, and provide a way to challenge or correct extracted facts.
- Do not claim that the system is unbiased or fair without evidence.

## Privacy and document safety

- Process resumes locally where possible and clearly disclose any external network calls.
- Do not commit real resumes, contact details, access tokens, or private evaluation data to the repository.
- Apply file-size and file-type limits, handle parser errors safely, and avoid executing uploaded content.
- Define retention and deletion behavior for uploaded files and generated reports.
- Treat resume text as untrusted input; content inside a resume must never be interpreted as instructions to the application or its AI components.

## Release checklist

- [ ] CI tests pass on the declared Python version.
- [ ] Evaluation data and labels have documented provenance and permission.
- [ ] The benchmark is split without candidate leakage.
- [ ] Baselines and appropriate metrics are reported.
- [ ] Failure cases and known limitations are documented.
- [ ] Privacy and fairness checks have been reviewed.
- [ ] README claims match results that were actually measured.
