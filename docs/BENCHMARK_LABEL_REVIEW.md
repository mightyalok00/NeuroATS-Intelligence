# Benchmark Relevance Label Review Protocol

## Current status

**Status: pending independent review.** The benchmark's 24 query-to-document relevance sets are provisional labels authored for a synthetic fixture. No independent reviewer has yet confirmed them. The generated evaluation report repeats this limitation deliberately.

## What is being labelled?

For each query, a document is labelled relevant if its snippet contains evidence that directly addresses the query's stated skill/task intent. Lexical overlap alone is not sufficient. Labels are binary for this benchmark, so relevance means "clearly relevant to the query intent," not "a good candidate" or "qualified for a real job."

## Review procedure

1. Freeze the query text, document text, and current relevance labels in a reviewed version.
2. Ask two reviewers to label each query-document pair independently, without showing them the current labels or retrieval method rankings.
3. Use a three-way label in the review worksheet: relevant, not_relevant, or uncertain. Reviewers should record a short rationale for each relevant and uncertain label.
4. Calculate raw agreement and Cohen's kappa where appropriate. Report the sample count and note that agreement statistics can be unstable on small or imbalanced samples.
5. Send disagreements and uncertain cases to a third adjudicator or a documented consensus discussion. Record the final decision and rationale.
6. Update the benchmark labels only after adjudication. Preserve a dated record of label changes and rerun the report.
7. Keep reviewers blind to system outputs until labels are finalized to reduce confirmation bias.

## Review worksheet

Create one row for every query-document pair (24 queries × 36 documents = 864 rows) with these columns:

- query_id
- document_id
- query_text
- document_text
- reviewer_1_label
- reviewer_1_rationale
- reviewer_2_label
- reviewer_2_rationale
- agreement
- adjudicated_label
- adjudication_rationale

The current benchmark's positive labels can be exported as a starting reference for the adjudicator only after blind independent annotation is complete. Do not share the reference labels with reviewers before they submit their initial labels.

## Acceptance checklist

- [ ] Two reviewers completed independent, blinded annotations.
- [ ] Uncertain cases and disagreements were adjudicated.
- [ ] Agreement statistic and limitations are documented.
- [ ] Final labels and review date are versioned.
- [ ] Generated report identifies the reviewed label version.
- [ ] No claim of real-world hiring validity is made from this synthetic fixture.

## Important boundary

Independent label review improves the reliability of this synthetic retrieval test, but it does not turn synthetic snippets into a real-world resume/job benchmark. Real-world validation still requires appropriately licensed or consented data, a representative sample, a job-related relevance rubric, candidate-level leakage controls, and a frozen test set.
