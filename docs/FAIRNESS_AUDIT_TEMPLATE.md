# Fairness and Error Audit Worksheet

## Status

**No fairness result is asserted by this template.** The repository has not established a lawful, representative, sufficiently powered dataset for subgroup evaluation. Complete this worksheet only after data governance and legal review.

## Guardrails

- Do not infer protected or sensitive traits from names, photos, addresses, resumes, or other proxies.
- Do not use protected attributes to rank candidates or make hiring recommendations.
- Only analyze subgroup data where collection/use is lawful, appropriate, purpose-limited, access-controlled, and reviewed by the data owner.
- Do not publish small cells or outputs that could identify individuals. Predefine minimum cell-size and suppression rules.
- Treat differences as investigation signals, not proof of discrimination or proof of fairness.
- Keep a human reviewer in the loop and provide a process to correct parsing errors or challenge extracted facts.

## Audit plan

| Audit dimension | Example failure to inspect | Evidence to record |
|---|---|---|
| Resume format | Columns, tables, headers, scanned PDFs parse incorrectly | Format counts, parse failure rate, manual sample review |
| Job family | Vocabulary differs by role family | Per-family ranking metrics and sample counts |
| Career stage | Nontraditional or career-change resumes are poorly matched | Job-related rubric, false-positive/negative examples |
| Language variety | Tokenization or terminology mismatches | Language coverage and per-language error analysis |
| Missing information | Missing dates or sections cause unsupported score penalties | Feature-level error and explanation review |
| Accessibility | Formatting or assistive-technology export changes extraction | Supported formats and reproducible test fixtures |
| Human override | Reviewer cannot correct an extracted fact | Override availability and correction audit trail |

## Results worksheet

Fill only from observed evidence:

- Audit date:
- Reviewer(s) and role:
- Dataset/version and permitted-use approval reference:
- Sample size and exclusions:
- Dimensions evaluated:
- Metrics and uncertainty intervals:
- Small-cell suppression policy:
- Key observed errors (with identifying information removed):
- Mitigations tested and follow-up results:
- Limitations and unresolved risks:
- Decision owner and next review date:

## Interpretation

Do not claim "bias-free" or "fair" based on a single metric, small sample, synthetic data, or absence of statistically significant differences. Report what was tested, what was not tested, and where evidence is insufficient. Do not make consequential hiring decisions solely from automated scores.
