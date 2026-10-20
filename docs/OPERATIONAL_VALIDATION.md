# Operational Validation Checklist

## Status

This is a validation plan, not evidence that every production control is implemented. Verify each item against the running application before marking it complete.

## Input and parser robustness

- [ ] Empty or whitespace-only resume and job-description inputs.
- [ ] Unsupported file extension, incorrect MIME type, and extension/content mismatch.
- [ ] Zero-byte, corrupt, encrypted, and truncated PDF/DOCX files.
- [ ] Scanned PDF or image-only document with no extractable text.
- [ ] Oversized file, excessive page count, long text, and malformed archive contents.
- [ ] Multiple columns, tables, unusual encodings, Unicode, and non-English text.
- [ ] Parser exception produces a safe, actionable message without a traceback or data leak.
- [ ] Uploaded content is parsed as untrusted data; embedded instructions are never treated as application instructions.
- [ ] Time and memory use are bounded for untrusted files.

## Explainability and workflow

- [ ] Every score has a documented definition, scale, limitations, and relevant input evidence.
- [ ] Missing or unparsed evidence is distinguishable from a confirmed skill gap.
- [ ] Suggested resume text never invents credentials, employment history, or quantified outcomes.
- [ ] User can inspect and correct extracted facts before relying on the report.
- [ ] Candidate comparisons show relevant evidence and do not present scores as hiring probabilities.
- [ ] Errors, warnings, and model/version metadata are visible to the operator.

## Privacy and security

- [ ] Confirm whether any network calls or model downloads occur and disclose them.
- [ ] Confirm whether uploads, extracted text, generated reports, or caches persist; document retention and deletion.
- [ ] Keep personal data, private evaluation data, API keys, and access tokens out of Git, logs, and CI artifacts.
- [ ] Apply least-privilege access and safe temporary-file handling.
- [ ] Test that errors do not expose document contents, file paths, secrets, or stack traces to other users.
- [ ] Document the process for deletion requests and security incident handling.

## Evidence log

For each test, record: test ID, software commit, environment, input fixture category (never real personal data), expected behavior, observed behavior, pass/fail, issue link, and reviewer. Use synthetic or intentionally crafted fixtures for CI. Do not mark a control complete from a checklist alone.
