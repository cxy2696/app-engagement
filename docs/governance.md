# Governance

## 1. Scope

This repository is a research-grade reference implementation maintained as an open-source project. It is not a hosted service and does not come with a warranty. The governance below keeps changes reviewable and the benchmark comparable over time.

## 2. Roles

| Role | Responsibility |
| --- | --- |
| Maintainer | Release approval, security response, data-licence decisions |
| Contributor | Code, docs, tests, or data-quality reports under the CLA/DCO |
| Reviewer | At least one maintainer for changes to `data/`, `evaluation/`, or `artifacts.py` |
| Data steward | Confirms provenance and permitted use before new data is added |

The current maintainer is Xinyi Cui ([@cxy2696](https://github.com/cxy2696)).

## 3. Change control

- **Protocol changes** (catalogue rules, eligibility, metrics, splits) require a design note in `docs/` and a version bump of the benchmark schema. They must not be slipped into a model PR.
- **Model changes** require a benchmark run and a comparison table with confidence intervals. A new model is merged only if it is reproducible and its artefact loads.
- **API changes** follow semantic versioning. Additive fields are minor; removing or renaming fields is major and requires a deprecation period.
- **Data changes** require provenance, a data card update, hashes, and a licence review.

## 4. Release process

1. Update `CHANGELOG.md` and the version in `pyproject.toml` / `__init__.py`.
2. Run `make check` (lint, tests, build) and the benchmark.
3. Generate `reports/` and `paper/` PDF.
4. Tag `vX.Y.Z`; attach the wheel, report PDF, and benchmark artefacts to the GitHub release.
5. Publish release notes with metric changes and known limitations.

Cadence: patch releases as needed; a minor release roughly quarterly; a major release only for breaking protocol/API changes.

## 5. Deprecation

- Announce in `CHANGELOG.md` and the release notes.
- Keep the old path working for at least one minor release.
- Remove only in a major release with a migration note.

## 6. Data ethics review

Before adding a new dataset:

- Confirm the legal basis and redistribution rights.
- Document collection, consent, and retention.
- Remove or hash direct identifiers.
- Assess re-identification risk and sensitive-attribute inference.
- State intended use and prohibited use in the data card.
- Provide a contact for takedown/correction requests.

The current exports have unresolved provenance; they are accepted for research reproducibility only, with the restrictions in [`DATA_LICENSE.md`](../DATA_LICENSE.md).

## 7. Security

Report vulnerabilities privately per [`SECURITY.md`](../SECURITY.md). Do not open a public issue for an unpatched vulnerability. The maintainer aims to acknowledge within 5 business days and to ship a fix or mitigation within 30 days for critical issues.

## 8. Community

- Be civil; follow the [Code of Conduct](../CODE_OF_CONDUCT.md).
- Use issues for bugs and proposals; use discussions for open-ended questions if enabled.
- PRs should be small, tested, and documented. See [`CONTRIBUTING.md`](../CONTRIBUTING.md).

## 9. Archival and succession

If the maintainer becomes unavailable, the repository should remain buildable from a tagged release. A successor maintainer should be added to `CODEOWNERS` and the release process documented here. The benchmark schema and artefacts are designed to outlive any single model.
