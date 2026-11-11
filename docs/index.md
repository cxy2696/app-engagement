# Documentation

The documentation is grouped by the question you are trying to answer. Every
path below is stable; the reference numbers live in the generated
[evaluation report](../reports/evaluation-report.md).

## Start here

| Document | What it answers |
| --- | --- |
| [README](../README.md) | What the project is, install, first run |
| [Showcase](showcase.md) | Screens, links, reference result |
| [Architecture](architecture.md) | How the package, artefact, API, and dashboard fit together |

## Protocol and evidence

For researchers and reviewers.

| Document | What it answers |
| --- | --- |
| [Methodology](methodology.md) | Task, estimators, leakage controls, statistical analysis |
| [Evaluation](evaluation.md) | Commands, outputs, how to read the metrics |
| [Algorithm notes](algorithm.md) | Scoring, percentile threshold, decision policy, complexity |
| [Data card](data-card.md) | Provenance, schema, quality, privacy, licence |
| [Model card](model-card.md) | Intended use, metrics, limitations |
| [Technical report](../paper/engagegraph-report.pdf) | Six-page PDF with the full protocol and results |
| [Legacy audit](../reports/legacy-audit.md) | Why the original 0.9257 AUC is not comparable |

## Build and integrate

For engineers adding a domain or deploying the service.

| Document | What it answers |
| --- | --- |
| [Generality](generality.md) | Data contract and domain profiles |
| [API](api.md) | Endpoints, examples, error semantics |
| [Deployment](deployment.md) | Local, Docker, reverse proxy, configuration |
| [Profiles](../profiles/README.md) | Ready-to-copy domain configurations |

## Operate and maintain

For the person on call or maintaining a release.

| Document | What it answers |
| --- | --- |
| [Operations](operations.md) | SLOs, monitoring, retraining, incident runbook |
| [Reproducibility](reproducibility.md) | Seeds, fingerprints, CI, verification, non-reproducibility |
| [Governance](governance.md) | Change control, releases, data ethics |
| [Contributing](../CONTRIBUTING.md) | Development setup and review checklist |
| [Security](../SECURITY.md) | Private vulnerability reporting |

## Reference

- [Glossary](glossary.md) — terms and symbols used across documents.
- [FAQ](faq.md) — short answers to common questions.
- [Roadmap](../ROADMAP.md) and [Changelog](../CHANGELOG.md).

## Reading paths

- **New researcher:** README → Methodology → Evaluation → Data card → Technical report.
- **Engineer adapting a domain:** README → Architecture → Generality → API → Deployment.
- **Operator:** Deployment → Operations → Reproducibility → Governance.

中文概览见 [README.zh-CN.md](../README.zh-CN.md)。
