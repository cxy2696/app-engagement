# Contributing

Thanks for improving EngageGraph. This project values reproducible results, clear failure modes, and honest evaluation over impressive-looking numbers.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,als,plots]"
pip install -e ".[graph]"   # optional, for MetaPath2Vec
pre-commit install
```

## Before opening a PR

```bash
make check          # lint, format check, type check, tests
make verify-docs    # links and required files (fast)
```

For changes that touch the protocol, models, or metrics:

```bash
make data
make benchmark
make report
```

Attach the resulting `benchmark.json` and report to the PR description, and state whether any test-set decision was made. If you tuned on February, say so explicitly; the maintainer may ask for a January-only validation protocol instead.

## Style

- Python 3.10+, type hints on public functions.
- Ruff for lint and formatting (line length 100).
- Tests use synthetic fixtures; never commit panel data derivatives.
- Docstrings explain *why* a choice was made when it is not obvious.
- No pickle, no secrets, no raw panelist IDs in logs or tests.

## Commit messages

Use the imperative mood and a scope when helpful:

```text
feat(api): add batch recommendation endpoint
fix(metrics): mask seen items before top-k selection
docs(data): document inferred millisecond unit
```

## Data contributions

New data requires a provenance note, licence confirmation, hashes, a data-card update, and a privacy review. Do not upload row-level panel data to issues or PRs.

## Review checklist

- [ ] Tests cover the change, including a failure case.
- [ ] Public API docs updated.
- [ ] No leakage introduced (January fit, February labels only).
- [ ] Artefact schema remains backwards compatible or the version is bumped.
- [ ] `CHANGELOG.md` updated for user-visible changes.
- [ ] Security/privacy implications considered.

## Code of conduct

By participating, you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
