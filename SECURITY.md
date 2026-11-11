# Security policy

## Supported versions

| Version | Supported |
| --- | --- |
| 2.x | Yes |
| 1.x | No (research snapshot) |

## Reporting a vulnerability

Please **do not** open a public issue for an unpatched vulnerability. Report privately via GitHub's [private vulnerability reporting](https://github.com/cxy2696/app-engagement/security/advisories/new) or email the maintainer listed in [`CITATION.cff`](CITATION.cff).

Include:

- affected version/commit;
- a minimal reproduction;
- impact assessment (data exposure, code execution, denial of service);
- any suggested mitigation.

You can expect an acknowledgement within 5 business days and a fix or mitigation within 30 days for critical issues. Credit will be given unless you prefer to remain anonymous.

## Scope

In scope: the Python package, API, artefact loader, dashboard, Docker image, and CI configuration. Out of scope: the panel data's provenance/licensing (see [`DATA_LICENSE.md`](DATA_LICENSE.md)), third-party dependencies, and deployments operated by third parties.

## Data and privacy

The service does not require raw identifiers in logs. If you find a path that logs panelist IDs, credentials, or row-level data, treat it as a security issue and report it privately. Do not attempt re-identification of panelists; that is explicitly out of scope for this project and prohibited by the data notice.
