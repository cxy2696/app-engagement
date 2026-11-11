#!/usr/bin/env python
"""Download the public research exports and verify their SHA-256 digests.

The exports are already tracked in the repository, so this script is only
needed for source installs or CI jobs that do not clone the data directory.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

FILES = {
    "jan_aggregated.parquet": (
        "https://raw.githubusercontent.com/cxy2696/app-engagement/main/"
        "Jan-Feb-parquet/jan_aggregated.parquet",
        "dd81a42edcf3539977f723e943be6f67ae536434cb1821077fc84c2f1a16d879",
    ),
    "feb_aggregated.parquet": (
        "https://raw.githubusercontent.com/cxy2696/app-engagement/main/"
        "Jan-Feb-parquet/feb_aggregated.parquet",
        "336f68edb607b7f0e88055d4ea346b0d6978ec2866683af62e83a0f05b157ee4",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, url: str, expected: str, out_dir: Path, force: bool) -> None:
    target = out_dir / name
    if target.exists() and not force:
        actual = sha256(target)
        if actual == expected:
            print(f"ok      {target} ({actual[:12]})")
            return
        raise SystemExit(
            f"hash mismatch for {target}: {actual} != {expected}; pass --force to replace"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"fetch   {url}")
    with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    actual = sha256(target)
    if actual != expected:
        target.unlink(missing_ok=True)
        raise SystemExit(f"downloaded hash mismatch: {actual} != {expected}")
    print(f"saved   {target} ({target.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="Jan-Feb-parquet")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    for name, (url, expected) in FILES.items():
        download(name, url, expected, out_dir, args.force)
    print("data ready; run `engagegraph data build --config configs/dataset.yaml`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
