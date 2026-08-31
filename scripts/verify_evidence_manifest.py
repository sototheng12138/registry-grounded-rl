#!/usr/bin/env python3
"""Verify every content hash declared in an evidence manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts_root = manifest_path.parent
    failures: list[str] = []
    for name, item in sorted(value["files"].items()):
        path = artifacts_root / item["path"]
        if not path.is_file():
            failures.append(f"{name}: missing {path}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item["sha256"]:
            failures.append(f"{name}: expected {item['sha256']}, got {actual}")
        else:
            print(f"PASS {name} {actual}")
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print(f"VERIFIED {len(value['files'])} evidence files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
