#!/usr/bin/env python3
"""Prepare the JSON-array layout required by AgentGym-RL generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    index = args.index.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing eval data: {output}")

    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("Evaluation index is empty")
    item_ids = [row.get("item_id") for row in rows]
    if len(set(item_ids)) != len(item_ids) or not all(
        isinstance(item_id, str) for item_id in item_ids
    ):
        raise ValueError("item_id values must be unique strings")

    output.mkdir(parents=True)
    test_path = output / "registryworkflow_test.json"
    category_path = output / "all.json"
    payload = json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    test_path.write_text(payload, encoding="utf-8")
    category_path.write_text(payload, encoding="utf-8")

    manifest = {
        "schema_version": "registry-grounded-rl/agentgym-eval-data-v1",
        "source_index": str(index),
        "source_sha256": sha256(index),
        "rows": len(rows),
        "files": {
            test_path.name: sha256(test_path),
            category_path.name: sha256(category_path),
        },
    }
    # AgentGym-RL treats every JSON file not starting with
    # ``registryworkflow_test`` as a category file.  Keep the manifest under
    # that ignored prefix so the upstream evaluator does not parse it as rows.
    (output / "registryworkflow_test.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
