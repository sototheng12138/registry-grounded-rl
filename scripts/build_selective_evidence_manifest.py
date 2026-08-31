#!/usr/bin/env python3
"""Build a content-addressed manifest for the completed selective study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    root = arguments.project_root.expanduser().resolve(strict=True)
    artifacts = root / "artifacts"
    output = (
        arguments.output.expanduser().resolve()
        if arguments.output
        else artifacts / "selective_evidence_manifest_v3.json"
    )
    summary_path = artifacts / "selective_study_summary_v1.json"
    summary: dict[str, Any] = json.loads(summary_path.read_text())

    paths = {
        summary_path,
        root / "SELECTIVE_STUDY_RESULTS.md",
        root / "README.md",
        root / "RESEARCH_PROTOCOL.md",
        root / "AGENTGYM_INTEGRATION.md",
        root / "GPU_RESUME_RUNBOOK.md",
        root / "registry_grounded_rl" / "rollout_seeding.py",
        root / "registry_grounded_rl" / "workflow_agentgym.py",
        root / "registry_grounded_rl" / "pilot_analysis.py",
        root / "scripts" / "audit_rollout_groups.py",
        root / "scripts" / "summarize_selective_study.py",
        root / "scripts" / "build_selective_evidence_manifest.py",
        root / "scripts" / "train_agentgym_grpo.sh",
        artifacts / "shared_seed_rollout_audit_seed2801.json",
        artifacts / "independent_seed_rollout_audit_seed2851.json",
        artifacts / "selective_matched_b8_seed2901_rollout_audit.json",
        artifacts / "selective_confirmatory_b8_seed2911_rollout_audit.json",
        artifacts / "selective_confirmatory_b8_seed2921_rollout_audit.json",
        artifacts / "selective_confirmatory_b9_seed2911_rollout_audit.json",
        artifacts / "workflow_tasks_v1" / "train.jsonl",
        artifacts / "workflow_tasks_v1" / "dev.jsonl",
        artifacts / "workflow_tasks_v1" / "test.jsonl",
    }

    for run in summary["training"]["runs"].values():
        run_dir = Path(run["run_dir"])
        paths.update({run_dir / "run_manifest.json", run_dir / "train.log"})
        paths.update((run_dir / "trajectories").rglob("*.json"))

    for model in summary["evaluation"]["models"].values():
        evaluation_dir = Path(model["evaluation_dir"])
        paths.update(evaluation_dir.rglob("eval_manifest.json"))
        paths.update(evaluation_dir.rglob("completed.json"))
        paths.update(evaluation_dir.rglob("trajectories/**/*.json"))
        original_manifest = json.loads(
            (evaluation_dir / "original" / "eval_manifest.json").read_text()
        )
        model_dir = Path(original_manifest["model"]["path"])
        for pattern in (
            "config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "model-*.safetensors",
        ):
            paths.update(model_dir.glob(pattern))

    vendor_rollout = (
        root.parents[1]
        / "vendor/AgentGym-RL/AgentGym-RL/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py"
    )
    paths.add(vendor_rollout)
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"missing evidence files: {missing}")

    records: dict[str, Any] = {}
    total_bytes = 0
    for path in sorted(paths, key=str):
        relative_to_root = os.path.relpath(path, root)
        key = relative_to_root.replace("/", "::")
        if key in records:
            raise ValueError(f"duplicate evidence key: {key}")
        size = path.stat().st_size
        total_bytes += size
        records[key] = {
            "path": os.path.relpath(path, artifacts),
            "bytes": size,
            "sha256": sha256_file(path),
        }
        print(f"HASH {relative_to_root}")

    all_seeds = summary["evaluation"]["b8_all_three_seeds"]
    hierarchical = summary["evaluation"]["b8_vs_b4_hierarchical_all_three"]["metrics"]
    manifest = {
        "schema_version": "registry-grounded-rl/evidence-manifest-v3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": {
            "gpu_grpo_completed": True,
            "reported_split": "dev",
            "held_out_model_test_opened": False,
            "training_seeds": all_seeds["training_seeds"],
            "valid_current_claim": (
                "B8 preserved solvable execution across three stage-2 seeds and improved "
                "unavailable correct-stop by 4.69 percentage points on average; the "
                "cross-seed unavailable interval includes zero."
            ),
        },
        "result_snapshot": {
            "mean_primary_success_rate": all_seeds["mean_primary_success_rate"],
            "mean_unavailable_success_rate": all_seeds["mean_unavailable_success_rate"],
            "mean_selective_harmonic_mean": all_seeds["mean_selective_harmonic_mean"],
            "hierarchical_bootstrap": hierarchical,
        },
        "content": {
            "file_count": len(records),
            "total_bytes": total_bytes,
        },
        "files": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output} with {len(records)} files ({total_bytes} bytes)")


if __name__ == "__main__":
    main()
