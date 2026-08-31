#!/usr/bin/env python3
"""Export a one-rank VERL FSDP actor checkpoint to Hugging Face format."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.distributed._tensor import DTensor
from transformers import AutoConfig, AutoModelForCausalLM


def full_single_rank_state(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Normalize NO_SHARD tensors or one-rank DTensors to CPU bfloat16 tensors."""

    normalized: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if isinstance(value, DTensor):
            tensor = value.full_tensor()
        elif isinstance(value, torch.Tensor):
            tensor = value
        else:
            raise TypeError(f"unsupported checkpoint value for {name}: {type(value)!r}")
        normalized[name] = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-dir", type=Path, required=True)
    args = parser.parse_args()
    actor_dir = args.actor_dir.expanduser().resolve(strict=True)
    checkpoint = actor_dir / "model_world_size_1_rank_0.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"one-rank model checkpoint is absent: {checkpoint}")
    hf_dir = actor_dir / "huggingface"
    config_path = hf_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint metadata is absent: {config_path}")
    if list(hf_dir.glob("*.safetensors")):
        raise FileExistsError(f"refusing to overwrite existing model weights in {hf_dir}")

    raw_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = full_single_rank_state(raw_state)
    config = AutoConfig.from_pretrained(hf_dir, local_files_only=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device="cpu")
    model.save_pretrained(
        hf_dir,
        state_dict=state,
        max_shard_size="5GB",
        safe_serialization=True,
    )
    print(f"exported {len(state)} tensors to {hf_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
