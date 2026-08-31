"""Deterministic, independent random seeds for repeated GRPO rollouts."""

from __future__ import annotations

import hashlib


def independent_rollout_seed(
    *,
    base_seed: int,
    global_step: int,
    round_index: int,
    rank: int,
    rollout_index: int,
) -> int:
    """Derive one stable vLLM seed without correlating repeated prompts.

    A single ``SamplingParams.seed`` is reused for every prompt by the pinned
    AgentGym-RL rollout worker. Repeated GRPO prompts can therefore receive
    identical samples. Hashing the full rollout coordinate keeps reruns
    deterministic while assigning each active trajectory its own RNG stream.
    """

    coordinates = (base_seed, global_step, round_index, rank, rollout_index)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in coordinates):
        raise TypeError("rollout seed coordinates must be integers")
    if any(value < 0 for value in coordinates):
        raise ValueError("rollout seed coordinates must be non-negative")
    payload = ":".join(str(value) for value in coordinates).encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"rg-grpo").digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)
