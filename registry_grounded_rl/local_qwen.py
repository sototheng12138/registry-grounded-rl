"""Minimal native Qwen multi-turn runner for frozen-policy smoke evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping, Protocol, Sequence

from .environment import RegistryGroundedEnv


class ExecutableRegistryEnv(Protocol):
    max_steps: int

    def reset(self) -> dict[str, Any]: ...

    def step(self, action: Mapping[str, Any]) -> Any: ...

    def trajectory(self) -> dict[str, Any]: ...


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FINAL_RE = re.compile(r"<final>\s*(.*?)\s*</final>", re.DOTALL)
_UNAVAILABLE_RE = re.compile(r"<unavailable>\s*(.*?)\s*</unavailable>", re.DOTALL)


@dataclass(frozen=True, slots=True)
class ParsedAction:
    action: dict[str, Any] | None
    parse_error: str | None


def _validate_action_payload(value: Mapping[str, Any], action_type: str) -> ParsedAction:
    if action_type == "tool_call":
        name = value.get("name")
        arguments = value.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, Mapping):
            return ParsedAction(None, "invalid_tool_call_payload")
        return ParsedAction(
            {"type": "tool_call", "name": name, "arguments": dict(arguments)}, None
        )
    if action_type == "final":
        answer = value.get("answer")
        if isinstance(answer, bool) or not isinstance(answer, (int, str)):
            return ParsedAction(None, "final_answer_not_scalar")
        if isinstance(answer, str) and not answer.strip():
            return ParsedAction(None, "final_answer_empty")
        return ParsedAction({"type": "final", "answer": answer}, None)
    if action_type == "unavailable":
        reason = value.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            return ParsedAction(None, "unavailable_reason_missing")
        return ParsedAction({"type": "unavailable", "reason": reason.strip()}, None)
    return ParsedAction(None, f"unknown_action_type:{action_type}")


def parse_model_action(text: str) -> ParsedAction:
    tool_blocks = _TOOL_CALL_RE.findall(text)
    final_blocks = _FINAL_RE.findall(text)
    unavailable_blocks = _UNAVAILABLE_RE.findall(text)
    block_count = len(tool_blocks) + len(final_blocks) + len(unavailable_blocks)
    if block_count != 1:
        return ParsedAction(None, f"expected_one_action_block:found={block_count}")
    block = (tool_blocks or final_blocks or unavailable_blocks)[0]
    try:
        value = json.loads(block)
    except json.JSONDecodeError as exc:
        return ParsedAction(None, f"invalid_json:{exc.msg}@{exc.pos}")
    if not isinstance(value, Mapping):
        return ParsedAction(None, "action_payload_not_object")
    if tool_blocks:
        return _validate_action_payload(value, "tool_call")
    if final_blocks:
        return _validate_action_payload(value, "final")
    return _validate_action_payload(value, "unavailable")


def parse_agentgym_action(text: str) -> ParsedAction:
    """Parse one wrapped action or one whole-response JSON action.

    AgentGym exposes a JSON action contract in every observation.  Qwen2.5 often
    follows that contract directly instead of adding the optional XML wrapper.
    The fallback remains strict: the complete response must be exactly one JSON
    object, so prose plus JSON and concatenated actions are rejected.
    """

    wrapped = parse_model_action(text)
    if wrapped.action is not None:
        return wrapped
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return wrapped
    if not isinstance(value, Mapping):
        return ParsedAction(None, "action_payload_not_object")
    action_type = value.get("type")
    if action_type is None:
        if isinstance(value.get("name"), str) and isinstance(value.get("arguments"), Mapping):
            action_type = "tool_call"
        elif "answer" in value:
            action_type = "final"
        elif "reason" in value:
            action_type = "unavailable"
    if not isinstance(action_type, str):
        return ParsedAction(None, "action_type_missing")
    return _validate_action_payload(value, action_type)


class LocalQwenAgent:
    """One resident Transformers model using Qwen's native tool chat template."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 160,
        enable_thinking: bool = False,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve(strict=True)
        self.requested_device = device
        self.requested_dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.torch: Any | None = None
        self.device: Any | None = None
        self.dtype: Any | None = None

    def ensure_loaded(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        requested = self.requested_device
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if str(requested).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        device = torch.device(requested)
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if self.requested_dtype == "auto":
            dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
                torch.float16 if device.type == "cuda" else torch.float32
            )
        else:
            dtype = dtype_map.get(self.requested_dtype)
            if dtype is None:
                raise ValueError(f"Unsupported dtype: {self.requested_dtype}")
        if device.type == "cpu" and dtype is torch.float16:
            raise ValueError("float16 CPU inference is unsupported")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        self.torch = torch
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.dtype = dtype

    def _generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        self.ensure_loaded()
        assert self.torch is not None and self.tokenizer is not None and self.model is not None
        assert self.device is not None
        kwargs = {
            "tools": list(tools),
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            prompt = self.tokenizer.apply_chat_template(
                list(messages), enable_thinking=self.enable_thinking, **kwargs
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(list(messages), **kwargs)
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            sequences = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = sequences[0, input_ids.shape[-1] :]
        raw = self.tokenizer.decode(
            generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        return {
            "raw_output": raw,
            "input_tokens": int(input_ids.shape[-1]),
            "output_tokens": int(generated.shape[-1]),
            "latency_seconds": time.perf_counter() - started,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        }

    @staticmethod
    def _assistant_message(action: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": action["name"],
                        "arguments": action["arguments"],
                    },
                }
            ],
        }

    def _run_env(self, env: ExecutableRegistryEnv, *, system_prompt: str) -> dict[str, Any]:
        observation = env.reset()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": observation["request"]},
        ]
        tools = observation["tools"]
        generations: list[dict[str, Any]] = []
        for turn in range(env.max_steps):
            generation = self._generate(messages, tools)
            parsed = parse_model_action(generation["raw_output"])
            generations.append(generation | {"turn": turn, "parse_error": parsed.parse_error})
            if parsed.action is None:
                result = env.step({"type": "invalid_model_output", "raw": generation["raw_output"]})
                messages.append({"role": "assistant", "content": generation["raw_output"]})
                messages.append(
                    {
                        "role": "user",
                        "content": json.dumps(result.observation, ensure_ascii=False, sort_keys=True),
                    }
                )
            else:
                result = env.step(parsed.action)
                if parsed.action["type"] == "tool_call":
                    messages.append(self._assistant_message(parsed.action))
                    messages.append(
                        {
                            "role": "tool",
                            "content": json.dumps(
                                result.observation, ensure_ascii=False, sort_keys=True
                            ),
                        }
                    )
                else:
                    messages.append({"role": "assistant", "content": generation["raw_output"]})
            if result.terminated or result.truncated:
                break
        trajectory = env.trajectory()
        trajectory["model"] = {
            "path": str(self.model_path),
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", "") if self.dtype is not None else None,
            "max_new_tokens": self.max_new_tokens,
            "enable_thinking": self.enable_thinking,
        }
        trajectory["generations"] = generations
        return trajectory

    def run(self, env: RegistryGroundedEnv) -> dict[str, Any]:
        return self._run_env(
            env,
            system_prompt=(
                "You are an executable tool agent. Follow the requested operation order. "
                "Call exactly one listed tool per turn. Never invent a tool or argument. "
                "After all required calls, return <final>{\"answer\": INTEGER}</final>. "
                "If a required semantic operation has no listed tool, make no tool calls and "
                "return <unavailable>{\"reason\": \"...\"}</unavailable>."
            ),
        )

    def run_workflow(self, env: ExecutableRegistryEnv) -> dict[str, Any]:
        return self._run_env(
            env,
            system_prompt=(
                "You are a stateful tool agent. Follow the current tool descriptions and JSON "
                "schemas even when names or argument keys change. First discover live project "
                "and ticket IDs, then persist every requested update. Preview tools do not change "
                "state. Emit exactly one tool call per turn. When the exact workspace state is "
                "complete, return <final>{\"answer\":\"done\"}</final>. If a required mutation "
                "capability is absent, make no mutations and return "
                "<unavailable>{\"reason\":\"...\"}</unavailable>."
            ),
        )
