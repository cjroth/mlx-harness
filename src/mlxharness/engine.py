from __future__ import annotations

from pathlib import Path
from typing import Iterator

import mlx.core as mx
from huggingface_hub import try_to_load_from_cache
from mlx_lm import load, stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper


SYSTEM_PROMPT = (
    "You are a coding agent. You can execute shell commands to accomplish tasks. "
    "Run one command at a time. After receiving a result, you may run more commands "
    "or respond to the user. When you are done, respond with plain text (no tool call)."
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Execute a shell command and return its output",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                }
            },
            "required": ["command"],
        },
    },
}]


def _resolve_local_path(model_name: str) -> str:
    """Return local cache path if model is already downloaded, otherwise the repo name."""
    if Path(model_name).is_dir():
        return model_name
    cached = try_to_load_from_cache(model_name, "config.json")
    if cached and isinstance(cached, str):
        return str(Path(cached).parent)
    return model_name


class Engine:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model, self.tokenizer = load(_resolve_local_path(model_name))

    def count_tokens(self, messages: list[dict], tools: list[dict] | None = None) -> int:
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if tools:
            kwargs["tools"] = tools
        text = self.tokenizer.apply_chat_template(messages, **kwargs)
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    @property
    def context_window(self) -> int:
        args = self.model.args
        # Check top-level args first, then nested text_config (Gemma 4 style)
        for source in (args, getattr(args, "text_config", None)):
            if source is None:
                continue
            if isinstance(source, dict):
                for key in ("max_position_embeddings", "max_seq_len", "seq_length"):
                    if key in source:
                        return source[key]
            else:
                for key in ("max_position_embeddings", "max_seq_len", "seq_length"):
                    val = getattr(source, key, None)
                    if val is not None:
                        return val
        return 8192

    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> Iterator[str]:
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if tools:
            kwargs["tools"] = tools
        prompt = self.tokenizer.apply_chat_template(messages, **kwargs)

        def greedy_sampler(logits: mx.array) -> mx.array:
            return mx.argmax(logits, axis=-1)

        for response in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=4096,
            sampler=greedy_sampler,
        ):
            yield response.text
