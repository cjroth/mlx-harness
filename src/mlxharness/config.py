from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
DEFAULT_SANDBOX = "docker"


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    sandbox: str = DEFAULT_SANDBOX
    workspace: Path = field(default_factory=Path.cwd)
    hf_token: str | None = None


def resolve_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()

    return None


def parse_args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        prog="mlxharness",
        description="Local AI coding agent powered by MLX",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--sandbox",
        choices=["docker", "none"],
        default=DEFAULT_SANDBOX,
        help=f"Sandbox mode (default: {DEFAULT_SANDBOX})",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace path (mounted into sandbox, default: cwd)",
    )

    args = parser.parse_args(argv)

    return Config(
        model=args.model,
        sandbox=args.sandbox,
        workspace=args.workspace.resolve(),
        hf_token=resolve_hf_token(),
    )
