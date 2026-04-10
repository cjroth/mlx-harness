# mlxharness

What happens when on-device inference and its harness are one process? This projects explores it!

## Setup

```bash
uv venv && uv pip install -e .
```

Model (~8GB) downloads automatically on first run.

## Usage

```bash
mlxharness                              # Docker sandbox (default)
mlxharness --sandbox=none               # No sandbox, runs on host
mlxharness --model mlx-community/...    # Custom model
mlxharness --workspace ~/my-project     # Custom workspace
```

## Testing

```bash
.venv/bin/pytest tests/unit/              # Fast, no model/Docker
.venv/bin/pytest tests/e2e/ --e2e         # Full pipeline, needs both
```

## Architecture

`engine` (MLX inference) / `agent` (conversation loop, tool parsing) / `executor` (Docker or subprocess) / `tui` (ANSI terminal rendering). Agent yields events, TUI renders them. See [SPEC.md](SPEC.md) for details.

Default model: [Gemma 4 E4B 4-bit](https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit) — 4B active params, 128K context, native tool calling, ~25 tok/s on M2.
