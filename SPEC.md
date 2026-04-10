# harnessthing — Local AI Coding Agent

## Vision

A minimal, single-process AI coding agent powered by MLX on Apple Silicon. The LLM inference and agent harness run in the same Python process — no APIs, no IPC, no serialization. You type a prompt, the model thinks, it runs shell commands, and it responds.

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    Python Process (macOS, Metal)           │
│                                                            │
│  ┌──────────┐       ┌──────────────────────────────────┐  │
│  │   TUI    │       │          Agent Loop              │  │
│  │          │       │                                  │  │
│  │ - prompt │──────>│ - conversation history           │  │
│  │ - stream │<──────│ - tool call detection            │  │
│  │   tokens │  cb   │ - multi-turn loop (until stop)   │  │
│  │ - tool   │       │ - context window tracking        │  │
│  │   display│       │ - tool result truncation         │  │
│  └──────────┘       └───────┬──────────────┬──────────┘  │
│                             │              │              │
│              ┌──────────────▼──────┐ ┌─────▼───────────┐ │
│              │  Inference Engine   │ │ Command Executor │ │
│              │                    │ │                  │ │
│              │ - mlx-lm + Metal   │ │ - docker exec    │ │
│              │ - streaming gen    │ │ - subprocess     │ │
│              │ - logit access     │ │ - stdout/stderr  │ │
│              └────────────────────┘ └──────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

### Separation of Concerns

Four modules with clear boundaries. Each module knows nothing about the others' internals.

| Module | Responsibility | Depends on | Does NOT know about |
|--------|---------------|------------|-------------------|
| **`tui`** | User input, streaming display, formatting | Nothing | Model, agent loop, Docker |
| **`engine`** | Model loading, token generation | `mlx_lm` | Conversation structure, tools, TUI |
| **`agent`** | Conversation management, tool call parsing, orchestration | `engine`, `executor` interfaces | MLX internals, Docker commands, ANSI codes |
| **`executor`** | Shell command execution, output capture | `subprocess` / `docker` | Model, conversation, display |

Communication between modules:

- **TUI ↔ Agent**: Agent yields `Event` objects (token, tool_call, tool_result, error, done). TUI consumes and renders them. TUI never calls engine directly.
- **Agent → Engine**: Agent passes message list, receives a token iterator. Agent doesn't touch logits or MLX tensors.
- **Agent → Executor**: Agent passes a command string, receives `CommandResult(exit_code, stdout, stderr)`. Agent doesn't know if it's Docker or subprocess.

```python
# The event protocol between agent and TUI
@dataclass
class TokenEvent:
    text: str

@dataclass
class ToolCallEvent:
    command: str

@dataclass
class ToolResultEvent:
    result: CommandResult

@dataclass
class ThinkingEvent:
    text: str

@dataclass
class ErrorEvent:
    message: str

@dataclass
class DoneEvent:
    pass

Event = TokenEvent | ToolCallEvent | ToolResultEvent | ThinkingEvent | ErrorEvent | DoneEvent
```

## Model

- **Default**: Gemma 4 E4B, 4-bit quantized (`mlx-community/gemma-4-e4b-it-4bit`)
- **Why E4B**: ~4B active params, 128K context, native tool calling, fits comfortably in 24GB (~8GB VRAM), ~25 tok/s on M2
- **Multi-model**: Swap models via `--model`. JSON tool calls are supported as fallback for non-Gemma models.
- **Download**: `mlx-lm` auto-downloads from HuggingFace on first run. On subsequent runs, the engine resolves the local cache path directly to skip HuggingFace hub checks (no progress bars, instant startup).
- **HF Token**: Gemma 4 is Apache 2.0 — no token required. Other gated models: `HF_TOKEN` env var → `~/.cache/huggingface/token` fallback → `None` (will fail at download time if the model is gated).
- **Inference**: Uses greedy decoding via a `sampler` callable (`mx.argmax`). The `temp` parameter is not used — `mlx-lm` expects a sampler function.

## Tool: Shell Command Execution

One tool. The model can run any shell command.

### System Prompt

```
You are a coding agent. You can execute shell commands to accomplish tasks. Run one command at a time. After receiving a result, you may run more commands or respond to the user. When you are done, respond with plain text (no tool call).
```

The tool is declared via the chat template's `tools` parameter using standard OpenAI-compatible function schema, not described in the system prompt text:

```python
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
```

### Tool Call Detection

Gemma 4 uses native function calling with special tokens. The model emits tool calls in this format:

```
<|tool_call>call:shell{command:<|"|>ls -la<|"|>}<tool_call|>
```

Detection: parse the model's response for this native format via regex. Falls back to JSON extraction (`{"tool": "shell", "command": "..."}`) for compatibility with other models.

Tool results are sent back as `role: "tool"` messages (not `role: "user"`).

### Tool Result Truncation

Tool results (stdout + stderr) are truncated to **4096 characters** (keeping the last 4096 — tail, not head, because errors and final output are most relevant). The truncation is noted in the result:

```json
{"exit_code": 0, "stdout": "[truncated, showing last 4096 chars]\n...", "stderr": ""}
```

### Agent Loop

The agent runs tool calls in a loop until the model emits a plain text response (no tool call). No step limit — the model decides when it's done.

```python
# In agent.py — yields Events for TUI to render

def step(self, user_input: str) -> Iterator[Event]:
    self.messages.append({"role": "user", "content": user_input})

    while True:
        self._check_context_window()

        response_text = ""
        for token, event in self._stream_tokens(self.messages):
            response_text += token
            if event is not None:
                yield event

        tool_call = parse_tool_call(response_text)

        if tool_call is None:
            self.messages.append({"role": "assistant", "content": response_text})
            yield DoneEvent()
            break

        yield ToolCallEvent(command=tool_call["command"])

        result = self.executor.run(tool_call["command"])
        result = truncate_result(result, max_chars=4096)

        yield ToolResultEvent(result=result)

        self.messages.append({"role": "assistant", "content": response_text})
        self.messages.append({"role": "tool", "content": format_tool_result(result)})
```

### Context Window

Gemma 4 E4B has 128K context — large enough that exhaustion is unlikely in normal use, but still possible with large tool results.

**Strategy: hard error.** Before each generation call, count tokens in the message list (including tool declarations). If the next generation would leave less than 512 tokens of headroom, raise `ContextWindowExhaustedError` with current count vs limit. No silent truncation, no summarization.

Context window size is read from `model.args`, checking `text_config` dict for `max_position_embeddings` (Gemma 4 style), with fallback to top-level attributes and a default of 8192.

## Streaming

The engine exposes a **token iterator**. Each call to `engine.generate(messages, tools=TOOLS)` yields individual tokens as they're produced by MLX. The agent forwards these as events to the TUI, which prints them immediately.

Streaming interacts with tool call detection: the agent accumulates the full response text while streaming, then parses for tool calls after generation completes (on EOS/stop token).

### Special Token Handling

Gemma 4 emits special tokens during streaming that the agent filters:

- **Thinking blocks** (`<|channel>thought\n...<channel|>`): Content is yielded as `ThinkingEvent`s, rendered dimly in the TUI. The `<|channel>`, channel name, and `<channel|>` tokens are suppressed.
- **Tool call blocks** (`<|tool_call>...<tool_call|>`): All tokens suppressed from display. The parsed command is shown via `ToolCallEvent` instead.
- **Quote tokens** (`<|"|>`): Suppressed.

## Command Executor

Two modes, selected at startup via `--sandbox` flag:

### `--sandbox=docker` (default)

Commands execute inside a long-running Docker container via `docker exec`.

```python
subprocess.run(
    ["docker", "exec", "harnessthing-sandbox", "bash", "-c", command],
    capture_output=True, text=True, timeout=120
)
```

The sandbox container:
- Started automatically on first command, stopped on exit
- Mounts `--workspace` path to `/workspace` inside the container
- Has network access (bridge mode)
- Commands run as non-root `agent` user

### `--sandbox=none`

Commands execute directly on the host via `subprocess.run`. No isolation.

### Interface

Both executors implement the same interface:

```python
@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str

class Executor(Protocol):
    def run(self, command: str) -> CommandResult: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
```

## Docker Sandbox Image

### Dockerfile.sandbox

```dockerfile
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core
    bash \
    coreutils \
    git \
    curl \
    wget \
    ca-certificates \
    # Build tools
    build-essential \
    cmake \
    pkg-config \
    # Python
    python3 \
    python3-pip \
    python3-venv \
    # Node
    nodejs \
    npm \
    # Text processing
    jq \
    ripgrep \
    tree \
    less \
    # Networking
    openssh-client \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -s /bin/bash agent
USER agent
WORKDIR /workspace
```

**Why Ubuntu**: Models are trained on mountains of Ubuntu shell commands. Priors align with the environment.

**Why these tools**: git, python, node, build-essential cover the majority of coding tasks. curl/jq for APIs. ripgrep/tree for navigation.

## TUI

Minimal. No TUI framework — stdin/stdout with ANSI escape codes.

The TUI consumes `Event` objects from the agent and renders them. It never calls the engine or executor directly.

```
> user types here
I need to list the files...              ← dim, thinking
  $ ls -la /workspace                    ← dim, shows tool call
  total 8                                ← dim, shows stdout
  drwxr-xr-x 2 agent agent 4096 ...     ← dim
The files showed...                      ← dim, thinking again
Here are the files in your workspace:    ← normal, response text

> next prompt
```

- `> ` prompt with readline history
- Streaming token display
- Thinking: dim (streamed as model reasons, separated from response by newline)
- Tool calls: dim gray, prefixed with `$`
- Tool stdout: dim white
- Tool stderr: dim yellow
- Errors: red
- Ctrl+C interrupts generation
- `/quit` or Ctrl+D to exit

## Project Structure

```
harnessthing/
├── pyproject.toml
├── SPEC.md
├── Dockerfile.sandbox
├── src/
│   └── harnessthing/
│       ├── __init__.py
│       ├── __main__.py       # CLI entry point, arg parsing
│       ├── engine.py         # Model loading, streaming generation
│       ├── agent.py          # Agent loop, conversation, tool parsing, context tracking
│       ├── events.py         # Event dataclasses (shared protocol between agent and tui)
│       ├── tui.py            # Terminal I/O, prompt, streaming display, formatting
│       ├── executor.py       # Executor protocol, DockerExecutor, SubprocessExecutor
│       └── config.py         # CLI args, model config, HF token resolution
├── tests/
│   ├── conftest.py           # Shared fixtures, pytest config for --e2e flag
│   ├── unit/
│   │   ├── test_agent.py     # Tool call parsing, message building, context overflow
│   │   ├── test_executor.py  # Command construction, output capture, timeout
│   │   ├── test_config.py    # HF token resolution, arg defaults
│   │   └── test_tui.py       # Output formatting, event rendering
│   └── e2e/
│       ├── test_e2e.py       # Full pipeline: real model → tool call → shell → response
│       └── conftest.py       # Session fixtures: model loading, sandbox lifecycle
└── workspace/                # Default workspace (mounted into sandbox)
```

## Testing Strategy

### Unit Tests

Fast. No model, no Docker. `pytest tests/unit/`

| Test file | What it covers |
|-----------|---------------|
| `test_agent.py` | JSON tool call parsing (valid, malformed, embedded in prose, escaped quotes). Context window overflow detection. Message list construction. Tool result truncation logic. Multi-turn loop with mocked engine (emits tool call, then plain text). |
| `test_executor.py` | Docker exec command construction. Subprocess fallback. stdout/stderr separation. Non-zero exit codes. Timeout (120s default). |
| `test_config.py` | HF token from env var. HF token from file. Missing token error. Model name defaults. Workspace path resolution. |
| `test_tui.py` | Event rendering: TokenEvent prints text, ToolCallEvent prints dim `$` line, ToolResultEvent prints dim output, ErrorEvent prints red. |

**Mocking policy**: Mock at module boundaries only. `subprocess.run` in executor tests. Token iterator in agent tests. Never mock internal functions.

### End-to-End Tests

Slow. Requires model (~8GB download) and Docker. `pytest tests/e2e/ --e2e`

These test the full pipeline with a real model. No mocking. This is the only way to verify the model actually produces valid tool calls with our system prompt.

```python
# tests/e2e/test_e2e.py

class TestEndToEnd:

    def test_simple_response(self, agent):
        """Model responds without using tools."""
        events = list(agent.step("Say hello in exactly 3 words."))
        tokens = [e for e in events if isinstance(e, TokenEvent)]
        assert len(tokens) > 0
        assert any(isinstance(e, DoneEvent) for e in events)

    def test_tool_call_and_result(self, agent):
        """Model uses shell tool and responds with result."""
        events = list(agent.step("What files are in /workspace? Use the shell tool."))
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(tool_calls) >= 1
        assert len(tool_results) >= 1

    def test_multi_step_tool_use(self, agent):
        """Model chains multiple commands."""
        events = list(agent.step(
            "Create a file called hello.py in /workspace that prints 'hello world', "
            "then run it and tell me the output."
        ))
        tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tool_calls) >= 2  # at least write + run

    def test_context_window_error(self, agent):
        """Hard error when context is exhausted."""
        with pytest.raises(ContextWindowExhaustedError):
            for _ in range(100):
                list(agent.step("Run: head -c 10000 /dev/urandom | base64"))
```

### E2E Fixtures

```python
# tests/e2e/conftest.py

@pytest.fixture(scope="session")
def engine():
    """Load model once for all e2e tests."""
    from harnessthing.engine import Engine
    return Engine(model_name="mlx-community/gemma-4-e4b-it-4bit")

@pytest.fixture(scope="session")
def sandbox():
    """Start Docker sandbox once, tear down after all tests."""
    subprocess.run(["docker", "build", "-t", "harnessthing-sandbox",
                     "-f", "Dockerfile.sandbox", "."], check=True)
    subprocess.run(["docker", "run", "-d", "--name", "harnessthing-test-sandbox",
                     "--network", "bridge",
                     "-v", f"{Path.cwd()}/workspace:/workspace",
                     "harnessthing-sandbox", "sleep", "infinity"], check=True)
    yield
    subprocess.run(["docker", "rm", "-f", "harnessthing-test-sandbox"])

@pytest.fixture
def agent(engine, sandbox):
    """Fresh agent per test. Shares model and sandbox."""
    from harnessthing.agent import Agent
    from harnessthing.executor import DockerExecutor
    executor = DockerExecutor(container_name="harnessthing-test-sandbox")
    return Agent(engine=engine, executor=executor)
```

### Testing principles

- **Unit tests mock boundaries, not internals.** Mock `subprocess.run` (testing command construction, not bash). Mock token iterator (testing loop logic, not the model).
- **E2E tests mock nothing.** Real model, real Docker, real shell. Only way to verify tool call reliability end-to-end.
- **Session-scoped model.** Load once, not per test.
- **E2E is opt-in** (`--e2e` flag). Unit tests run in seconds. E2E runs when you mean it.
- **Deterministic seeding.** E2E sets `temperature=0`, fixed seed. Assertions check structure ("tool was called", "response non-empty"), not exact text.

## CLI Interface

```bash
# Default: docker sandbox, default model, current dir as workspace
harnessthing

# No sandbox
harnessthing --sandbox=none

# Custom model
harnessthing --model mlx-community/gemma-2-9b-it-4bit

# Custom workspace (any path — a repo, a scratch dir, whatever)
harnessthing --workspace ~/projects/my-app

# Specific HF token (only needed for gated models, not Gemma 4)
HF_TOKEN=hf_xxx harnessthing
```

## Non-Goals

- Permissions / sandboxing beyond Docker
- Multiple tool types (just shell)
- Conversation persistence / save / load
- Web UI
- Plugin system
- Multi-user / multi-session
- Step limits on tool call chains
- Constrained decoding (native tool calling via chat template; not needed)
