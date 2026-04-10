from __future__ import annotations

import sys

from harnessthing.config import parse_args
from harnessthing.engine import Engine
from harnessthing.executor import DockerExecutor, SubprocessExecutor
from harnessthing.agent import Agent
from harnessthing.tui import run_loop


def main() -> None:
    config = parse_args()

    print(f"Loading model: {config.model}")
    engine = Engine(model_name=config.model)
    print("Model loaded.")

    if config.sandbox == "docker":
        executor = DockerExecutor(
            workspace=config.workspace,
        )
    else:
        executor = SubprocessExecutor()

    try:
        executor.start()
        agent = Agent(engine=engine, executor=executor)
        run_loop(agent)
    finally:
        executor.stop()


if __name__ == "__main__":
    main()
