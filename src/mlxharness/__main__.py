from __future__ import annotations

import sys

from mlxharness.config import parse_args
from mlxharness.engine import Engine
from mlxharness.executor import DockerExecutor, SubprocessExecutor
from mlxharness.agent import Agent
from mlxharness.tui import HarnessApp


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
        HarnessApp(agent).run()
    finally:
        executor.stop()


if __name__ == "__main__":
    main()
