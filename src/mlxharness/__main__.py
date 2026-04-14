from __future__ import annotations

from mlxharness.bus import EventBus
from mlxharness.config import parse_args
from mlxharness.engine import Engine
from mlxharness.executor import DockerExecutor, SubprocessExecutor
from mlxharness.orchestrator import Orchestrator
from mlxharness.tui import HarnessApp


def main() -> None:
    config = parse_args()

    print(f"Loading model: {config.model}")
    engine = Engine(model_name=config.model)
    print("Model loaded.")

    if config.sandbox == "docker":
        executor = DockerExecutor(workspace=config.workspace)
    else:
        executor = SubprocessExecutor()

    bus = EventBus()
    orchestrator = Orchestrator(engine=engine, executor=executor, bus=bus)

    try:
        executor.start()
        orchestrator.start()
        HarnessApp(orchestrator).run()
    finally:
        orchestrator.shutdown()
        executor.stop()


if __name__ == "__main__":
    main()
