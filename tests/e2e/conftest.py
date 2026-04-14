from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def engine():
    """Load model once for all e2e tests."""
    from mlxharness.engine import Engine
    return Engine(model_name="mlx-community/gemma-4-e4b-it-4bit")


@pytest.fixture(scope="session")
def sandbox():
    """Start Docker sandbox once, tear down after all tests."""
    subprocess.run(
        ["docker", "build", "-t", "mlxharness-sandbox", "-f", "Dockerfile.sandbox", "."],
        check=True,
    )
    # Remove stale container from previous interrupted runs
    subprocess.run(["docker", "rm", "-f", "mlxharness-test-sandbox"], capture_output=True)
    subprocess.run(
        [
            "docker", "run", "-d",
            "--name", "mlxharness-test-sandbox",
            "--network", "bridge",
            "-v", f"{Path.cwd()}/workspace:/workspace",
            "mlxharness-sandbox", "sleep", "infinity",
        ],
        check=True,
    )
    yield
    subprocess.run(["docker", "rm", "-f", "mlxharness-test-sandbox"])


@pytest.fixture
def agent(engine, sandbox):
    """Fresh agent per test. Shares model and sandbox."""
    from mlxharness.agent import Agent
    from mlxharness.executor import DockerExecutor
    executor = DockerExecutor(container_name="mlxharness-test-sandbox")
    executor._started = True  # Container already running from fixture
    return Agent(engine=engine, executor=executor)


@pytest.fixture
def orchestrator(engine, sandbox):
    """Fresh orchestrator per test. Shares model and sandbox."""
    from mlxharness.bus import EventBus
    from mlxharness.executor import DockerExecutor
    from mlxharness.orchestrator import Orchestrator

    executor = DockerExecutor(container_name="mlxharness-test-sandbox")
    executor._started = True
    bus = EventBus()
    orch = Orchestrator(engine=engine, executor=executor, bus=bus)
    orch.start()
    yield orch
    orch.shutdown()
