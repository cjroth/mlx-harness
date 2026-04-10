from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def engine():
    """Load model once for all e2e tests."""
    from harnessthing.engine import Engine
    return Engine(model_name="mlx-community/gemma-4-e4b-it-4bit")


@pytest.fixture(scope="session")
def sandbox():
    """Start Docker sandbox once, tear down after all tests."""
    subprocess.run(
        ["docker", "build", "-t", "harnessthing-sandbox", "-f", "Dockerfile.sandbox", "."],
        check=True,
    )
    # Remove stale container from previous interrupted runs
    subprocess.run(["docker", "rm", "-f", "harnessthing-test-sandbox"], capture_output=True)
    subprocess.run(
        [
            "docker", "run", "-d",
            "--name", "harnessthing-test-sandbox",
            "--network", "bridge",
            "-v", f"{Path.cwd()}/workspace:/workspace",
            "harnessthing-sandbox", "sleep", "infinity",
        ],
        check=True,
    )
    yield
    subprocess.run(["docker", "rm", "-f", "harnessthing-test-sandbox"])


@pytest.fixture
def agent(engine, sandbox):
    """Fresh agent per test. Shares model and sandbox."""
    from harnessthing.agent import Agent
    from harnessthing.executor import DockerExecutor
    executor = DockerExecutor(container_name="harnessthing-test-sandbox")
    executor._started = True  # Container already running from fixture
    return Agent(engine=engine, executor=executor)
