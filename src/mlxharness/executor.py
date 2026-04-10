from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

DEFAULT_TIMEOUT = 120


@dataclass
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str


class Executor(Protocol):
    def run(self, command: str) -> CommandResult: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...


class DockerExecutor:
    def __init__(
        self,
        container_name: str = "mlxharness-sandbox",
        workspace: Path | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.container_name = container_name
        self.workspace = workspace
        self.timeout = timeout
        self._started = False

    def start(self) -> None:
        if self._started:
            return

        # Build the image
        subprocess.run(
            ["docker", "build", "-t", "mlxharness-sandbox", "-f", "Dockerfile.sandbox", "."],
            check=True,
            capture_output=True,
        )

        # Remove any existing container with this name
        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            capture_output=True,
        )

        # Start the container
        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "--network", "bridge",
        ]
        if self.workspace:
            cmd.extend(["-v", f"{self.workspace}:/workspace"])
        cmd.extend(["mlxharness-sandbox", "sleep", "infinity"])

        subprocess.run(cmd, check=True, capture_output=True)
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            capture_output=True,
        )
        self._started = False

    def run(self, command: str) -> CommandResult:
        if not self._started:
            self.start()

        try:
            proc = subprocess.run(
                ["docker", "exec", self.container_name, "bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return CommandResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                exit_code=124,
                stdout="",
                stderr=f"Command timed out after {self.timeout}s",
            )


class SubprocessExecutor:
    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        self.timeout = timeout

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def run(self, command: str) -> CommandResult:
        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return CommandResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                exit_code=124,
                stdout="",
                stderr=f"Command timed out after {self.timeout}s",
            )
