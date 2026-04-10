from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from harnessthing.executor import CommandResult, DockerExecutor, SubprocessExecutor


class TestDockerExecutor:
    def test_run_constructs_correct_command(self):
        executor = DockerExecutor(container_name="test-sandbox")
        executor._started = True

        with patch("harnessthing.executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="output\n", stderr=""
            )
            result = executor.run("ls -la")

        mock_run.assert_called_once_with(
            ["docker", "exec", "test-sandbox", "bash", "-c", "ls -la"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.exit_code == 0
        assert result.stdout == "output\n"

    def test_run_captures_stderr(self):
        executor = DockerExecutor(container_name="test-sandbox")
        executor._started = True

        with patch("harnessthing.executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="error\n"
            )
            result = executor.run("bad-cmd")

        assert result.exit_code == 1
        assert result.stderr == "error\n"

    def test_run_non_zero_exit_code(self):
        executor = DockerExecutor(container_name="test-sandbox")
        executor._started = True

        with patch("harnessthing.executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=127, stdout="", stderr="command not found"
            )
            result = executor.run("nonexistent")

        assert result.exit_code == 127

    def test_run_timeout(self):
        executor = DockerExecutor(container_name="test-sandbox", timeout=5)
        executor._started = True

        with patch("harnessthing.executor.subprocess.run") as mock_run:
            import subprocess
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=5)
            result = executor.run("sleep 100")

        assert result.exit_code == 124
        assert "timed out" in result.stderr


class TestSubprocessExecutor:
    def test_run_constructs_correct_command(self):
        executor = SubprocessExecutor()

        with patch("harnessthing.executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="output\n", stderr=""
            )
            result = executor.run("echo hello")

        mock_run.assert_called_once_with(
            ["bash", "-c", "echo hello"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.exit_code == 0
        assert result.stdout == "output\n"

    def test_run_captures_stdout_and_stderr(self):
        executor = SubprocessExecutor()

        with patch("harnessthing.executor.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="out\n", stderr="warn\n"
            )
            result = executor.run("some-cmd")

        assert result.stdout == "out\n"
        assert result.stderr == "warn\n"

    def test_run_timeout(self):
        executor = SubprocessExecutor(timeout=5)

        with patch("harnessthing.executor.subprocess.run") as mock_run:
            import subprocess
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=5)
            result = executor.run("sleep 100")

        assert result.exit_code == 124
        assert "timed out" in result.stderr
