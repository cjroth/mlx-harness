from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from harnessthing.config import DEFAULT_MODEL, parse_args, resolve_hf_token


class TestResolveHfToken:
    def test_from_env_var(self):
        with patch.dict("os.environ", {"HF_TOKEN": "hf_test123"}):
            assert resolve_hf_token() == "hf_test123"

    def test_from_file(self, tmp_path):
        token_file = tmp_path / ".cache" / "huggingface" / "token"
        token_file.parent.mkdir(parents=True)
        token_file.write_text("hf_fromfile\n")

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("harnessthing.config.Path.home", return_value=tmp_path),
        ):
            assert resolve_hf_token() == "hf_fromfile"

    def test_env_var_takes_precedence(self, tmp_path):
        token_file = tmp_path / ".cache" / "huggingface" / "token"
        token_file.parent.mkdir(parents=True)
        token_file.write_text("hf_fromfile\n")

        with (
            patch.dict("os.environ", {"HF_TOKEN": "hf_env"}),
            patch("harnessthing.config.Path.home", return_value=tmp_path),
        ):
            assert resolve_hf_token() == "hf_env"

    def test_returns_none_when_missing(self, tmp_path):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("harnessthing.config.Path.home", return_value=tmp_path),
        ):
            assert resolve_hf_token() is None


class TestParseArgs:
    def test_defaults(self):
        config = parse_args([])
        assert config.model == DEFAULT_MODEL
        assert config.sandbox == "docker"
        assert config.workspace == Path.cwd()

    def test_custom_model(self):
        config = parse_args(["--model", "some/other-model"])
        assert config.model == "some/other-model"

    def test_sandbox_none(self):
        config = parse_args(["--sandbox=none"])
        assert config.sandbox == "none"

    def test_custom_workspace(self, tmp_path):
        config = parse_args(["--workspace", str(tmp_path)])
        assert config.workspace == tmp_path.resolve()
