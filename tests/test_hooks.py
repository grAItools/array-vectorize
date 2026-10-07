# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""The hook runner installs both stages and preserves shared tool commands."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).parents[1]


def test_prek_installs_both_configured_hook_stages(tmp_path: pathlib.Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".pre-commit-config.yaml").write_bytes(
        (ROOT / ".pre-commit-config.yaml").read_bytes()
    )
    old_hook = tmp_path / ".git" / "hooks" / "pre-commit"
    old_hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8", newline="\n")
    prek = pathlib.Path(sys.executable).with_name("prek.exe" if os.name == "nt" else "prek")
    subprocess.run([str(prek), "install", "--force"], cwd=tmp_path, check=True, capture_output=True)
    for stage in ("pre-commit", "commit-msg"):
        hook = tmp_path / ".git" / "hooks" / stage
        text = hook.read_text(encoding="utf-8")
        assert "prek" in text
        assert stage in text
        assert not hook.with_name(stage + ".legacy").exists()


def test_hooks_share_locked_tools_and_make_type_targets() -> None:
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    hooks = {hook["id"]: hook for repo in config["repos"] for hook in repo["hooks"]}
    assert config["default_install_hook_types"] == ["pre-commit", "commit-msg"]
    assert hooks["license-headers"]["entry"] == "make headers"
    assert hooks["license-headers"]["always_run"] is True
    for checker in ("mypy", "pyright", "zuban", "pyrefly"):
        assert hooks[checker]["entry"] == f"make type-{checker}"
        assert hooks[checker]["pass_filenames"] is False
    for name in ("cleanporter-fix", "ruff-check", "ruff-format"):
        assert hooks[name]["entry"].startswith("uv run --frozen ")
    assert hooks["conventional-pre-commit"]["stages"] == ["commit-msg"]
