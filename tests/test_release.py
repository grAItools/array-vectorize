# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Release metadata, checkout safety, and artifact smoke orchestration."""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import types
import typing

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = _load("release")
package_smoke = _load("package_smoke")


@pytest.fixture
def project(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "src/array_vectorize").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "array-vectorize"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    (tmp_path / "src/array_vectorize/__init__.py").write_text(
        '__version__ = "0.1.0"\n', encoding="utf-8"
    )
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname = "array-vectorize"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    (tmp_path / "CHANGELOG.md").write_text("## [0.1.0] - 2026-10-06\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("tag", ["v0.1.0", "v1!2.0rc1.post2.dev3", "v1.0.post1", "v1.0.dev0"])
def test_normalized_public_versions(tag: str) -> None:
    assert release.version_from_tag(tag) == tag[1:]


@pytest.mark.parametrize(
    "tag", ["1.0", "v", "v01.0", "v1.0+local", "v1.0-rc1", "v1.0RC1", "v1.0rc", "v0!1.0"]
)
def test_release_tags_reject_aliases_and_local_versions(tag: str) -> None:
    with pytest.raises(ValueError, match="invalid release tag"):
        release.version_from_tag(tag)


def test_versions_and_changelog_match(project: pathlib.Path) -> None:
    assert release.validate_metadata(project, "v0.1.0") == "0.1.0"


@pytest.mark.parametrize(
    ("filename", "text", "message"),
    [
        ("pyproject.toml", '[project]\nname="array-vectorize"\nversion="0.2.0"', "pyproject"),
        ("src/array_vectorize/__init__.py", '__version__="0.2.0"', "source"),
        ("uv.lock", '[[package]]\nname="array-vectorize"\nversion="0.2.0"', "uv.lock"),
        ("CHANGELOG.md", "## [Unreleased]\n", "CHANGELOG"),
    ],
)
def test_metadata_mismatch_rejected(
    project: pathlib.Path, filename: str, text: str, message: str
) -> None:
    (project / filename).write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        release.validate_metadata(project, "v0.1.0")


@pytest.mark.parametrize(
    ("version", "expected"),
    [("1.0", False), ("1.0.post1", False), ("1.0rc1", True), ("1.0.dev1", True)],
)
def test_prerelease_detection(version: str, expected: bool) -> None:
    assert release.is_prerelease(version) is expected


def _git(root: pathlib.Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)


@pytest.fixture
def checkout(project: pathlib.Path) -> pathlib.Path:
    _git(project, "init")
    _git(project, "config", "user.email", "test@example.com")
    _git(project, "config", "user.name", "Test")
    _git(project, "add", ".")
    _git(project, "commit", "-m", "initial")
    _git(project, "update-ref", "refs/remotes/origin/main", "HEAD")
    return project


def test_clean_checkout(checkout: pathlib.Path) -> None:
    release.check_clean(checkout)


@pytest.mark.parametrize("filename", ["CHANGELOG.md", "new.py"])
def test_dirty_checkout_is_rejected(checkout: pathlib.Path, filename: str) -> None:
    (checkout / filename).write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        release.check_clean(checkout)


def test_commit_outside_main_is_rejected(checkout: pathlib.Path) -> None:
    _git(checkout, "commit", "--allow-empty", "-m", "outside main")
    with pytest.raises(ValueError, match="ancestor"):
        release.validate_checkout(checkout)


def test_stale_lock_failure_is_propagated(
    checkout: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = subprocess.run

    def run(command: list[str], **kwargs: typing.Any) -> subprocess.CompletedProcess[str]:
        if command == ["uv", "lock", "--check"]:
            raise subprocess.CalledProcessError(1, command)
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        release.validate_checkout(checkout)


def test_release_workflow_validates_and_smokes_before_upload() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"))
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] is False
    build = workflow["jobs"]["build"]
    steps = build["steps"]
    commands = [step["run"] for step in steps if "run" in step]
    assert commands.index(
        'uv run --frozen python scripts/release.py --tag "$GITHUB_REF_NAME"'
    ) < commands.index("make check")
    assert commands.index(
        "uv run --frozen python scripts/release.py --check-clean"
    ) < commands.index("uv build --no-sources")
    assert commands.index("uv build --no-sources") < commands.index(
        'make package-smoke VERSION="${GITHUB_REF_NAME#v}"'
    )
    assert next(step for step in steps if step.get("run") == "make check")["env"]["CI"] == "1"
    assert "upload-artifact" in steps[-1]["uses"]
    publish = workflow["jobs"]["publish"]
    assert publish["needs"] == "build"
    assert publish["permissions"] == {"contents": "write"}
    assert "download-artifact" in publish["steps"][0]["uses"]
    assert "--verify-tag" in publish["steps"][1]["run"]


def test_artifacts_reject_stale_or_missing_distributions(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        package_smoke.artifacts(tmp_path)
    (tmp_path / "package.whl").touch()
    (tmp_path / "package.tar.gz").touch()
    assert len(package_smoke.artifacts(tmp_path)) == 2
    (tmp_path / ".gitignore").write_bytes(b"*")
    assert len(package_smoke.artifacts(tmp_path)) == 2
    (tmp_path / "stale.whl").touch()
    with pytest.raises(ValueError, match="exactly one"):
        package_smoke.artifacts(tmp_path)


def test_artifact_smoke_uses_isolated_environment(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, typing.Any]]] = []

    def run(command: list[str], **kwargs: typing.Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    package_smoke.smoke(tmp_path / "package.whl", "0.1.0")
    assert len(calls) == 3
    assert calls[1][0][-2:] == [str(tmp_path / "package.whl"), "numpy>=2.0"]
    assert calls[2][0][1] == "-I"
    for _, kwargs in calls:
        assert ROOT not in pathlib.Path(kwargs["cwd"]).parents
        assert "PYTHONPATH" not in kwargs["env"]


def test_artifact_install_failure_prevents_smoke(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: typing.Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[1:3] == ["pip", "install"]:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        package_smoke.smoke(tmp_path / "package.whl", "0.1.0")
    assert len(calls) == 2
