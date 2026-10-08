# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""License enforcement covers repository-owned code and preserves source bytes."""

from __future__ import annotations

import ast
import pathlib
import shutil
import subprocess
import sys
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "license_headers.py"
HEADER = (ROOT / ".license-header.txt").read_bytes() + b"\n"


def _git(root: pathlib.Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, capture_output=True, check=True)


@pytest.fixture
def repository(tmp_path: pathlib.Path) -> pathlib.Path:
    script = tmp_path / "scripts" / "license_headers.py"
    script.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPT, script)
    shutil.copyfile(ROOT / ".license-header.txt", tmp_path / ".license-header.txt")
    _git(tmp_path, "init", "--quiet")
    # The copied utility is itself part of the checked repository.
    assert _run(tmp_path, "--fix").returncode == 0
    return tmp_path


def _run(root: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(root / "scripts" / "license_headers.py"), *args],
        cwd=root / "scripts",
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("preamble", "body"),
    [
        (b"", b""),
        (b"", b"x = 1"),
        (b"", b"\n\nx = 1\n"),
        (b"", b"\r\n\r\nx = 1\r\n"),
        (b"", b"\n\n"),
        (b"#!/usr/bin/env python\n", b"\n\nx = 1\n"),
        (
            b"",
            b'"""Module documentation."""\nfrom __future__ import annotations\nx = "caf\xc3\xa9"\n',
        ),
        (b"#!/usr/bin/env python\n", b"print('hello')\n"),
        (b"# coding: latin-1\n", b"x = 'caf\xe9'\n"),
        (b"#!/usr/bin/python\n# coding: latin-1\n", b"x = 'caf\xe9'\n"),
        (b"# an unrelated comment\n# coding: latin-1\n", b"x = 'caf\xe9'\n"),
        (b"\xef\xbb\xbf", b"x = 1\n"),
        (b"#!/usr/bin/python\r\n", b"x = 1\r\n"),
    ],
)
def test_fix_preserves_source_and_is_idempotent(
    repository: pathlib.Path, preamble: bytes, body: bytes
) -> None:
    path = repository / "a file.py"
    raw = preamble + body
    path.write_bytes(raw)
    before = ast.dump(ast.parse(raw))
    checked = _run(repository, "--check")
    assert checked.returncode == 1
    assert "a file.py" in checked.stdout
    assert path.read_bytes() == raw
    assert _run(repository, "--fix").returncode == 0
    header = HEADER.replace(b"\n", b"\r\n") if b"\r\n" in raw else HEADER
    expected = preamble + header + body
    assert path.read_bytes() == expected
    assert ast.dump(ast.parse(expected)) == before
    assert _run(repository, "--check").returncode == 0
    assert _run(repository, "--fix").returncode == 0
    assert path.read_bytes() == expected


@pytest.mark.parametrize(
    "old",
    [
        HEADER + HEADER,
        HEADER.rstrip(b"\n") + b"\n",
        HEADER.split(b"\n", 1)[1],
        b"# array-vectorize: Previous project description.\n" + HEADER.split(b"\n", 1)[1],
        HEADER.replace(b"2026", b"2025"),
        HEADER.replace(b"Copyright (c)", b"copyright   (c)").replace(b"# SPDX", b"#SPDX"),
        HEADER.replace(b"See LICENSE for the full license text.", b"See LICENSE for license text."),
        b"# Copyright (c) 2026 grAItools\n",
    ],
)
def test_normalize_owned_headers(repository: pathlib.Path, old: bytes) -> None:
    path = repository / "example.py"
    body = b"# Preserve this comment.\nx = 1\n"
    path.write_bytes(old + body)
    assert _run(repository, "--check").returncode == 1
    assert path.read_bytes() == old + body
    assert _run(repository, "--fix").returncode == 0
    assert path.read_bytes() == HEADER + body
    assert _run(repository, "--check").returncode == 0


@pytest.mark.parametrize(
    "notice",
    [
        b"# Copyright (c) 2026 Someone Else\n",
        b"# SPDX-License-Identifier: Apache-2.0\n",
        b"# Copyright (c) 2026 grAItools\n# SPDX-License-Identifier: MIT\n",
        b"# Copyright (c) 2026 grAItools\n# SPDX-License-Identifier: Apache-2.0\n",
        b"# Copyright (c) 2026 grAItools\n# Copyright (c) 2026 Someone Else\n",
        b"# Licensed under Apache-2.0\n",
        b"# This file is licensed under GPL-3.0-only\n",
        b"# License: Apache-2.0\n",
    ],
)
def test_conflicting_notices_are_never_overwritten(repository: pathlib.Path, notice: bytes) -> None:
    path = repository / "conflict.py"
    raw = notice + b"x = 1\n"
    path.write_bytes(raw)
    for mode in ("--check", "--fix"):
        result = _run(repository, mode)
        assert result.returncode == 1
        assert "conflict.py" in result.stderr
        assert path.read_bytes() == raw


def test_git_scope_includes_hidden_untracked_and_fixtures(repository: pathlib.Path) -> None:
    ignored = repository / "ignored.py"
    ignored.write_bytes(b"x = 1\n")
    (repository / ".gitignore").write_text("ignored.py\n", encoding="utf-8", newline="\n")
    fixtures = repository / "tests" / "fixtures"
    fixtures.mkdir(parents=True)
    fixture = fixtures / "empty.py"
    fixture.touch()
    _git(repository, "add", "tests/fixtures/empty.py")
    hidden = repository / ".github" / "new.py"
    hidden.parent.mkdir()
    hidden.touch()
    stub = repository / "module.pyi"
    stub.write_bytes(b"x: int\n")
    deleted = repository / "deleted.py"
    deleted.touch()
    _git(repository, "add", "deleted.py")
    deleted.unlink()
    assert _run(repository, "--fix").returncode == 0
    assert fixture.read_bytes() == HEADER
    assert hidden.read_bytes() == HEADER
    assert stub.read_bytes() == HEADER + b"x: int\n"
    assert ignored.read_bytes() == b"x = 1\n"


def test_misplaced_notices_are_rejected_and_embedded_inputs_are_untouched(
    repository: pathlib.Path,
) -> None:
    path = repository / "misplaced.py"
    raw = b"x = 1\n" + HEADER
    path.write_bytes(raw)
    for mode in ("--check", "--fix"):
        assert _run(repository, mode).returncode == 1
        assert path.read_bytes() == raw
    embedded = b'input_code = """\n' + HEADER + b'x = 1\n"""\n'
    path.write_bytes(embedded)
    assert _run(repository, "--fix").returncode == 0
    assert path.read_bytes() == HEADER + embedded
    assert _run(repository, "--check").returncode == 0


def test_unrelated_copyright_comment_is_preserved(repository: pathlib.Path) -> None:
    path = repository / "comment.py"
    raw = b"# Verify copyright headers in our tests.\nx = 1\n"
    path.write_bytes(raw)
    assert _run(repository, "--fix").returncode == 0
    assert path.read_bytes() == HEADER + raw


def test_standalone_project_description_is_preserved(repository: pathlib.Path) -> None:
    path = repository / "description.py"
    raw = HEADER.split(b"\n", 1)[0] + b"\nx = 1\n"
    path.write_bytes(raw)
    for _ in range(2):
        assert _run(repository, "--fix").returncode == 0
        assert path.read_bytes() == HEADER + raw
        assert _run(repository, "--check").returncode == 0


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
@pytest.mark.parametrize("existing_header", [False, True])
@pytest.mark.parametrize(
    "body",
    [
        b"# See LICENSE for details\nx = 1\n",
        b"# See LICENSE for the full license text.\nx = 1\n",
        b"# Module explanation.\n# See LICENSE for details\n\nx = 1\n",
    ],
)
def test_license_references_outside_owned_header_are_preserved(
    repository: pathlib.Path, newline: bytes, existing_header: bool, body: bytes
) -> None:
    path = repository / "reference.py"
    header = HEADER.replace(b"\n", newline)
    body = body.replace(b"\n", newline)
    raw = (header if existing_header else b"") + body
    path.write_bytes(raw)
    assert _run(repository, "--check").returncode == (0 if existing_header else 1)
    assert path.read_bytes() == raw
    for _ in range(2):
        assert _run(repository, "--fix").returncode == 0
        assert path.read_bytes() == header + body
        assert _run(repository, "--check").returncode == 0


def test_source_symlinks_are_rejected(repository: pathlib.Path, tmp_path: pathlib.Path) -> None:
    target = tmp_path / "outside.txt"
    target.write_bytes(b"x = 1\n")
    link = repository / "linked.py"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.fail(f"symlink support is required: {exc}")
    for mode in ("--check", "--fix"):
        result = _run(repository, mode)
        assert result.returncode == 1
        assert "symlink" in result.stderr
        assert target.read_bytes() == b"x = 1\n"


def test_bad_template_and_git_errors_are_operational(repository: pathlib.Path) -> None:
    template = repository / ".license-header.txt"
    original = template.read_bytes()
    template.write_bytes(b"not a comment\n")
    assert _run(repository, "--check").returncode == 2
    template.write_bytes(original)
    shutil.rmtree(repository / ".git")
    assert _run(repository, "--check").returncode == 2


def test_symlinked_parent_directories_are_never_followed(
    repository: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    directory = repository / "src"
    directory.mkdir()
    path = directory / "module.py"
    path.write_bytes(b"x = 1\n")
    _git(repository, "add", "src/module.py")
    path.unlink()
    directory.rmdir()
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    target = outside / "module.py"
    target.write_bytes(b"x = 1\n")
    directory.symlink_to(outside, target_is_directory=True)
    for mode in ("--check", "--fix"):
        result = _run(repository, mode)
        assert result.returncode == 1
        assert "symlink" in result.stderr
        assert target.read_bytes() == b"x = 1\n"


def test_unknown_source_encoding_is_operational(repository: pathlib.Path) -> None:
    path = repository / "bad.py"
    raw = b"# coding: nonexistent-encoding\nx = 1\n"
    path.write_bytes(raw)
    assert _run(repository, "--fix").returncode == 2
    assert path.read_bytes() == raw


def test_fix_preserves_file_permissions(repository: pathlib.Path) -> None:
    path = repository / "executable.py"
    path.write_bytes(b"x = 1\n")
    path.chmod(0o755)
    original = path.stat().st_mode
    assert _run(repository, "--fix").returncode == 0
    assert path.stat().st_mode == original


def test_shebang_without_a_final_newline(repository: pathlib.Path) -> None:
    path = repository / "script.py"
    path.write_bytes(b"#!/usr/bin/env python")
    assert _run(repository, "--fix").returncode == 0
    expected = b"#!/usr/bin/env python\n" + HEADER
    assert path.read_bytes() == expected
    assert _run(repository, "--check").returncode == 0


def test_notice_before_an_encoding_cookie_is_rejected(repository: pathlib.Path) -> None:
    path = repository / "misplaced.py"
    raw = b"# Copyright (c) 2026 grAItools\n# coding: utf-8\nx = 1\n"
    path.write_bytes(raw)
    for mode in ("--check", "--fix"):
        assert _run(repository, mode).returncode == 1
        assert path.read_bytes() == raw


def test_cli_requires_exactly_one_mode(repository: pathlib.Path) -> None:
    assert _run(repository).returncode == 2
    assert _run(repository, "--check", "--fix").returncode == 2


def test_repository_headers_are_canonical() -> None:
    result = _run(ROOT, "--check")
    assert result.returncode == 0, result.stdout + result.stderr


def test_all_type_checkers_cover_the_header_utility() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]
    for checker, key in (
        ("mypy", "files"),
        ("pyright", "include"),
        ("zuban", "files"),
        ("pyrefly", "project-includes"),
    ):
        assert "scripts" in config[checker][key]
    assert config["pyrefly"]["disable-project-excludes-heuristics"] is True


def test_header_hook_checks_the_whole_tree_even_on_template_only_changes() -> None:
    # PyYAML is a dev dependency, independent of libcst's optional YAML backend.
    code = (
        "import json, sys, yaml\n"
        "with open(sys.argv[1], encoding='utf-8') as f:\n"
        "    json.dump(yaml.safe_load(f), sys.stdout)\n"
    )
    import json

    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT / ".pre-commit-config.yaml")],
        capture_output=True,
        text=True,
        check=True,
    )
    config = json.loads(result.stdout)
    [hook] = [
        hook
        for repo in config["repos"]
        for hook in repo["hooks"]
        if hook["id"] == "license-headers"
    ]
    assert hook["entry"] == "make headers"
    assert hook["language"] == "system"
    assert hook["always_run"] is True
    assert hook["pass_filenames"] is False
