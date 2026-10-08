# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Validate release metadata and Git state before building distribution artifacts."""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import subprocess
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
_NUMBER = r"(?:0|[1-9][0-9]*)"
_VERSION = re.compile(
    rf"(?:[1-9][0-9]*!)?{_NUMBER}(?:\.{_NUMBER})*"
    rf"(?:(?:a|b|rc){_NUMBER})?(?:\.post{_NUMBER})?(?:\.dev{_NUMBER})?"
)


def version_from_tag(tag: str) -> str:
    """Return a normalized public Python version, rejecting aliases and local versions."""
    if not tag.startswith("v") or _VERSION.fullmatch(tag[1:]) is None:
        raise ValueError(f"invalid release tag: {tag!r}; expected v<normalized public version>")
    return tag[1:]


def is_prerelease(version: str) -> bool:
    """Identify development and alpha, beta, or release candidate versions."""
    return re.search(r"(?:a|b|rc|\.dev)[0-9]+", version) is not None


def validate_metadata(root: pathlib.Path, tag: str) -> str:
    """Require matching project, source, lock, and changelog versions."""
    version = version_from_tag(tag)
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if project["version"] != version:
        raise ValueError("pyproject.toml version does not match release tag")
    tree = ast.parse((root / "src/array_vectorize/__init__.py").read_text(encoding="utf-8"))
    declarations = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
    ]
    if declarations != [version]:
        raise ValueError("source __version__ does not match release tag")
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    entries = [entry for entry in lock["package"] if entry["name"] == project["name"]]
    if len(entries) != 1 or entries[0]["version"] != version:
        raise ValueError("uv.lock project version does not match release tag")
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    if re.search(rf"^## \[{re.escape(version)}\](?:\s|$)", changelog, re.MULTILINE) is None:
        raise ValueError("CHANGELOG.md has no release section for this version")
    return version


def check_clean(root: pathlib.Path) -> None:
    """Reject changes to tracked files and nonignored untracked files."""
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    if result.stdout:
        raise ValueError("release checkout is dirty; commit or remove changes before releasing")


def validate_checkout(root: pathlib.Path) -> None:
    """Check cleanliness, ancestry against origin/main, and lock freshness."""
    check_clean(root)
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "HEAD", "origin/main"], cwd=root, check=False
    )
    if result.returncode:
        raise ValueError("release commit must be an ancestor of origin/main (fetch origin first)")
    subprocess.run(["uv", "lock", "--check"], cwd=root, check=True)


def main() -> int:
    """Validate a tag or perform a final cleanliness check before building."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="release tag, for example v0.1.0")
    parser.add_argument("--check-clean", action="store_true", help="check Git cleanliness only")
    parser.add_argument(
        "--prerelease", action="store_true", help="print tag prerelease status only"
    )
    args = parser.parse_args()
    if not args.check_clean and not args.tag:
        parser.error("--tag is required unless --check-clean is supplied")
    try:
        if args.check_clean:
            check_clean(ROOT)
        elif args.prerelease:
            print(str(is_prerelease(version_from_tag(args.tag))).lower())
        else:
            validate_metadata(ROOT, args.tag)
            validate_checkout(ROOT)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"release validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
