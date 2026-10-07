# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Coding-agent configuration layout (see "Agent configuration" in AGENTS.md).

The harness-agnostic sources are AGENTS.md and .agents/; harness
directories such as .claude/ only forward to them.
"""

from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).parents[1]
SKILLS = ROOT / ".agents" / "skills"
CLAUDE_SKILLS = ROOT / ".claude" / "skills"
CLAUDE_AGENTS = ROOT / ".claude" / "agents"

#: frontmatter fields defined by the Agent Skills specification
#: (agentskills.io); harness-specific fields would not travel
STANDARD_FIELDS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
SKILL_NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


def _frontmatter(path: pathlib.Path) -> dict[str, str | list[str]]:
    """Top-level keys of a markdown file's YAML frontmatter (scalars, lists)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "---", f"{path} has no frontmatter"
    fields: dict[str, str | list[str]] = {}
    key = ""
    for line in lines[1 : lines.index("---", 1)]:
        if item := re.fullmatch(r"\s+- (.+)", line):
            entries = fields.setdefault(key, [])
            assert isinstance(entries, list)
            entries.append(item[1])
        elif field := re.fullmatch(r"([\w-]+):\s*(.*)", line):
            key = field[1]
            fields[key] = field[2] if field[2] else []
    return fields


SKILL_DIRS = sorted(path for path in SKILLS.iterdir() if path.is_dir())


def test_there_are_skills() -> None:
    assert SKILL_DIRS


@pytest.mark.parametrize("skill", SKILL_DIRS, ids=lambda p: p.name)
def test_skill_uses_standard_frontmatter(skill: pathlib.Path) -> None:
    fields = _frontmatter(skill / "SKILL.md")
    assert set(fields) <= STANDARD_FIELDS, set(fields) - STANDARD_FIELDS
    assert fields["name"] == skill.name
    assert SKILL_NAME.fullmatch(skill.name)
    assert len(skill.name) <= 64
    assert isinstance(fields["description"], str)
    assert 0 < len(fields["description"]) <= 1024


def test_claude_skills_are_symlinks_to_agents_skills() -> None:
    linked = {}
    for entry in CLAUDE_SKILLS.iterdir():
        assert entry.is_symlink(), f"{entry} must be a symlink into .agents/skills"
        linked[entry.name] = entry.resolve()
    assert linked == {skill.name: skill.resolve() for skill in SKILL_DIRS}


@pytest.mark.parametrize("agent", sorted(CLAUDE_AGENTS.glob("*.md")), ids=lambda p: p.name)
def test_claude_subagents_preload_shared_skills(agent: pathlib.Path) -> None:
    skills = _frontmatter(agent).get("skills", [])
    assert skills, f"{agent} should wrap a skill from .agents/skills"
    assert isinstance(skills, list)
    assert set(skills) <= {skill.name for skill in SKILL_DIRS}


def test_claude_md_forwards_to_agents_md() -> None:
    assert (ROOT / "CLAUDE.md").read_text(encoding="utf-8").strip() == "@AGENTS.md"
