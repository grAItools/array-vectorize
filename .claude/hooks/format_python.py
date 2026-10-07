#!/usr/bin/env python3
"""PostToolUse hook: format an edited Python file with the locked ruff.

Only formatting and import sorting run here. Other autofixes would delete
an import the agent adds one edit before the code that uses it; the Stop
hook reports remaining lint instead.
"""

import json
import pathlib
import subprocess
import sys

path = json.load(sys.stdin).get("tool_input", {}).get("file_path", "")
if path.endswith((".py", ".pyi")) and pathlib.Path(path).is_file():
    ruff = ["uv", "run", "--frozen", "ruff"]
    subprocess.run([*ruff, "check", "--select", "I", "--fix", "--force-exclude", "-q", path])
    subprocess.run([*ruff, "format", "--force-exclude", "-q", path])
