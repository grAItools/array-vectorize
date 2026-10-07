#!/usr/bin/env python3
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Stop hook: when Python files changed, run `make lint type` before stopping.

On failure the output goes back to the agent (exit 2) so it fixes the
problems first. A second consecutive stop is let through
(``stop_hook_active``), so the hook never loops.
"""

import json
import subprocess
import sys

if json.load(sys.stdin).get("stop_hook_active"):
    sys.exit(0)

status = subprocess.run(
    ["git", "status", "--porcelain", "--", "*.py", "*.pyi"],
    capture_output=True,
    text=True,
)
if not status.stdout.strip():
    sys.exit(0)

gate = subprocess.run(["make", "lint", "type"], capture_output=True, text=True)
if gate.returncode != 0:
    tail = (gate.stdout + gate.stderr).splitlines()[-60:]
    print("`make lint type` fails on the current changes:", *tail, sep="\n", file=sys.stderr)
    sys.exit(2)
