# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Install and smoke-test the wheel and sdist in independent temporary environments."""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
_SMOKE = """import importlib.metadata
import pathlib
import sys
import numpy as np
import array_vectorize

package = pathlib.Path(array_vectorize.__file__).resolve()
assert pathlib.Path(sys.prefix).resolve() in package.parents, package
assert array_vectorize.__version__ == sys.argv[1]
assert importlib.metadata.version("array-vectorize") == sys.argv[1]
assert package.with_name("py.typed").is_file()

def scalar(x):
    if x < 0:
        return -x
    return x * 2

vec = array_vectorize.vectorize(scalar)
np.testing.assert_array_equal(vec(np.array([-3, 0, 4])), np.array([3, 0, 8]))
"""


def artifacts(directory: pathlib.Path) -> list[pathlib.Path]:
    """Require exactly one wheel and one sdist, preventing stale release uploads."""
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    # uv creates this marker in a fresh output directory; it is not uploaded.
    entries = {
        path for path in directory.iterdir() if path.name != ".gitignore" or not path.is_file()
    }
    if len(wheels) != 1 or len(sdists) != 1 or entries != set(wheels + sdists):
        raise ValueError(
            "dist must contain exactly one wheel and one sdist; remove stale artifacts"
        )
    return [wheels[0].resolve(), sdists[0].resolve()]


def smoke(artifact: pathlib.Path, version: str) -> None:
    """Install an exact artifact and execute source-backed vectorization outside the checkout."""
    environment = os.environ.copy()
    # Editable installs and user Python configuration must not shadow the artifact.
    for key in ("PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        environment.pop(key, None)
    with tempfile.TemporaryDirectory(prefix="array-vectorize-release-") as temporary:
        directory = pathlib.Path(temporary)
        venv = directory / "venv"
        subprocess.run(
            ["uv", "venv", "--python", sys.executable, str(venv)],
            cwd=directory,
            env=environment,
            check=True,
        )
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), str(artifact), "numpy>=2.0"],
            cwd=directory,
            env=environment,
            check=True,
        )
        script = directory / "smoke.py"
        script.write_text(_SMOKE, encoding="utf-8", newline="\n")
        subprocess.run(
            [str(python), "-I", str(script), version], cwd=directory, env=environment, check=True
        )


def main() -> int:
    """Smoke-test both artifacts from dist against the requested release version."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist", type=pathlib.Path, default=ROOT / "dist")
    args = parser.parse_args()
    try:
        for artifact in artifacts(args.dist):
            smoke(artifact, args.version)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"package smoke failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
