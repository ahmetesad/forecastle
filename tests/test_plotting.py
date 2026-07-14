from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_cli_import_ignores_unavailable_notebook_backend() -> None:
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    environment["PYTHONPATH"] = str(Path("src").resolve())

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import forecastle.cli; import matplotlib; print(matplotlib.get_backend())",
        ],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().lower() == "agg"
