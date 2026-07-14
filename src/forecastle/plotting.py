from __future__ import annotations

import os

# write plots to files and never require an interactive GUI backend
os.environ["MPLBACKEND"] = "Agg"

from matplotlib import pyplot as plt

__all__ = ["plt"]
