from __future__ import annotations

from pathlib import Path

import yaml


def test_wig20_preset_uses_direct_yahoo_symbol_not_proxy() -> None:
    raw = yaml.safe_load(Path("configs/downloads.yaml").read_text(encoding="utf-8"))

    assert raw["datasets"]["wig20"]["symbol"] == "WIG20.WA"
