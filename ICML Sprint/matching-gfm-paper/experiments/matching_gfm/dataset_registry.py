from __future__ import annotations

from pathlib import Path
from typing import Any

from .hm_local import load_hm_local_market
from .polymarket_real import load_polymarket_market
from .real_market import ObservedMarket


AVAILABLE_SOURCES = ("polymarket", "hm_local")


def load_observed_market(source: str, **kwargs: Any) -> ObservedMarket:
    if source == "polymarket":
        return load_polymarket_market(**kwargs)
    if source == "hm_local":
        data_dir = kwargs.pop("data_dir", None)
        if data_dir is None:
            raise ValueError("hm_local requires data_dir pointing to customers.csv, articles.csv, and transactions_train.csv")
        return load_hm_local_market(data_dir=Path(data_dir), **kwargs)
    raise ValueError(f"Unknown source '{source}'. Expected one of: {', '.join(AVAILABLE_SOURCES)}")
