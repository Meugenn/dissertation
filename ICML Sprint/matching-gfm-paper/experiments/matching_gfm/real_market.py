from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .synthetic_market import TemporalEdge


@dataclass(frozen=True)
class ObservedMarket:
    buyer_ids: tuple[str, ...]
    seller_ids: tuple[str, ...]
    buyer_features: np.ndarray
    seller_features: np.ndarray
    train_edges: tuple[TemporalEdge, ...]
    eval_edges: tuple[TemporalEdge, ...]
    edge_type_names: tuple[str, ...]
    edge_type_weights: np.ndarray
    train_cutoff: float
    total_horizon: float
    source_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_buyers(self) -> int:
        return int(self.buyer_features.shape[0])

    @property
    def num_sellers(self) -> int:
        return int(self.seller_features.shape[0])

    @property
    def num_edge_types(self) -> int:
        return len(self.edge_type_names)

    def pair_tensor(
        self,
        edges: tuple[TemporalEdge, ...],
        reference_time: float | None = None,
        half_life: float = 10.0,
    ) -> np.ndarray:
        reference = self.total_horizon if reference_time is None else reference_time
        tensor = np.zeros((self.num_buyers, self.num_sellers, self.num_edge_types), dtype=np.float64)
        for edge in edges:
            age = max(reference - edge.time, 0.0)
            decay = 0.5 ** (age / max(half_life, 1e-6))
            tensor[edge.buyer, edge.seller, edge.edge_type] += decay
        return tensor

    def weighted_pair_matrix(
        self,
        edges: tuple[TemporalEdge, ...],
        reference_time: float | None = None,
        half_life: float = 10.0,
    ) -> np.ndarray:
        tensor = self.pair_tensor(edges=edges, reference_time=reference_time, half_life=half_life)
        return np.tensordot(tensor, self.edge_type_weights, axes=([2], [0]))

    def train_pair_tensor(self, half_life: float = 10.0) -> np.ndarray:
        return self.pair_tensor(self.train_edges, reference_time=self.train_cutoff, half_life=half_life)

    def eval_pair_tensor(self, half_life: float = 10.0) -> np.ndarray:
        return self.pair_tensor(self.eval_edges, reference_time=self.total_horizon, half_life=half_life)

    def train_pair_matrix(self, half_life: float = 10.0) -> np.ndarray:
        return self.weighted_pair_matrix(self.train_edges, reference_time=self.train_cutoff, half_life=half_life)

    def eval_pair_matrix(self, half_life: float = 10.0) -> np.ndarray:
        return self.weighted_pair_matrix(self.eval_edges, reference_time=self.total_horizon, half_life=half_life)

    def edge_frame(self, split: str = "train") -> pd.DataFrame:
        edges = self.train_edges if split == "train" else self.eval_edges
        rows = []
        for edge in edges:
            rows.append(
                {
                    "buyer_idx": edge.buyer,
                    "seller_idx": edge.seller,
                    "buyer_id": self.buyer_ids[edge.buyer],
                    "seller_id": self.seller_ids[edge.seller],
                    "edge_type": self.edge_type_names[edge.edge_type],
                    "time": edge.time,
                }
            )
        return pd.DataFrame(rows)

    def write_snapshot(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"buyer_idx": np.arange(self.num_buyers), "buyer_id": self.buyer_ids}).to_csv(
            output_dir / "buyers.csv",
            index=False,
        )
        pd.DataFrame({"seller_idx": np.arange(self.num_sellers), "seller_id": self.seller_ids}).to_csv(
            output_dir / "sellers.csv",
            index=False,
        )
        self.edge_frame("train").to_csv(output_dir / "train_edges.csv", index=False)
        self.edge_frame("eval").to_csv(output_dir / "eval_edges.csv", index=False)
