from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from meta_swag.axbench_meta_swag import RetainedCheckpoint  # noqa: E402
from meta_swag.hypersteer_posterior import (  # noqa: E402
    aggregate_sampled_factor_rows,
    attach_multi_concept_validation_metrics,
    choose_risk_sensitive_factor,
)


def test_attach_multi_concept_validation_metrics_averages_scores() -> None:
    record = RetainedCheckpoint(
        checkpoint_id="hyper_ckpt",
        step=10,
        epoch=0,
        train_loss=1.0,
        adapter_vector=np.zeros(4, dtype=np.float32),
        adapter_dimension=4,
    )
    updated = attach_multi_concept_validation_metrics(
        record,
        [
            {"concept_id": 0, "selected_factor": 0.8, "validation_composite": 0.7, "weighting_metric": 1.2},
            {"concept_id": 1, "selected_factor": 1.2, "validation_composite": 0.8, "weighting_metric": 1.8},
        ],
    )
    assert updated.weighting_metric == 1.5
    assert updated.selected_factor == 1.0
    assert len(updated.validation_factor_sweep) == 2


def test_aggregate_sampled_factor_rows_computes_mean_and_std() -> None:
    summary = aggregate_sampled_factor_rows(
        [
            [
                {"factor": 0.5, "composite": 0.8, "instruction_relevance": 1.0, "fluency": 1.2, "perplexity": 4.0},
                {"factor": 1.0, "composite": 0.6, "instruction_relevance": 0.9, "fluency": 1.0, "perplexity": 5.0},
            ],
            [
                {"factor": 0.5, "composite": 0.7, "instruction_relevance": 1.1, "fluency": 1.0, "perplexity": 6.0},
                {"factor": 1.0, "composite": 0.9, "instruction_relevance": 1.2, "fluency": 1.1, "perplexity": 4.5},
            ],
        ]
    )
    row = next(item for item in summary if item["factor"] == 0.5)
    assert np.isclose(row["composite_mean"], 0.75)
    assert row["composite_std"] > 0.0
    assert np.isclose(row["instruction_relevance_mean"], 1.05)


def test_choose_risk_sensitive_factor_prefers_lower_variance_when_means_match() -> None:
    factor, score = choose_risk_sensitive_factor(
        [
            {"factor": 0.8, "composite_mean": 0.82, "composite_std": 0.10, "instruction_relevance_mean": 1.1, "fluency_mean": 1.0, "perplexity_mean": 5.0},
            {"factor": 1.0, "composite_mean": 0.82, "composite_std": 0.03, "instruction_relevance_mean": 1.0, "fluency_mean": 1.0, "perplexity_mean": 5.0},
        ],
        risk_aversion=1.0,
    )
    assert factor == 1.0
    assert np.isclose(score, 0.79)
