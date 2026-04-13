from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from meta_swag.adapter_posterior import (  # noqa: E402
    aggregate_adapter_checkpoints,
    build_retention_schedule,
    effective_sample_size,
    find_beta_for_target_ess,
    softmax_weights,
    threshold_weights,
)
from meta_swag.adapter_state import build_manifest, flatten_adapter_state, restore_adapter_state  # noqa: E402
from meta_swag.axbench_meta_swag import attach_validation_metrics, choose_factor_from_factor_sweep  # noqa: E402
from run_axbench_meta_swag import select_concept_ids  # noqa: E402


class TinyAdapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(3, 2)
        self.adapter_a = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        self.adapter_b = torch.nn.Parameter(torch.arange(2, dtype=torch.float32))
        self.base.weight.requires_grad_(False)
        self.base.bias.requires_grad_(False)


def test_flatten_and_restore_adapter_state_is_lossless() -> None:
    module = TinyAdapter()
    manifest = build_manifest(module)
    original_vector, _ = flatten_adapter_state(module, manifest)

    with torch.no_grad():
        module.adapter_a.zero_()
        module.adapter_b.zero_()

    restore_adapter_state(module, original_vector, manifest)
    restored_vector, _ = flatten_adapter_state(module, manifest)
    np.testing.assert_allclose(restored_vector, original_vector)


def test_manifest_excludes_frozen_parameters() -> None:
    module = TinyAdapter()
    manifest = build_manifest(module)
    parameter_names = [spec.name for spec in manifest.parameters]
    assert "base.weight" not in parameter_names
    assert "base.bias" not in parameter_names
    assert "adapter_a" in parameter_names
    assert "adapter_b" in parameter_names


def test_retention_schedule_keeps_tail_steps() -> None:
    schedule = build_retention_schedule(total_steps=10, keep_last=4, tail_fraction=0.4)
    assert schedule[-1] == 10
    assert len(schedule) <= 4
    assert all(step >= 6 for step in schedule)


def test_find_beta_hits_target_ess_within_tolerance() -> None:
    scores = np.asarray([0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    target_ess = 2.5
    beta = find_beta_for_target_ess(scores, target_ess)
    achieved_ess = effective_sample_size(softmax_weights(scores, beta))
    assert achieved_ess >= target_ess - 0.05
    assert achieved_ess <= len(scores)


def test_threshold_weights_are_non_empty_and_normalized() -> None:
    scores = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    weights, threshold = threshold_weights(scores, 0.99)
    assert threshold >= 0.0
    assert np.isclose(weights.sum(), 1.0)
    assert np.count_nonzero(weights) >= 1


def test_aggregate_adapter_checkpoints_shapes_and_diagnostics() -> None:
    checkpoints = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [0.2, 1.1, 2.2],
            [0.4, 1.3, 2.4],
            [0.6, 1.5, 2.6],
        ],
        dtype=np.float32,
    )
    scores = np.asarray([0.1, 0.5, 0.7, 1.2], dtype=np.float32)
    result = aggregate_adapter_checkpoints(
        checkpoints=checkpoints,
        scores=scores,
        scheme="ess",
        beta=1.0,
        target_ess=2.0,
        threshold_quantile=0.75,
        low_rank_rank=3,
    )
    assert result.mean_vector.shape == (3,)
    assert result.weights.shape == (4,)
    assert result.retained_count == 4
    assert result.posterior_trace >= 0.0
    assert result.max_normalized_weight > 0.0
    assert len(result.top_eigenvalues) <= 3


def test_choose_factor_prefers_higher_composite() -> None:
    factor, score = choose_factor_from_factor_sweep(
        [
            {"factor": 0.5, "composite": 0.8, "instruction_relevance": 1.5, "fluency": 1.5},
            {"factor": 1.0, "composite": 1.1, "instruction_relevance": 1.2, "fluency": 1.0},
        ]
    )
    assert factor == 1.0
    assert score == 1.1


def test_choose_factor_handles_nan_perplexity() -> None:
    factor, score = choose_factor_from_factor_sweep(
        [
            {"factor": 0.5, "composite": 1.0, "instruction_relevance": 1.5, "fluency": 1.5, "perplexity": np.nan},
            {"factor": 1.0, "composite": 1.0, "instruction_relevance": 1.5, "fluency": 1.5, "perplexity": 5.0},
        ]
    )
    assert factor == 1.0
    assert score == 1.0


def test_attach_validation_metrics_uses_selected_factor() -> None:
    from meta_swag.axbench_meta_swag import RetainedCheckpoint

    record = RetainedCheckpoint(
        checkpoint_id="ckpt",
        step=10,
        epoch=0,
        train_loss=1.0,
        adapter_vector=np.zeros(3, dtype=np.float32),
        adapter_dimension=3,
    )
    updated = attach_validation_metrics(
        record,
        [
            {"factor": 0.5, "composite": 0.8, "instruction_relevance": 1.5, "fluency": 0.5},
            {"factor": 1.0, "composite": 1.2, "instruction_relevance": 2.0, "fluency": 1.0},
        ],
    )
    assert updated.selected_factor == 1.0
    assert updated.weighting_metric > 0.0
    assert len(updated.validation_factor_sweep) == 2


def test_select_concept_ids_filters_negative_concepts() -> None:
    class Args:
        concept_ids = None
        max_concepts = None

    df = pd.DataFrame({"concept_id": [-1, 0, 2, 2, 1]})
    assert select_concept_ids(df, Args()) == [0, 1, 2]
