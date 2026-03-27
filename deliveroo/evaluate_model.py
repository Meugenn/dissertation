"""Implementation of evaluate_model for the ML training pipeline."""

import math
import logging

from pipeline_utils.config import MODEL_PERF_THRESHOLD

logger = logging.getLogger(__name__)

# Small constant to avoid log(0)
_EPS = 1e-15


class ModelPerformanceError(Exception):
    """Raised when model log-loss exceeds the acceptable threshold."""
    pass


def _clip(value: float, low: float = _EPS, high: float = 1.0 - _EPS) -> float:
    """Clip a value to [low, high] to ensure numerical stability in log computation."""
    return max(low, min(high, value))


def _log_loss(y_true: list[int], y_pred: list[float]) -> float:
    """Compute binary cross-entropy (log-loss).

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_pred: Predicted probabilities for the positive class, each in [0, 1].

    Returns:
        Mean log-loss across all samples.
    """
    n = len(y_true)
    total = 0.0
    for label, prob in zip(y_true, y_pred):
        p = _clip(prob)
        total += label * math.log(p) + (1 - label) * math.log(1 - p)
    return -total / n


def evaluate_model(y_test: list[int], y_pred: list[float]) -> float:
    """Evaluate model predictions and enforce the performance threshold.

    Computes binary log-loss between ground-truth labels and predicted
    probabilities. Raises an error if the loss exceeds the configured
    threshold, preventing a poor model from being published to the store.

    Args:
        y_test: Ground-truth binary labels (0 or 1).
        y_pred: Predicted probabilities for the positive class, each in [0, 1].

    Returns:
        The computed log-loss value.

    Raises:
        ValueError: If inputs are empty or have mismatched lengths.
        ModelPerformanceError: If log-loss exceeds MODEL_PERF_THRESHOLD.
    """
    if len(y_test) == 0:
        raise ValueError("y_test must not be empty")
    if len(y_test) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_test has {len(y_test)} samples, "
            f"y_pred has {len(y_pred)}"
        )

    loss = _log_loss(y_test, y_pred)

    logger.info("Log-loss: %.6f (threshold: %.6f)", loss, MODEL_PERF_THRESHOLD)

    if loss > MODEL_PERF_THRESHOLD:
        raise ModelPerformanceError(
            f"Model log-loss {loss:.6f} exceeds threshold "
            f"{MODEL_PERF_THRESHOLD:.6f}"
        )

    return loss
