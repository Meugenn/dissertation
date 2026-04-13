from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np


KIM_REPO_ROOT = Path("/Users/meuge/coding/maynard/ICML Sprint/meta-swag/external/meta-mapg")


@dataclass(frozen=True)
class PersonaBundle:
    env_name: str
    split: str
    persona_group: str
    personas: list[np.ndarray]


def _load_pickle(path: Path) -> list[np.ndarray]:
    with path.open("rb") as handle:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
                category=Warning,
            )
            data = pickle.load(handle)
    return [np.asarray(item, dtype=float) for item in data]


def load_ipd_personas(split: str = "test") -> dict[str, PersonaBundle]:
    base = KIM_REPO_ROOT / "pretrain_model" / "IPD-v0"
    bundles = {}
    for group in ("cooperative", "defective"):
        path = base / group / split
        personas = _load_pickle(path)
        bundles[group] = PersonaBundle("IPD-v0", split, group, personas)
    return bundles


def load_rps_personas(split: str = "test") -> dict[str, PersonaBundle]:
    base = KIM_REPO_ROOT / "pretrain_model" / "RPS-v0"
    bundles = {}
    for group in ("rock", "paper", "scissors"):
        path = base / group / split
        personas = _load_pickle(path)
        bundles[group] = PersonaBundle("RPS-v0", split, group, personas)
    return bundles


def policy_from_persona_logits(persona: np.ndarray) -> np.ndarray:
    shifted = persona - np.max(persona, axis=-1, keepdims=True)
    probs = np.exp(shifted)
    return probs / probs.sum(axis=-1, keepdims=True)


def summarize_persona_bundle(bundle: PersonaBundle) -> dict[str, float | str]:
    stacked = np.stack([policy_from_persona_logits(persona) for persona in bundle.personas], axis=0)
    dominant_action_share = np.mean(np.max(stacked, axis=-1))
    entropy = -np.sum(stacked * np.log(np.clip(stacked, 1e-12, None)), axis=-1).mean()
    return {
        "env_name": bundle.env_name,
        "split": bundle.split,
        "persona_group": bundle.persona_group,
        "num_personas": len(bundle.personas),
        "state_count": int(stacked.shape[1]),
        "action_count": int(stacked.shape[2]),
        "mean_dominant_action_prob": float(dominant_action_share),
        "mean_entropy": float(entropy),
    }
