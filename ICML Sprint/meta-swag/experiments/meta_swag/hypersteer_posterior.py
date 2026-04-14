from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import numpy as np
import torch

from .adapter_posterior import build_retention_schedule
from .adapter_state import AdapterStateManifest, build_manifest, flatten_adapter_state
from .axbench_meta_swag import RetainedCheckpoint, distributed_rank, weighting_metric_from_row


def attach_multi_concept_validation_metrics(
    record: RetainedCheckpoint,
    concept_summaries: list[dict[str, float]],
) -> RetainedCheckpoint:
    if not concept_summaries:
        raise ValueError("concept_summaries cannot be empty.")
    weighting_metrics = [float(summary["weighting_metric"]) for summary in concept_summaries]
    selected_factors = [float(summary["selected_factor"]) for summary in concept_summaries]
    record.weighting_metric = float(np.mean(weighting_metrics))
    record.selected_factor = float(np.mean(selected_factors))
    record.validation_factor_sweep = concept_summaries
    return record


def aggregate_sampled_factor_rows(
    sampled_factor_rows: list[list[dict[str, float]]],
) -> list[dict[str, float]]:
    if not sampled_factor_rows:
        raise ValueError("sampled_factor_rows cannot be empty.")

    grouped: dict[float, list[dict[str, float]]] = defaultdict(list)
    for sample_rows in sampled_factor_rows:
        for row in sample_rows:
            grouped[float(row["factor"])].append(row)

    summary_rows: list[dict[str, float]] = []
    for factor, rows in sorted(grouped.items()):
        summary: dict[str, float] = {"factor": factor}
        for key in ["composite", "concept_relevance", "instruction_relevance", "fluency", "perplexity"]:
            values = [
                float(row[key])
                for row in rows
                if key in row and row[key] is not None and not np.isnan(float(row[key]))
            ]
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))
            else:
                summary[f"{key}_mean"] = float("nan")
                summary[f"{key}_std"] = float("nan")
        summary_rows.append(summary)
    return summary_rows


def choose_risk_sensitive_factor(
    factor_statistics: list[dict[str, float]],
    risk_aversion: float = 0.0,
) -> tuple[float, float]:
    if not factor_statistics:
        raise ValueError("factor_statistics cannot be empty.")

    def sort_key(row: dict[str, float]) -> tuple[float, float, float, float]:
        composite_mean = float(row.get("composite_mean", 0.0))
        composite_std = float(row.get("composite_std", 0.0))
        instruction_mean = float(row.get("instruction_relevance_mean", 0.0))
        fluency_mean = float(row.get("fluency_mean", 0.0))
        perplexity_mean = row.get("perplexity_mean", float("inf"))
        if perplexity_mean is None or np.isnan(float(perplexity_mean)):
            perplexity_mean = float("inf")
        return (
            composite_mean - risk_aversion * composite_std,
            instruction_mean,
            fluency_mean,
            -float(perplexity_mean),
        )

    best_row = max(factor_statistics, key=sort_key)
    return float(best_row["factor"]), float(best_row["composite_mean"] - risk_aversion * best_row["composite_std"])


def _capture_hypersteer_checkpoint(
    records: list[RetainedCheckpoint],
    checkpoint_id_prefix: str,
    step: int,
    epoch: int,
    train_loss: float,
    module: torch.nn.Module,
    manifest: AdapterStateManifest,
) -> None:
    flat_vector, _ = flatten_adapter_state(module, manifest)
    records.append(
        RetainedCheckpoint(
            checkpoint_id=f"{checkpoint_id_prefix}_step_{step:05d}",
            step=step,
            epoch=epoch,
            train_loss=float(train_loss),
            adapter_vector=flat_vector,
            adapter_dimension=int(flat_vector.size),
        )
    )


def train_hypersteer_with_retention(
    model,
    examples,
    keep_last: int,
    tail_fraction: float,
    checkpoint_id_prefix: str,
    **kwargs,
) -> tuple[list[RetainedCheckpoint], AdapterStateManifest]:
    from tqdm.auto import tqdm
    from transformers import get_scheduler

    train_dataloader = model.make_dataloader(
        examples,
        rank=0,
        world_size=1,
        distributed=False,
        concept_tokenizer=model.hypernet_tokenizer,
        **kwargs,
    )
    optimizer = torch.optim.AdamW(
        model.concept_embedding.parameters(),
        lr=model.training_args.lr,
        weight_decay=model.training_args.weight_decay,
    )
    grad_accum = max(1, int(model.training_args.gradient_accumulation_steps))
    num_training_steps = max(1, model.training_args.n_epochs * math.ceil(len(train_dataloader) / grad_accum))
    retention_steps = set(build_retention_schedule(num_training_steps, keep_last, tail_fraction))
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    manifest = build_manifest(model.concept_embedding)
    retained: list[RetainedCheckpoint] = []
    rank = distributed_rank()
    progress = tqdm(range(num_training_steps), position=rank, leave=True)
    current_step = 0

    model.concept_embedding.train()
    model.ax.train()

    for epoch in range(model.training_args.n_epochs):
        for batch_index, batch in enumerate(train_dataloader):
            inputs = {key: value.to(model.device) for key, value in batch.items()}

            unit_locations = {
                "sources->base": (
                    None,
                    inputs["intervention_locations"].permute(1, 0, 2).tolist(),
                )
            }
            subspaces = [{"k": model.training_args.topk}]

            base_intervention_mask = (inputs["labels"] == -100) & inputs["attention_mask"].bool()
            base_hidden_state = model.model(
                input_ids=inputs["input_ids"],
                attention_mask=base_intervention_mask,
                output_hidden_states=True,
            ).hidden_states[model.layer]

            steering_vectors = model.concept_embedding(
                input_ids=inputs["concept_input_ids"],
                inputs_embeds=None,
                attention_mask=inputs["concept_attention_mask"],
                base_encoder_hidden_states=base_hidden_state,
                base_encoder_attention_mask=base_intervention_mask,
                output_hidden_states=False,
            ).last_hidden_state

            model.ax._update_v(steering_vectors)
            _, cf_outputs = model.ax_model(
                base={
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                },
                unit_locations=unit_locations,
                labels=inputs["labels"],
                subspaces=subspaces,
                use_cache=False,
            )

            loss = cf_outputs.loss.mean()
            (loss / grad_accum).backward()
            model.ax._reset_v()

            should_step = (batch_index + 1) % grad_accum == 0 or (batch_index + 1) == len(train_dataloader)
            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(model.concept_embedding.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            current_step += 1
            progress.update(1)
            progress.set_description(f"lr {optimizer.param_groups[0]['lr']:.6f} || loss {float(loss):.6f}")

            if current_step in retention_steps:
                _capture_hypersteer_checkpoint(
                    retained,
                    checkpoint_id_prefix=checkpoint_id_prefix,
                    step=current_step,
                    epoch=epoch,
                    train_loss=float(loss.detach().cpu()),
                    module=model.concept_embedding,
                    manifest=manifest,
                )

    progress.close()
    return retained, manifest


def summarize_checkpoint_concept_rows(
    concept_id: int,
    concept_name: str,
    factor_rows: list[dict[str, float]],
) -> dict[str, Any]:
    if not factor_rows:
        raise ValueError("factor_rows cannot be empty.")
    best_row = max(
        factor_rows,
        key=lambda row: (
            float(row.get("composite", 0.0)),
            float(row.get("instruction_relevance", 0.0)),
            float(row.get("fluency", 0.0)),
        ),
    )
    return {
        "concept_id": int(concept_id),
        "concept": concept_name,
        "selected_factor": float(best_row["factor"]),
        "validation_composite": float(best_row["composite"]),
        "weighting_metric": float(weighting_metric_from_row(best_row)),
    }
