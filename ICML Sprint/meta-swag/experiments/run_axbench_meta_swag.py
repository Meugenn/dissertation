from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import re
from typing import Any

import numpy as np
import pandas as pd
import torch

from meta_swag.adapter_state import restore_adapter_state, save_manifest
from meta_swag.axbench_meta_swag import (
    FinalMethodResult,
    aggregate_checkpoint_records,
    attach_validation_metrics,
    choose_factor_from_factor_sweep,
    harmonic_mean,
    split_validation_test,
    train_lora_with_retention,
    train_preference_lora_with_retention,
)
from meta_swag.axbench_runtime import describe_external_repo, import_alpaca_eval, import_axbench


DEFAULT_FACTORS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]
DEFAULT_METHODS = ["map", "uniform", "softmax", "ess", "threshold"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Meta-SWAG benchmark runner for AxBench and AlpacaEval.")
    parser.add_argument("--output-dir", required=True, help="Directory for benchmark artifacts.")
    parser.add_argument("--model-kind", choices=["lora", "preference_lora"], default="lora")
    parser.add_argument("--model-name", default="google/gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--component", default="res")
    parser.add_argument("--train-data-path", required=True, help="AxBench training parquet.")
    parser.add_argument("--metadata-path", required=True, help="AxBench metadata.jsonl.")
    parser.add_argument("--steering-data-path", help="Optional pre-generated steering_data.parquet.")
    parser.add_argument("--alpacaeval-inputs-path", help="Optional CSV/JSONL with instruction prompts for transfer.")
    parser.add_argument("--max-concepts", type=int, default=None)
    parser.add_argument("--concept-ids", nargs="+", type=int, default=None)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--low-rank-dimension", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-components", nargs="+", default=["q_proj"])
    parser.add_argument("--lora-layers", nargs="+", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-output-length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--steering-factors", nargs="+", type=float, default=DEFAULT_FACTORS)
    parser.add_argument("--keep-last", type=int, default=20)
    parser.add_argument("--tail-fraction", type=float, default=0.4)
    parser.add_argument("--threshold-quantile", type=float, default=0.75)
    parser.add_argument("--validation-ratio", type=float, default=0.5)
    parser.add_argument("--max-validation-examples", type=int, default=32)
    parser.add_argument("--max-test-examples", type=int, default=32)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--mock-judge", action="store_true")
    parser.add_argument("--skip-alpacaeval", action="store_true")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gemma", type=float, default=0.0)
    parser.add_argument("--simpo-scaler", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--reference-free", action="store_true")
    parser.add_argument("--loss-type", default="dpo")
    parser.add_argument("--preference-pairs", nargs="+", default=["orig_add"])
    parser.add_argument("--steering-prompt-type", default="prepend")
    parser.add_argument("--substraction-type", default="null_it_out")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_metadata(metadata_path: str | Path) -> list[dict[str, Any]]:
    metadata = []
    with Path(metadata_path).open() as handle:
        for line in handle:
            metadata.append(json.loads(line))
    return metadata


def load_dataframe(path: str | Path) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"Unsupported data format for {path}.")


def select_concept_ids(train_df: pd.DataFrame, args: argparse.Namespace) -> list[int]:
    if args.concept_ids:
        return [int(concept_id) for concept_id in args.concept_ids]
    concept_ids = sorted(
        int(value)
        for value in train_df["concept_id"].unique()
        if int(value) >= 0
    )
    if args.max_concepts is not None:
        concept_ids = concept_ids[: args.max_concepts]
    return concept_ids


def build_model_params(axbench, args: argparse.Namespace):
    params = axbench.ModelParams()
    params.batch_size = args.batch_size
    params.n_epochs = args.n_epochs
    params.lr = args.lr
    params.dropout = args.dropout
    params.low_rank_dimension = args.low_rank_dimension
    params.gradient_accumulation_steps = args.gradient_accumulation_steps
    params.lora_layers = args.lora_layers
    params.lora_components = args.lora_components
    params.lora_alpha = args.lora_alpha
    params.weight_decay = args.weight_decay
    params.topk = args.topk
    params.loss_type = args.loss_type
    params.beta = args.beta
    params.gemma = args.gemma
    params.reference_free = args.reference_free
    params.label_smoothing = args.label_smoothing
    params.steering_factors = list(args.steering_factors)
    params.simpo_scaler = args.simpo_scaler
    params.preference_pairs = list(args.preference_pairs)
    params.steering_prompt_type = args.steering_prompt_type
    params.substraction_type = args.substraction_type
    params.intervention_positions = "all"
    return params


def build_training_dataframe(axbench_train_module, train_df, negative_df, metadata_entry, concept: str, tokenizer, args, model_params):
    prepared = train_df.copy()
    is_chat_model = True if args.model_name in axbench_train_module.CHAT_MODELS else False
    return axbench_train_module.prepare_df(
        prepared,
        negative_df,
        concept,
        metadata_entry,
        tokenizer,
        binarize=model_params.binarize_dataset,
        train_on_negative=model_params.train_on_negative,
        is_chat_model=is_chat_model,
        output_length=args.eval_output_length,
        model_name=args.model_name,
        max_num_of_examples=None,
        use_dpo_loss=args.model_kind == "preference_lora",
        steering_prompt_type=model_params.steering_prompt_type,
        keep_orig_axbench_format=True,
    )


def build_fallback_steering_df(
    concept_df: pd.DataFrame,
    metadata_entry: dict[str, Any],
    factors: list[float],
    max_examples: int,
) -> pd.DataFrame:
    base_rows = concept_df[concept_df["category"] == "positive"].copy()
    if base_rows.empty:
        base_rows = concept_df.copy()
    base_rows = base_rows.head(max_examples).copy()
    if "input" not in base_rows.columns:
        raise KeyError("Fallback steering data requires an input column.")
    base_rows["original_prompt"] = base_rows["input"]
    base_rows["input_concept"] = metadata_entry["concept"]
    base_rows["dataset_name"] = "TrainFallback"
    base_rows["input_id"] = np.arange(len(base_rows))

    repeated = []
    for factor in factors:
        current = base_rows[["concept_id", "input", "original_prompt", "input_concept", "dataset_name", "input_id"]].copy()
        current["factor"] = factor
        repeated.append(current)
    return pd.concat(repeated, ignore_index=True)


def load_concept_steering_df(
    concept_df: pd.DataFrame,
    metadata_entry: dict[str, Any],
    concept_id: int,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if args.steering_data_path:
        full_df = load_dataframe(args.steering_data_path)
        filtered = full_df[full_df["concept_id"] == concept_id].copy()
        if not filtered.empty:
            return filtered
    return build_fallback_steering_df(
        concept_df=concept_df,
        metadata_entry=metadata_entry,
        factors=list(args.steering_factors),
        max_examples=max(args.max_validation_examples, args.max_test_examples),
    )


def build_prompt_eval_df(
    prompts_df: pd.DataFrame,
    concept_id: int,
    concept_name: str,
    factor: float,
) -> pd.DataFrame:
    working = prompts_df.copy()
    if "instruction" not in working.columns:
        raise KeyError("Prompt dataframe must contain an instruction column.")
    working["input"] = working["instruction"]
    working["original_prompt"] = working["instruction"]
    working["input_concept"] = concept_name
    working["dataset_name"] = "AlpacaEvalTransfer"
    working["concept_id"] = concept_id
    working["input_id"] = np.arange(len(working))
    working["factor"] = factor
    return working


def compute_perplexities(base_model, tokenizer, generated_sequences: list[str], device: torch.device) -> list[float]:
    if not generated_sequences:
        return []
    encoded = tokenizer(generated_sequences, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()
    mask = attention_mask[:, 1:].contiguous()
    loss = torch.nn.CrossEntropyLoss(reduction="none")
    token_losses = loss(logits.view(-1, logits.size(-1)), targets.view(-1)).view(targets.size(0), -1)
    seq_losses = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return torch.exp(seq_losses).detach().cpu().tolist()


def generate_unsteered_outputs(
    base_model,
    tokenizer,
    evaluation_df: pd.DataFrame,
    batch_size: int,
    output_length: int,
    temperature: float,
    device: torch.device,
) -> dict[str, list[Any]]:
    tokenizer.padding_side = "left"
    all_generations: list[str] = []
    all_perplexities: list[float] = []
    for start in range(0, len(evaluation_df), batch_size):
        batch = evaluation_df.iloc[start : start + batch_size]
        inputs = tokenizer(batch["input"].tolist(), return_tensors="pt", padding=True, truncation=True).to(device)
        generations = base_model.generate(
            **inputs,
            max_new_tokens=output_length,
            do_sample=True,
            temperature=temperature,
        )
        input_lengths = [len(row) for row in inputs.input_ids]
        generated = [
            tokenizer.decode(sequence[input_length:], skip_special_tokens=True)
            for sequence, input_length in zip(generations, input_lengths)
        ]
        all_generations.extend(generated)
        decoded_full = [tokenizer.decode(sequence, skip_special_tokens=True) for sequence in generations]
        all_perplexities.extend(compute_perplexities(base_model, tokenizer, decoded_full, device))
    return {
        "steered_generation": all_generations,
        "perplexity": all_perplexities,
    }


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def evaluate_mock_factor_sweep(results_df: pd.DataFrame, model_name: str) -> list[dict[str, float]]:
    scored = results_df.copy()

    def concept_score(row) -> float:
        concept = row["input_concept"].lower()
        generation = row[f"{model_name}_steered_generation"].lower()
        if concept in generation:
            return 2.0
        concept_tokens = [token for token in _tokenize_words(concept) if len(token) > 2]
        if any(token in generation for token in concept_tokens):
            return 1.0
        return 0.0

    def instruction_score(row) -> float:
        prompt_tokens = {token for token in _tokenize_words(row["original_prompt"]) if len(token) > 3}
        generation_tokens = set(_tokenize_words(row[f"{model_name}_steered_generation"]))
        overlap = len(prompt_tokens & generation_tokens)
        if overlap >= 3:
            return 2.0
        if overlap >= 1 or generation_tokens:
            return 1.0
        return 0.0

    def fluency_score(row) -> float:
        generation = row[f"{model_name}_steered_generation"]
        if len(generation.strip()) < 4:
            return 0.0
        ascii_ratio = sum(character.isascii() and (character.isalnum() or character.isspace()) for character in generation) / max(len(generation), 1)
        if ascii_ratio > 0.85 and len(generation.split()) >= 4:
            return 2.0
        return 1.0

    scored[f"{model_name}_concept"] = scored.apply(concept_score, axis=1)
    scored[f"{model_name}_instruction"] = scored.apply(instruction_score, axis=1)
    scored[f"{model_name}_fluency"] = scored.apply(fluency_score, axis=1)

    factor_rows = []
    for factor, group in scored.groupby("factor"):
        concept_relevance = float(group[f"{model_name}_concept"].mean())
        instruction_relevance = float(group[f"{model_name}_instruction"].mean())
        fluency = float(group[f"{model_name}_fluency"].mean())
        composite = harmonic_mean([concept_relevance, instruction_relevance, fluency])
        perplexity = float(group[f"{model_name}_perplexity"].mean()) if f"{model_name}_perplexity" in group.columns else None
        factor_rows.append(
            {
                "factor": float(factor),
                "composite": composite,
                "concept_relevance": concept_relevance,
                "instruction_relevance": instruction_relevance,
                "fluency": fluency,
                "perplexity": perplexity if perplexity is not None else np.nan,
            }
        )
    return sorted(factor_rows, key=lambda row: row["factor"])


def build_language_model(axbench, judge_model: str, output_dir: Path):
    from openai import AsyncOpenAI
    import httpx

    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=60.0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=1000),
            headers={"Connection": "close"},
        ),
        max_retries=3,
    )
    return client, axbench.LanguageModel(
        judge_model,
        client,
        dump_dir=output_dir,
        use_cache=True,
        cache_level="prompt",
        cache_tag="meta_swag",
        master_data_dir=str(output_dir),
        temperature=0.0,
    )


def evaluate_real_factor_sweep(axbench, results_df: pd.DataFrame, model_name: str, concept_id: int, lm_model) -> list[dict[str, float]]:
    judge = axbench.LMJudgeEvaluator(
        model_name,
        lm_model=lm_model,
        concept_id=concept_id,
        steer_dataset_type="concept",
    )
    metrics = judge.compute_metrics(results_df)

    perplexity_metrics = {}
    if f"{model_name}_perplexity" in results_df.columns:
        perplexity_metrics = axbench.PerplexityEvaluator(model_name).compute_metrics(results_df)

    factor_to_perplexity = {
        float(factor): float(perplexity)
        for factor, perplexity in zip(
            perplexity_metrics.get("factor", []),
            perplexity_metrics.get("perplexity", []),
        )
    }

    rows = []
    for index, factor in enumerate(metrics["factor"]):
        factor_value = float(factor)
        rows.append(
            {
                "factor": factor_value,
                "composite": float(metrics["lm_judge_rating"][index]),
                "concept_relevance": float(metrics["relevance_concept_ratings"][index]),
                "instruction_relevance": float(metrics["relevance_instruction_ratings"][index]),
                "fluency": float(metrics["fluency_ratings"][index]),
                "perplexity": factor_to_perplexity.get(factor_value, np.nan),
            }
        )
    return rows


def evaluate_factor_sweep(
    model,
    evaluation_df: pd.DataFrame,
    model_name: str,
    axbench,
    args: argparse.Namespace,
    concept_id: int,
    lm_model=None,
) -> tuple[list[dict[str, float]], pd.DataFrame]:
    working_df = evaluation_df.copy()
    results = model.predict_steer(
        working_df,
        concept_id=concept_id,
        batch_size=args.eval_batch_size,
        eval_output_length=args.eval_output_length,
        temperature=args.temperature,
        prefix_length=1,
        positions="all",
        use_synergy=False,
        disable_neuronpedia_max_act=True,
        intervene_on_prompt=True,
        return_vector=False,
    )
    for key, values in results.items():
        working_df[f"{model_name}_{key}"] = values

    if args.mock_judge or lm_model is None:
        factor_rows = evaluate_mock_factor_sweep(working_df, model_name)
    else:
        factor_rows = evaluate_real_factor_sweep(axbench, working_df, model_name, concept_id, lm_model)
    return factor_rows, working_df


def restore_record(model, record, manifest) -> None:
    restore_adapter_state(model.ax_model, record.adapter_vector, manifest)


def restore_aggregated_result(model, aggregation_result, manifest) -> None:
    restore_adapter_state(model.ax_model, aggregation_result.mean_vector, manifest)


def summarize_method(
    scheme: str,
    aggregation_result,
    validation_rows: list[dict[str, float]],
    test_rows: list[dict[str, float]],
    unsteered_test_composite: float,
) -> FinalMethodResult:
    selected_factor, validation_composite = choose_factor_from_factor_sweep(validation_rows)
    chosen_test_row = next(row for row in test_rows if float(row["factor"]) == float(selected_factor))
    diagnostics = {
        "retained_count": float(aggregation_result.retained_count),
        "ess": float(aggregation_result.effective_sample_size),
        "max_normalized_weight": float(aggregation_result.max_normalized_weight),
        "posterior_trace": float(aggregation_result.posterior_trace),
        "top_eigenvalue_ratio": float(aggregation_result.top_eigenvalue_ratio),
        "score_variance": float(aggregation_result.score_variance),
    }
    for index, value in enumerate(aggregation_result.top_eigenvalues, start=1):
        diagnostics[f"top_eigenvalue_{index}"] = float(value)

    return FinalMethodResult(
        scheme=scheme,
        selected_factor=float(selected_factor),
        validation_composite=float(validation_composite),
        test_composite=float(chosen_test_row["composite"]),
        concept_relevance=float(chosen_test_row["concept_relevance"]),
        instruction_relevance=float(chosen_test_row["instruction_relevance"]),
        fluency=float(chosen_test_row["fluency"]),
        perplexity=None if pd.isna(chosen_test_row["perplexity"]) else float(chosen_test_row["perplexity"]),
        delta_over_unsteered=float(chosen_test_row["composite"] - unsteered_test_composite),
        diagnostics=diagnostics,
    )


def run_alpacaeval_transfer(
    output_dir: Path,
    promoted_methods: list[str],
    concept_results: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    if not concept_results:
        return {"status": "skipped", "reason": "alpacaeval_inputs_path_not_provided"}

    alpaca_eval = import_alpaca_eval()
    summary: dict[str, Any] = {"status": "completed", "leaderboards": {}}

    for method_name in promoted_methods:
        if method_name not in concept_results:
            continue
        method_df = concept_results[method_name].copy()
        if "output" not in method_df.columns:
            return {"status": "skipped", "reason": f"{method_name}_outputs_missing_output_column"}
        leaderboard, _ = alpaca_eval.evaluate(
            model_outputs=method_df[["instruction", "output"]],
            name=method_name,
            output_path=str(output_dir / "alpacaeval" / method_name),
            is_return_instead_of_print=True,
            max_instances=len(method_df),
        )
        summary["leaderboards"][method_name] = leaderboard.reset_index().to_dict("records")
    return summary


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.base_seed)
    axbench = import_axbench()
    import axbench.scripts.train as axbench_train_module  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer

    metadata = load_metadata(args.metadata_path)
    metadata_by_concept_id = {int(entry["concept_id"]): entry for entry in metadata}
    train_df = load_dataframe(args.train_data_path)
    concept_ids = [
        concept_id
        for concept_id in select_concept_ids(train_df, args)
        if concept_id in metadata_by_concept_id
    ]

    dependency_manifest = {
        "axbench": describe_external_repo("axbench").as_json(),
        "alpaca_eval": describe_external_repo("alpaca_eval").as_json(),
    }
    write_json(output_dir / "dependency_manifest.json", dependency_manifest)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=512)
    tokenizer.padding_side = "right"
    original_vocab_size = len(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(args.use_bf16 and torch.cuda.is_available())
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if use_bf16 else None,
    ).eval().to(device)
    if len(tokenizer) != original_vocab_size:
        base_model.resize_token_embeddings(len(tokenizer))

    language_model_client = None
    language_model = None
    if not args.mock_judge:
        try:
            language_model_client, language_model = build_language_model(axbench, args.judge_model, output_dir)
        except Exception as exc:  # pragma: no cover - environment specific
            print(f"Falling back to mock judge because real judge setup failed: {exc}")
            args.mock_judge = True
            language_model_client, language_model = None, None

    model_params = build_model_params(axbench, args)
    negative_df = train_df[train_df["category"] == "negative"].copy() if "category" in train_df.columns else train_df.iloc[0:0].copy()
    alpaca_prompts_df = None
    if not args.skip_alpacaeval and args.alpacaeval_inputs_path:
        alpaca_prompts_df = load_dataframe(args.alpacaeval_inputs_path)

    checkpoint_rows: list[dict[str, Any]] = []
    factor_sweep_rows: list[dict[str, Any]] = []
    final_summary_rows: list[dict[str, Any]] = []
    concept_method_outputs: dict[str, list[pd.DataFrame]] = defaultdict(list)

    for seed_index in range(args.seed_count):
        seed = args.base_seed + seed_index
        set_global_seed(seed)

        for concept_id in concept_ids:
            metadata_entry = metadata_by_concept_id[concept_id]
            concept_name = metadata_entry["concept"]
            concept_train_df = train_df[train_df["concept_id"] == concept_id].copy()
            steering_df = load_concept_steering_df(concept_train_df, metadata_entry, concept_id, args)
            validation_df, test_df = split_validation_test(steering_df, validation_ratio=args.validation_ratio)
            validation_df = validation_df.head(args.max_validation_examples * max(1, len(args.steering_factors))).copy()
            test_df = test_df.head(args.max_test_examples * max(1, len(args.steering_factors))).copy()

            prepared_train_df = build_training_dataframe(
                axbench_train_module=axbench_train_module,
                train_df=concept_train_df,
                negative_df=negative_df,
                metadata_entry=metadata_entry,
                concept=concept_name,
                tokenizer=tokenizer,
                args=args,
                model_params=model_params,
            )

            model_class = axbench.PreferenceLoRA if args.model_kind == "preference_lora" else axbench.LoRA
            model = model_class(
                base_model,
                tokenizer,
                layer=args.layer,
                training_args=model_params,
                lm_model_name=args.model_name,
                device=device,
                seed=seed,
            )
            model.make_model(
                mode="train",
                embed_dim=base_model.config.hidden_size,
                low_rank_dimension=args.low_rank_dimension,
                concept_id=concept_id,
                dtype=torch.bfloat16 if use_bf16 else None,
                intervention_type="addition",
                metadata_path=args.metadata_path,
                dump_dir=str(output_dir),
                model_params=model_params,
                dropout=args.dropout,
                intervention_positions_dropout=0.0,
                preference_pairs=args.preference_pairs,
            )

            trainer_kwargs = {
                "prefix_length": 1,
                "positions": "all",
                "exclude_bos": True,
                "metadata_path": args.metadata_path,
                "use_dpo_loss": args.model_kind == "preference_lora",
                "logging_metadata": {
                    "concept_id": concept_id,
                    "model_name": args.model_kind,
                    "layer": args.layer,
                },
                "negative_only": False,
                "preference_pairs": args.preference_pairs,
                "steering_prompt_type": args.steering_prompt_type,
                "substraction_type": args.substraction_type,
            }

            if args.model_kind == "preference_lora":
                retained_records, manifest = train_preference_lora_with_retention(
                    model,
                    prepared_train_df,
                    keep_last=args.keep_last,
                    tail_fraction=args.tail_fraction,
                    checkpoint_id_prefix=f"seed_{seed}_concept_{concept_id}",
                    **trainer_kwargs,
                )
            else:
                retained_records, manifest = train_lora_with_retention(
                    model,
                    prepared_train_df,
                    keep_last=args.keep_last,
                    tail_fraction=args.tail_fraction,
                    checkpoint_id_prefix=f"seed_{seed}_concept_{concept_id}",
                    **trainer_kwargs,
                )

            concept_dir = output_dir / f"seed_{seed}" / f"concept_{concept_id}"
            concept_dir.mkdir(parents=True, exist_ok=True)
            save_manifest(manifest, concept_dir / "adapter_manifest.json")
            concept_checkpoint_rows: list[dict[str, Any]] = []
            concept_factor_sweep_rows: list[dict[str, Any]] = []

            for record in retained_records:
                restore_record(model, record, manifest)
                validation_rows, _ = evaluate_factor_sweep(
                    model,
                    validation_df,
                    model_name="checkpoint",
                    axbench=axbench,
                    args=args,
                    concept_id=concept_id,
                    lm_model=language_model,
                )
                attach_validation_metrics(record, validation_rows)
                checkpoint_rows.append(
                    {
                        "seed": seed,
                        "concept_id": concept_id,
                        "concept": concept_name,
                        **record.metadata(),
                    }
                )
                concept_checkpoint_rows.append(
                    {
                        "seed": seed,
                        "concept_id": concept_id,
                        "concept": concept_name,
                        **record.metadata(),
                    }
                )
                for row in validation_rows:
                    payload = {
                        "seed": seed,
                        "concept_id": concept_id,
                        "concept": concept_name,
                        "partition": "validation_checkpoint",
                        "scheme": record.checkpoint_id,
                        **row,
                    }
                    factor_sweep_rows.append(payload)
                    concept_factor_sweep_rows.append(payload)

            unsteered_eval_df = validation_df.copy()
            unsteered_eval_df["factor"] = 0.0
            unsteered_results = generate_unsteered_outputs(
                base_model=base_model,
                tokenizer=tokenizer,
                evaluation_df=unsteered_eval_df,
                batch_size=args.eval_batch_size,
                output_length=args.eval_output_length,
                temperature=args.temperature,
                device=device,
            )
            for key, values in unsteered_results.items():
                unsteered_eval_df[f"unsteered_{key}"] = values
            unsteered_validation_rows = (
                evaluate_mock_factor_sweep(unsteered_eval_df, "unsteered")
                if args.mock_judge or language_model is None
                else evaluate_real_factor_sweep(axbench, unsteered_eval_df, "unsteered", concept_id, language_model)
            )

            unsteered_test_df = test_df.copy()
            unsteered_test_df["factor"] = 0.0
            unsteered_test_results = generate_unsteered_outputs(
                base_model=base_model,
                tokenizer=tokenizer,
                evaluation_df=unsteered_test_df,
                batch_size=args.eval_batch_size,
                output_length=args.eval_output_length,
                temperature=args.temperature,
                device=device,
            )
            for key, values in unsteered_test_results.items():
                unsteered_test_df[f"unsteered_{key}"] = values
            unsteered_test_rows = (
                evaluate_mock_factor_sweep(unsteered_test_df, "unsteered")
                if args.mock_judge or language_model is None
                else evaluate_real_factor_sweep(axbench, unsteered_test_df, "unsteered", concept_id, language_model)
            )
            _, unsteered_test_composite = choose_factor_from_factor_sweep(unsteered_test_rows)

            if alpaca_prompts_df is not None:
                alpaca_unsteered_df = build_prompt_eval_df(
                    prompts_df=alpaca_prompts_df,
                    concept_id=concept_id,
                    concept_name=concept_name,
                    factor=0.0,
                )
                alpaca_unsteered = generate_unsteered_outputs(
                    base_model=base_model,
                    tokenizer=tokenizer,
                    evaluation_df=alpaca_unsteered_df,
                    batch_size=args.eval_batch_size,
                    output_length=args.eval_output_length,
                    temperature=args.temperature,
                    device=device,
                )
                concept_method_outputs["unsteered"].append(
                    alpaca_unsteered_df.assign(output=alpaca_unsteered["steered_generation"])[["instruction", "output"]]
                )
            for scheme in args.methods:
                aggregation_result = aggregate_checkpoint_records(
                    retained_records,
                    scheme=scheme,
                    beta=1.0,
                    threshold_quantile=args.threshold_quantile,
                    low_rank_rank=min(args.keep_last, 20),
                )
                restore_aggregated_result(model, aggregation_result, manifest)
                validation_rows, _ = evaluate_factor_sweep(
                    model,
                    validation_df,
                    model_name=scheme,
                    axbench=axbench,
                    args=args,
                    concept_id=concept_id,
                    lm_model=language_model,
                )
                test_rows, test_results_df = evaluate_factor_sweep(
                    model,
                    test_df,
                    model_name=scheme,
                    axbench=axbench,
                    args=args,
                    concept_id=concept_id,
                    lm_model=language_model,
                )
                for row in validation_rows:
                    payload = {
                        "seed": seed,
                        "concept_id": concept_id,
                        "concept": concept_name,
                        "partition": "validation_method",
                        "scheme": scheme,
                        **row,
                    }
                    factor_sweep_rows.append(payload)
                    concept_factor_sweep_rows.append(payload)
                for row in test_rows:
                    payload = {
                        "seed": seed,
                        "concept_id": concept_id,
                        "concept": concept_name,
                        "partition": "test_method",
                        "scheme": scheme,
                        **row,
                    }
                    factor_sweep_rows.append(payload)
                    concept_factor_sweep_rows.append(payload)

                summary = summarize_method(
                    scheme=scheme,
                    aggregation_result=aggregation_result,
                    validation_rows=validation_rows,
                    test_rows=test_rows,
                    unsteered_test_composite=unsteered_test_composite,
                )
                final_summary_rows.append(
                    {
                        "seed": seed,
                        "concept_id": concept_id,
                        "concept": concept_name,
                        "scheme": scheme,
                        "selected_factor": summary.selected_factor,
                        "validation_composite": summary.validation_composite,
                        "test_composite": summary.test_composite,
                        "concept_relevance": summary.concept_relevance,
                        "instruction_relevance": summary.instruction_relevance,
                        "fluency": summary.fluency,
                        "perplexity": summary.perplexity,
                        "delta_over_unsteered": summary.delta_over_unsteered,
                        **summary.diagnostics,
                    }
                )
                if alpaca_prompts_df is not None:
                    alpaca_eval_df = build_prompt_eval_df(
                        prompts_df=alpaca_prompts_df,
                        concept_id=concept_id,
                        concept_name=concept_name,
                        factor=summary.selected_factor,
                    )
                    restore_aggregated_result(model, aggregation_result, manifest)
                    _, alpaca_outputs = evaluate_factor_sweep(
                        model,
                        alpaca_eval_df,
                        model_name=f"{scheme}_alpaca",
                        axbench=axbench,
                        args=args,
                        concept_id=concept_id,
                        lm_model=None,
                    )
                    concept_method_outputs[scheme].append(
                        alpaca_outputs.rename(
                            columns={f"{scheme}_alpaca_steered_generation": "output"}
                        )[["instruction", "output"]]
                    )

            pd.DataFrame(concept_checkpoint_rows).to_csv(concept_dir / "checkpoint_validation_metrics.csv", index=False)
            pd.DataFrame(concept_factor_sweep_rows).to_csv(concept_dir / "factor_sweeps.csv", index=False)

    checkpoint_df = pd.DataFrame(checkpoint_rows)
    factor_sweeps_df = pd.DataFrame(factor_sweep_rows)
    final_summary_df = pd.DataFrame(final_summary_rows)
    checkpoint_df.to_csv(output_dir / "checkpoint_validation_metrics.csv", index=False)
    factor_sweeps_df.to_csv(output_dir / "factor_sweeps.csv", index=False)
    final_summary_df.to_csv(output_dir / "final_summary.csv", index=False)

    grouped = (
        final_summary_df.groupby("scheme", as_index=False)[
            ["test_composite", "delta_over_unsteered", "instruction_relevance", "fluency"]
        ]
        .mean()
        .sort_values("test_composite", ascending=False)
    )
    grouped.to_csv(output_dir / "summary_by_scheme.csv", index=False)

    promoted_methods = grouped.head(2)["scheme"].tolist()
    if "map" not in promoted_methods:
        promoted_methods.append("map")
    promoted_methods = sorted(set(promoted_methods + ["unsteered"]))
    alpaca_inputs = {
        method: pd.concat(frames, ignore_index=True)
        for method, frames in concept_method_outputs.items()
        if frames
    }
    alpaca_summary = run_alpacaeval_transfer(
        output_dir=output_dir,
        promoted_methods=promoted_methods,
        concept_results=alpaca_inputs,
    )
    write_json(output_dir / "alpacaeval_summary.json", alpaca_summary)

    if language_model is not None:
        language_model.save_cache()
        if language_model_client is not None:  # pragma: no cover - network resource cleanup
            asyncio.run(language_model_client.close())


if __name__ == "__main__":
    main()
