from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from meta_swag.adapter_state import restore_adapter_state, save_manifest
from meta_swag.axbench_meta_swag import aggregate_checkpoint_records, choose_factor_from_factor_sweep, split_validation_test
from meta_swag.axbench_runtime import describe_external_repo, ensure_single_process_distributed_compat, import_alpaca_eval, import_axbench
from meta_swag.hypersteer_posterior import (
    aggregate_sampled_factor_rows,
    attach_multi_concept_validation_metrics,
    choose_risk_sensitive_factor,
    summarize_checkpoint_concept_rows,
    train_hypersteer_with_retention,
)

from run_axbench_meta_swag import (
    DEFAULT_FACTORS,
    build_language_model,
    build_prompt_eval_df,
    evaluate_factor_sweep,
    evaluate_mock_factor_sweep,
    evaluate_real_factor_sweep,
    generate_unsteered_outputs,
    load_concept_steering_df,
    load_dataframe,
    load_metadata,
    select_concept_ids,
    set_global_seed,
    write_json,
)


DEFAULT_METHODS = ["map", "uniform", "softmax", "ess", "threshold"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Posterior HyperSteer benchmark runner for AxBench and AlpacaEval.")
    parser.add_argument("--output-dir", required=True, help="Directory for benchmark artifacts.")
    parser.add_argument("--model-name", default="google/gemma-2-2b-it")
    parser.add_argument("--hypernet-name-or-path", default="google/gemma-2-2b")
    parser.add_argument("--metadata-path", required=True, help="AxBench metadata.jsonl.")
    parser.add_argument("--train-data-path", required=True, help="AxBench train_data.parquet.")
    parser.add_argument("--steering-data-path", help="Optional steering_data.parquet.")
    parser.add_argument("--alpacaeval-inputs-path", help="Optional CSV/JSONL with instruction prompts for transfer.")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--component", default="res")
    parser.add_argument("--max-concepts", type=int, default=None)
    parser.add_argument("--concept-ids", nargs="+", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--low-rank-dimension", type=int, default=1)
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--intervention-positions", default="all")
    parser.add_argument("--exclude-bos", action="store_true")
    parser.add_argument("--train-on-negative", action="store_true")
    parser.add_argument("--negative-example-ratio", type=int, default=None)
    parser.add_argument("--keep-last", type=int, default=20)
    parser.add_argument("--tail-fraction", type=float, default=0.4)
    parser.add_argument("--threshold-quantile", type=float, default=0.75)
    parser.add_argument("--validation-ratio", type=float, default=0.5)
    parser.add_argument("--steering-factors", nargs="+", type=float, default=DEFAULT_FACTORS)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-output-length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-validation-examples", type=int, default=32)
    parser.add_argument("--max-test-examples", type=int, default=32)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--posterior-factor-samples", type=int, default=0)
    parser.add_argument("--risk-aversion", type=float, default=0.0)
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--mock-judge", action="store_true")
    parser.add_argument("--skip-alpacaeval", action="store_true")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hypernet-initialize-from-pretrained", action="store_true")
    parser.set_defaults(hypernet_initialize_from_pretrained=True)
    return parser.parse_args()


def build_hypersteer_model_params(axbench, args: argparse.Namespace):
    params = axbench.ModelParams()
    params.batch_size = args.batch_size
    params.n_epochs = args.n_epochs
    params.topk = args.topk
    params.lr = args.lr
    params.low_rank_dimension = args.low_rank_dimension
    params.intervention_positions = args.intervention_positions
    params.exclude_bos = bool(args.exclude_bos)
    params.binarize_dataset = False
    params.train_on_negative = bool(args.train_on_negative)
    params.intervention_type = "addition"
    params.gradient_accumulation_steps = args.gradient_accumulation_steps
    params.weight_decay = args.weight_decay
    params.output_length = args.eval_output_length
    params.hypernet_name_or_path = args.hypernet_name_or_path
    params.hypernet_initialize_from_pretrained = bool(args.hypernet_initialize_from_pretrained)
    params.num_hidden_layers = args.num_hidden_layers
    params.steering_factors = list(args.steering_factors)
    return params


def build_hypersteer_training_dataframe(
    train_df: pd.DataFrame,
    selected_concept_ids: list[int],
    tokenizer,
    model_name: str,
    is_chat_model: bool,
    output_length: int,
    train_on_negative: bool,
    negative_example_ratio: int | None,
) -> pd.DataFrame:
    from axbench.models.hypernet.utils import prepare_df_combined  # type: ignore

    positive_df = train_df[train_df["concept_id"].isin(selected_concept_ids)].copy()
    negative_df = train_df[train_df["category"] == "negative"].copy()
    return prepare_df_combined(
        original_df=positive_df,
        negative_df=negative_df,
        tokenizer=tokenizer,
        binarize=False,
        train_on_negative=train_on_negative,
        is_chat_model=is_chat_model,
        output_length=output_length,
        model_name=model_name,
        max_num_of_examples=None,
        negative_example_ratio=negative_example_ratio,
    )


def restore_hypersteer_vector(model, flat_vector: np.ndarray, manifest) -> None:
    restore_adapter_state(model.concept_embedding, flat_vector, manifest)


def evaluate_posterior_factor_statistics(
    model,
    aggregation_result,
    manifest,
    evaluation_df: pd.DataFrame,
    axbench,
    args: argparse.Namespace,
    concept_id: int,
    lm_model,
    sample_prefix: str,
    seed: int,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    sampled_rows: list[list[dict[str, float]]] = []
    sampled_vectors = aggregation_result.sample(args.posterior_factor_samples, rng)
    for sample_index, flat_vector in enumerate(sampled_vectors):
        restore_hypersteer_vector(model, flat_vector, manifest)
        factor_rows, _ = evaluate_factor_sweep(
            model,
            evaluation_df,
            model_name=f"{sample_prefix}_sample_{sample_index}",
            axbench=axbench,
            args=args,
            concept_id=concept_id,
            lm_model=lm_model,
        )
        sampled_rows.append(factor_rows)
    restore_hypersteer_vector(model, aggregation_result.mean_vector, manifest)
    return aggregate_sampled_factor_rows(sampled_rows)


def run_alpacaeval_transfer(
    output_dir: Path,
    promoted_methods: list[str],
    method_outputs: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    if not method_outputs:
        return {"status": "skipped", "reason": "alpacaeval_inputs_path_not_provided"}

    alpaca_eval = import_alpaca_eval()
    summary: dict[str, Any] = {"status": "completed", "leaderboards": {}}
    for method_name in promoted_methods:
        if method_name not in method_outputs:
            continue
        leaderboard, _ = alpaca_eval.evaluate(
            model_outputs=method_outputs[method_name][["instruction", "output"]],
            name=method_name,
            output_path=str(output_dir / "alpacaeval" / method_name),
            is_return_instead_of_print=True,
            max_instances=len(method_outputs[method_name]),
        )
        summary["leaderboards"][method_name] = leaderboard.reset_index().to_dict("records")
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_single_process_distributed_compat()
    set_global_seed(args.seed)

    axbench = import_axbench()
    import axbench.scripts.train as axbench_train_module  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer

    metadata = load_metadata(args.metadata_path)
    metadata_by_concept_id = {int(entry["concept_id"]): entry for entry in metadata}
    train_df = load_dataframe(args.train_data_path)
    selected_concept_ids = [
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

    is_chat_model = True if args.model_name in axbench_train_module.CHAT_MODELS else False
    prefix_length = axbench_train_module.get_prefix_length(tokenizer) if is_chat_model else 1

    language_model_client = None
    language_model = None
    if not args.mock_judge:
        try:
            language_model_client, language_model = build_language_model(axbench, args.judge_model, output_dir)
        except Exception as exc:  # pragma: no cover - environment specific
            print(f"Falling back to mock judge because real judge setup failed: {exc}")
            args.mock_judge = True
            language_model_client, language_model = None, None

    model_params = build_hypersteer_model_params(axbench, args)
    training_df = build_hypersteer_training_dataframe(
        train_df=train_df,
        selected_concept_ids=selected_concept_ids,
        tokenizer=tokenizer,
        model_name=args.model_name,
        is_chat_model=is_chat_model,
        output_length=args.eval_output_length,
        train_on_negative=bool(args.train_on_negative),
        negative_example_ratio=args.negative_example_ratio,
    )

    model = axbench.HyperSteer(
        base_model,
        tokenizer,
        layer=args.layer,
        training_args=model_params,
        lm_model_name=args.model_name,
        device=device,
        seed=args.seed,
    )
    model.make_model(
        mode="train",
        embed_dim=base_model.config.hidden_size,
        low_rank_dimension=args.low_rank_dimension,
        num_hidden_layers=args.num_hidden_layers,
        dtype=torch.bfloat16 if use_bf16 else None,
        intervention_type="addition",
        metadata_path=args.metadata_path,
        dump_dir=str(output_dir),
        model_params=model_params,
        hypernet_name_or_path=args.hypernet_name_or_path,
        hypernet_initialize_from_pretrained=args.hypernet_initialize_from_pretrained,
    )

    retained_records, manifest = train_hypersteer_with_retention(
        model,
        training_df,
        keep_last=args.keep_last,
        tail_fraction=args.tail_fraction,
        checkpoint_id_prefix="hypersteer",
        prefix_length=prefix_length,
        positions=model_params.intervention_positions,
        exclude_bos=model_params.exclude_bos,
        metadata_path=args.metadata_path,
        world_size=1,
    )
    save_manifest(manifest, output_dir / "hypersteer_manifest.json")

    validation_by_concept: dict[int, pd.DataFrame] = {}
    test_by_concept: dict[int, pd.DataFrame] = {}
    concept_method_outputs: dict[str, list[pd.DataFrame]] = defaultdict(list)
    alpaca_prompts_df = None if args.skip_alpacaeval or not args.alpacaeval_inputs_path else load_dataframe(args.alpacaeval_inputs_path)

    for concept_id in selected_concept_ids:
        metadata_entry = metadata_by_concept_id[concept_id]
        concept_train_df = train_df[train_df["concept_id"] == concept_id].copy()
        steering_df = load_concept_steering_df(concept_train_df, metadata_entry, concept_id, args)
        validation_df, test_df = split_validation_test(steering_df, validation_ratio=args.validation_ratio)
        validation_by_concept[concept_id] = validation_df.head(args.max_validation_examples * max(1, len(args.steering_factors))).copy()
        test_by_concept[concept_id] = test_df.head(args.max_test_examples * max(1, len(args.steering_factors))).copy()

    checkpoint_rows: list[dict[str, Any]] = []
    factor_sweep_rows: list[dict[str, Any]] = []
    final_summary_rows: list[dict[str, Any]] = []
    posterior_factor_rows: list[dict[str, Any]] = []
    unsteered_test_scores: dict[int, float] = {}

    for concept_id in selected_concept_ids:
        metadata_entry = metadata_by_concept_id[concept_id]
        unsteered_test_df = test_by_concept[concept_id].copy()
        unsteered_test_df["factor"] = 0.0
        unsteered_test_outputs = generate_unsteered_outputs(
            base_model=base_model,
            tokenizer=tokenizer,
            evaluation_df=unsteered_test_df,
            batch_size=args.eval_batch_size,
            output_length=args.eval_output_length,
            temperature=args.temperature,
            device=device,
        )
        for key, values in unsteered_test_outputs.items():
            unsteered_test_df[f"unsteered_{key}"] = values
        unsteered_test_rows = (
            evaluate_mock_factor_sweep(unsteered_test_df, "unsteered")
            if args.mock_judge or language_model is None
            else evaluate_real_factor_sweep(axbench, unsteered_test_df, "unsteered", concept_id, language_model)
        )
        _, unsteered_test_scores[concept_id] = choose_factor_from_factor_sweep(unsteered_test_rows)

    for record in retained_records:
        restore_hypersteer_vector(model, record.adapter_vector, manifest)
        concept_summaries = []
        for concept_id in selected_concept_ids:
            concept_name = metadata_by_concept_id[concept_id]["concept"]
            validation_rows, _ = evaluate_factor_sweep(
                model,
                validation_by_concept[concept_id],
                model_name="checkpoint",
                axbench=axbench,
                args=args,
                concept_id=concept_id,
                lm_model=language_model,
            )
            concept_summary = summarize_checkpoint_concept_rows(
                concept_id=concept_id,
                concept_name=concept_name,
                factor_rows=validation_rows,
            )
            concept_summaries.append(concept_summary)
            for row in validation_rows:
                factor_sweep_rows.append(
                    {
                        "checkpoint_id": record.checkpoint_id,
                        "partition": "validation_checkpoint",
                        "concept_id": concept_id,
                        "concept": concept_name,
                        **row,
                    }
                )

        attach_multi_concept_validation_metrics(record, concept_summaries)
        checkpoint_rows.append(
            {
                "checkpoint_id": record.checkpoint_id,
                "step": record.step,
                "epoch": record.epoch,
                "train_loss": record.train_loss,
                "weighting_metric": record.weighting_metric,
                "selected_factor_mean": record.selected_factor,
                "concept_summaries": json.dumps(concept_summaries),
                "vector_dimension": record.adapter_dimension,
            }
        )

    for scheme in args.methods:
        aggregation_result = aggregate_checkpoint_records(
            retained_records,
            scheme=scheme,
            beta=1.0,
            threshold_quantile=args.threshold_quantile,
            low_rank_rank=min(args.keep_last, 20),
        )
        restore_hypersteer_vector(model, aggregation_result.mean_vector, manifest)

        for concept_id in selected_concept_ids:
            concept_name = metadata_by_concept_id[concept_id]["concept"]
            validation_rows, _ = evaluate_factor_sweep(
                model,
                validation_by_concept[concept_id],
                model_name=scheme,
                axbench=axbench,
                args=args,
                concept_id=concept_id,
                lm_model=language_model,
            )
            test_rows, _ = evaluate_factor_sweep(
                model,
                test_by_concept[concept_id],
                model_name=scheme,
                axbench=axbench,
                args=args,
                concept_id=concept_id,
                lm_model=language_model,
            )

            factor_statistics = None
            if args.posterior_factor_samples > 0:
                factor_statistics = evaluate_posterior_factor_statistics(
                    model=model,
                    aggregation_result=aggregation_result,
                    manifest=manifest,
                    evaluation_df=validation_by_concept[concept_id],
                    axbench=axbench,
                    args=args,
                    concept_id=concept_id,
                    lm_model=language_model,
                    sample_prefix=f"{scheme}_posterior",
                    seed=args.seed + concept_id,
                )
                for row in factor_statistics:
                    posterior_factor_rows.append(
                        {
                            "scheme": scheme,
                            "concept_id": concept_id,
                            "concept": concept_name,
                            **row,
                        }
                    )
                selected_factor, risk_adjusted_score = choose_risk_sensitive_factor(
                    factor_statistics,
                    risk_aversion=args.risk_aversion,
                )
                validation_stats_row = next(row for row in factor_statistics if float(row["factor"]) == float(selected_factor))
                validation_composite = float(validation_stats_row["composite_mean"])
            else:
                selected_factor, validation_composite = choose_factor_from_factor_sweep(validation_rows)
                risk_adjusted_score = float(validation_composite)

            for row in validation_rows:
                factor_sweep_rows.append(
                    {
                        "scheme": scheme,
                        "partition": "validation_method",
                        "concept_id": concept_id,
                        "concept": concept_name,
                        **row,
                    }
                )
            for row in test_rows:
                factor_sweep_rows.append(
                    {
                        "scheme": scheme,
                        "partition": "test_method",
                        "concept_id": concept_id,
                        "concept": concept_name,
                        **row,
                    }
                )

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

            final_summary_rows.append(
                {
                    "scheme": scheme,
                    "concept_id": concept_id,
                    "concept": concept_name,
                    "selected_factor": float(selected_factor),
                    "validation_composite": float(validation_composite),
                    "validation_risk_adjusted": float(risk_adjusted_score),
                    "test_composite": float(chosen_test_row["composite"]),
                    "concept_relevance": float(chosen_test_row["concept_relevance"]),
                    "instruction_relevance": float(chosen_test_row["instruction_relevance"]),
                    "fluency": float(chosen_test_row["fluency"]),
                    "perplexity": float(chosen_test_row["perplexity"]) if not pd.isna(chosen_test_row["perplexity"]) else np.nan,
                    "delta_over_unsteered": float(chosen_test_row["composite"] - unsteered_test_scores[concept_id]),
                    **diagnostics,
                }
            )

            if alpaca_prompts_df is not None:
                alpaca_df = build_prompt_eval_df(
                    prompts_df=alpaca_prompts_df,
                    concept_id=concept_id,
                    concept_name=concept_name,
                    factor=float(selected_factor),
                )
                restore_hypersteer_vector(model, aggregation_result.mean_vector, manifest)
                _, alpaca_outputs = evaluate_factor_sweep(
                    model,
                    alpaca_df,
                    model_name=f"{scheme}_alpaca",
                    axbench=axbench,
                    args=args,
                    concept_id=concept_id,
                    lm_model=None,
                )
                concept_method_outputs[scheme].append(
                    alpaca_outputs.rename(columns={f"{scheme}_alpaca_steered_generation": "output"})[["instruction", "output"]]
                )

    checkpoint_df = pd.DataFrame(checkpoint_rows)
    factor_sweep_df = pd.DataFrame(factor_sweep_rows)
    final_summary_df = pd.DataFrame(final_summary_rows)
    checkpoint_df.to_csv(output_dir / "hypersteer_checkpoint_validation_metrics.csv", index=False)
    factor_sweep_df.to_csv(output_dir / "hypersteer_factor_sweeps.csv", index=False)
    final_summary_df.to_csv(output_dir / "hypersteer_final_summary.csv", index=False)
    if posterior_factor_rows:
        pd.DataFrame(posterior_factor_rows).to_csv(output_dir / "hypersteer_posterior_factor_stats.csv", index=False)

    summary_by_scheme = (
        final_summary_df.groupby("scheme", as_index=False)[["test_composite", "delta_over_unsteered", "instruction_relevance", "fluency"]]
        .mean()
        .sort_values("test_composite", ascending=False)
    )
    summary_by_scheme.to_csv(output_dir / "hypersteer_summary_by_scheme.csv", index=False)

    promoted_methods = summary_by_scheme.head(2)["scheme"].tolist()
    alpaca_inputs = {
        method: pd.concat(frames, ignore_index=True)
        for method, frames in concept_method_outputs.items()
        if frames
    }
    write_json(
        output_dir / "hypersteer_alpacaeval_summary.json",
        run_alpacaeval_transfer(output_dir, promoted_methods, alpaca_inputs),
    )

    if language_model is not None:
        language_model.save_cache()
        if language_model_client is not None:  # pragma: no cover
            asyncio.run(language_model_client.close())


if __name__ == "__main__":
    main()
