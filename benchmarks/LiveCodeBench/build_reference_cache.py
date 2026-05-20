import argparse
import json
import os
import re
import sys
from pathlib import Path

from tqdm import tqdm

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "global_utils").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from build_single_question_bank import (
    build_prompt,
    build_prompt_nemotron_thinking,
    remove_think,
    transfer_format,
)
from global_utils.utils import generate_general
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.evaluation import extract_instance_results
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
from lcb_runner.lm_styles import LanguageModelStore
from lcb_runner.runner.scenario_router import combine_results, sort_and_extract_save_results
from lcb_runner.utils.scenarios import Scenario


REFERENCE_START_DATE = "2024-08-01"
REFERENCE_END_DATE = "2025-02-01"


def build_reference_cache(model, max_tokens=8192):
    exp_name = "single_agent_8k" if max_tokens == 8192 else "single_agent"
    model_name = model + "_thinking" if model == "Llama-3_3-Nemotron-Super-49B-v1" else model
    result_dir = os.path.join("result", exp_name, model_name)
    result_cache_file = os.path.join(result_dir, "result_cache.json")
    result_file = os.path.join(result_dir, "result.json")
    eval_all_file = os.path.join(result_dir, "result_eval_all.json")
    summary_file = os.path.join(result_dir, "summary.json")
    os.makedirs(result_dir, exist_ok=True)

    benchmark = load_code_generation_dataset(
        "release_v6",
        start_date=REFERENCE_START_DATE,
        end_date=REFERENCE_END_DATE,
    )
    benchmark = sorted(benchmark, key=lambda x: x.question_id)

    old_save_results = []
    if os.path.exists(result_cache_file):
        with open(result_cache_file, "r", encoding="utf-8") as f:
            old_save_results = json.load(f)
    seen_question_ids = set()
    old_save_results = [
        instance
        for instance in old_save_results
        if instance["output_list"]
        and instance["question_id"] not in seen_question_ids
        and not seen_question_ids.add(instance["question_id"])
    ]
    done_question_ids = {instance["question_id"] for instance in old_save_results}
    remaining_benchmark = [
        instance for instance in benchmark if instance.question_id not in done_question_ids
    ]

    results = []
    model_fake = LanguageModelStore["Qwen/QwQ-32B"]
    for example in tqdm(remaining_benchmark, total=len(remaining_benchmark)):
        if model == "Llama-3_3-Nemotron-Super-49B-v1":
            messages = build_prompt_nemotron_thinking(example)
        else:
            messages = build_prompt(example)
        response = generate_general(model, messages, max_tokens, 0.7)
        results.append([response])

        combined_results = combine_results(
            Scenario.codegeneration,
            results,
            model_fake,
            False,
        )
        save_results = [
            instance.insert_output(outputs_list, extracted_list)
            for instance, (outputs_list, extracted_list) in zip(
                remaining_benchmark,
                combined_results,
            )
        ]
        save_results += old_save_results
        save_results, _ = sort_and_extract_save_results(Scenario.codegeneration, save_results)
        with open(result_cache_file, "w", encoding="utf-8") as f:
            json.dump(save_results, f, indent=4)

    combined_results = combine_results(Scenario.codegeneration, results, model_fake, False)
    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(
            remaining_benchmark,
            combined_results,
        )
    ]
    save_results += old_save_results
    save_results, combined_results = sort_and_extract_save_results(
        Scenario.codegeneration,
        save_results,
    )
    with open(result_cache_file, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=4)

    eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
    generations = [[remove_think(response) for response in extracted] for _, extracted in combined_results]
    assert len(eval_samples) == len(generations)
    metrics = codegen_metrics(eval_samples, generations, k_list=[1])
    graded = extract_instance_results(metrics[1])
    metadatas = metrics[2] if metrics else [[] for _ in benchmark]
    save_eval_results = [
        instance.insert_output_evaluation(outputs_list, extracted_list, graded_list, metadata=meta)
        for instance, (outputs_list, extracted_list), graded_list, meta in zip(
            benchmark,
            combined_results,
            graded,
            metadatas,
        )
    ]

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    with open(eval_all_file, "w", encoding="utf-8") as f:
        json.dump(save_eval_results, f, indent=4)
    with open(result_file, "w", encoding="utf-8") as f:
        for record in transfer_format(save_eval_results):
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=8192)
    args = parser.parse_args()
    build_reference_cache(args.model, max_tokens=args.max_tokens)
