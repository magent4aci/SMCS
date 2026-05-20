import argparse
import json
import os
import sys
from pathlib import Path

import jsonlines
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
from global_utils.data_utils import load_jsonl_records
from global_utils.utils import generate_general
from instruction_following_eval.evaluation_main import evaluate


DATA_PATH = "./dataset/test.jsonl"


def build_reference_cache(model, max_tokens=8192):
    exp_name = "single_agent_8k" if max_tokens == 8192 else "single_agent"
    model_name = model + "_thinking" if model == "Llama-3_3-Nemotron-Super-49B-v1" else model
    result_dir = os.path.join("result", exp_name, model_name)
    result_cache_file = os.path.join(result_dir, "result_cache.json")
    result_file = os.path.join(result_dir, "result.json")
    eval_result_file = os.path.join(result_dir, "eval_results_strict.jsonl")
    eval_summary_file = os.path.join(result_dir, "eval_results_strict_summary.json")
    summary_file = os.path.join(result_dir, "summary.json")
    os.makedirs(result_dir, exist_ok=True)

    examples = load_jsonl_records(DATA_PATH)
    done_question_ids = set()
    if os.path.exists(result_cache_file):
        with jsonlines.Reader(open(result_cache_file, "r", encoding="utf-8")) as reader:
            for previous in reader:
                done_question_ids.add(previous["question_id"])

    for question_id, example in tqdm(enumerate(examples), total=len(examples)):
        if question_id in done_question_ids:
            continue
        question = example["prompt"]
        if model == "Llama-3_3-Nemotron-Super-49B-v1":
            messages = build_prompt_nemotron_thinking(question)
        else:
            messages = build_prompt(question)
        response = generate_general(model, messages, max_tokens, 0.7)
        with open(result_cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"question_id": question_id, "prompt": question, "response": response}) + "\n")

    evaluate(DATA_PATH, result_cache_file, result_dir)
    with open(eval_result_file, "r", encoding="utf-8") as f:
        result = [json.loads(line) for line in f]
    with open(result_file, "w", encoding="utf-8") as f:
        for record in transfer_format(result):
            f.write(json.dumps(record) + "\n")
    with open(eval_summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=8192)
    args = parser.parse_args()
    build_reference_cache(args.model, max_tokens=args.max_tokens)
