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

from build_single_question_bank import clean_answer
from harness import check_correctness
from run_api import DATASET_MAPPING, PROMPT_WRAPPER_API, make_request
from global_utils.utils import generate_general


def build_reference_cache(model, max_tokens=8192):
    exp_name = "single_agent_8k" if max_tokens == 8192 else "single_agent"
    result_dir = os.path.join("result", exp_name, model)
    result_file = os.path.join(result_dir, "result.json")
    response_cache_file = os.path.join(result_dir, "results.jsonl")
    correctness_file = os.path.join(result_dir, "eval_acc.jsonl")
    summary_file = os.path.join(result_dir, "summary.json")
    os.makedirs(result_dir, exist_ok=True)

    raw_problem_fn, map_problem_fn = DATASET_MAPPING["mbpp"]
    raw_problems = raw_problem_fn()
    problems = list(map(map_problem_fn, raw_problems))
    for problem in problems:
        problem["question_id"] = problem["id"]

    final_res = []
    done_question_ids = set()
    exec_result = []
    if os.path.exists(result_file):
        with jsonlines.Reader(open(result_file, "r", encoding="utf-8")) as reader:
            for previous in reader:
                done_question_ids.add(previous["question_id"])
                final_res.append(previous["is_correct"])
                exec_result.append({"task_id": previous["question_id"], "passed": previous["is_correct"]})

    llm_kwargs = {"temperature": 0.7, "max_tokens": max_tokens, "model": model}
    with jsonlines.Writer(open(result_file, "a", encoding="utf-8")) as result_writer, jsonlines.Writer(
        open(response_cache_file, "a", encoding="utf-8")
    ) as cache_writer:
        for example, raw_example in tqdm(zip(problems, raw_problems), total=len(problems)):
            question_id = example["question_id"]
            if question_id in done_question_ids:
                continue
            prompt = PROMPT_WRAPPER_API.format(
                instruction=example["instruction"],
                response=example["response_prefix"],
            )
            response = make_request(prompt, generate_general, llm_kwargs.copy())
            pred = clean_answer(response)
            check_result = check_correctness(
                problem=raw_example,
                completion=clean_answer(pred),
                timeout=500,
                completion_id=question_id,
            )
            is_correct = check_result["passed"]
            final_res.append(is_correct)
            exec_result.append({"task_id": question_id, "passed": is_correct})
            result_writer.write(
                {
                    "question_id": question_id,
                    "question": example["instruction"],
                    "pred": pred,
                    "response": response,
                    "model_response": response,
                    "is_correct": is_correct,
                }
            )
            cache_writer.write({"task_id": question_id, "response": response})
            with open(correctness_file, "w", encoding="utf-8") as f:
                json.dump({"exec_result": exec_result}, f)
            summary = {
                "corr": sum(final_res),
                "wrong": len(final_res) - sum(final_res),
                "acc": sum(final_res) / len(final_res),
                "schedule": f"{len(final_res)}/{len(problems)}",
            }
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=8192)
    args = parser.parse_args()
    build_reference_cache(args.model, max_tokens=args.max_tokens)
