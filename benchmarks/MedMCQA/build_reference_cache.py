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

from build_single_question_bank import build_prompt, extract_answer
from global_utils.utils import generate_general
from medmcqa_utils import load_medmcqa_test_split


DATA_PATH = "./dataset/MedMCQA_test.json"


def build_reference_cache(model, max_tokens=8192):
    examples = load_medmcqa_test_split(DATA_PATH)
    exp_name = "single_agent_8k" if max_tokens == 8192 else "single_agent"
    result_dir = os.path.join("result", exp_name, model)
    result_file = os.path.join(result_dir, "result.json")
    summary_file = os.path.join(result_dir, "summary.json")
    os.makedirs(result_dir, exist_ok=True)

    final_res = []
    done_question_ids = set()
    if os.path.exists(result_file):
        with jsonlines.Reader(open(result_file, "r", encoding="utf-8")) as reader:
            for previous in reader:
                done_question_ids.add(previous["question_id"])
                final_res.append(previous["answer"] == previous["pred"])

    with jsonlines.Writer(open(result_file, "a", encoding="utf-8")) as writer:
        for example in tqdm(examples, total=len(examples)):
            question_id = example["question_id"]
            if question_id in done_question_ids:
                continue
            messages = [
                {
                    "role": "system",
                    "content": "You are a very intelligent assistant, who is very professional in medical science and biology like an expert doctor.",
                },
                {"role": "user", "content": build_prompt(example["question"])},
            ]
            response = generate_general(model, messages, max_tokens, 0.7)
            pred = extract_answer(response)
            answer = example["gold_answer"]
            is_correct = answer == pred
            final_res.append(is_correct)
            writer.write(
                {
                    "question_id": question_id,
                    "question": example["question"],
                    "answer": answer,
                    "pred": pred,
                    "is_correct": is_correct,
                    "model_response": response,
                }
            )
            summary = {
                "corr": sum(final_res),
                "wrong": len(final_res) - sum(final_res),
                "acc": sum(final_res) / len(final_res),
                "schedule": f"{len(final_res)}/{len(examples)}",
            }
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=8192)
    args = parser.parse_args()
    build_reference_cache(args.model, max_tokens=args.max_tokens)
