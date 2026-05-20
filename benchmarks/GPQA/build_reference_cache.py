import argparse
import json
import os
import re
import sys
from pathlib import Path

import jsonlines
from tqdm import tqdm

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "global_utils").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from global_utils.utils import generate_general
from gpqa_utils import create_prompts, load_gpqa_test_split


DATA_PATH = "./dataset"
LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}
INDEX_TO_LETTER = {v: k for k, v in LETTER_TO_INDEX.items()}


def extract_answer(answer):
    patterns = [
        r"answer is \((.)\)",
        r"Answer: \((.)\)",
        r"answer: \((.)\)",
        r"answer \((.)\)",
        r"[Aa][nN][sS][wW][eE][rR]\s+is.*?([a-zA-Z])",
        r"\((.)\)",
    ]
    for pattern in patterns:
        match = re.findall(pattern, answer)
        if match and match[-1] in LETTER_TO_INDEX:
            return match[-1]
    print(f"Not match, select A \n {answer}")
    return "A"


def build_reference_cache(model, max_tokens=8192):
    examples = load_gpqa_test_split(DATA_PATH, seed=42)
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

    prompts, examples = create_prompts(
        examples,
        prompt_type="chain_of_thought",
        few_shot_n=None,
        model_name=model,
    )
    with jsonlines.Writer(open(result_file, "a", encoding="utf-8")) as writer:
        for prompt, example in tqdm(zip(prompts, examples), total=len(examples)):
            if example.question_id in done_question_ids:
                continue
            messages = [
                {
                    "role": "system",
                    "content": "You are a very intelligent assistant, who follows instructions directly.",
                },
                {"role": "user", "content": prompt},
            ]
            response = generate_general(model, messages, max_tokens, 0.7)
            pred = extract_answer(response)
            answer = INDEX_TO_LETTER[example.correct_index]
            is_correct = pred == answer
            final_res.append(is_correct)
            writer.write(
                {
                    "question_id": example.question_id,
                    "question": example.question,
                    "option": example[1:5],
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
