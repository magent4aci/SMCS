"""
HumanEval evaluation harness - batch evaluation of generated results.
"""
import json
import os
from pathlib import Path
from tqdm import tqdm

from run_api import DATASET_MAPPING
from utils import post_process_humaneval, build_humaneval_solution
from execution import check_correctness


def read_data(path):
    with open(path, "r", encoding="utf-8") as f:
        if str(path).endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def harness_humaneval(source_path, save_path):
    """Evaluate HumanEval generated results"""
    raw_problem_fn, map_problem_fn = DATASET_MAPPING["humaneval"]
    raw_problems = raw_problem_fn()
    problems = list(map(map_problem_fn, raw_problems))
    problems_dict = {str(p["id"]): p for p in problems}

    gen_code_file = read_data(Path(source_path))
    if isinstance(gen_code_file, list) and gen_code_file and "task_id" not in gen_code_file[0]:
        gen_code_file = [{"task_id": d.get("question_id", i), "solution": d.get("response", d.get("solution", ""))} for i, d in enumerate(gen_code_file)]

    exec_result = []
    is_passed = []
    for gen_code in tqdm(gen_code_file):
        completion = gen_code.get("solution", gen_code.get("response", ""))
        task_id = gen_code.get("task_id", gen_code.get("question_id"))
        problem = problems_dict.get(str(task_id))
        if problem is None:
            for p in problems:
                if str(p["id"]) == str(task_id):
                    problem = p
                    break
        if problem is None:
            continue
        if "```python" in completion:
            part = completion.split("```python", 1)[-1]
            completion_clean = post_process_humaneval(part[: index if (index := part.find("```")) != -1 else len(part)])
        else:
            completion_clean = post_process_humaneval(completion)
        solution = build_humaneval_solution(
            completion_clean,
            problem["prompt"],
            problem["test"],
            problem["entry_point"],
        )
        check_result = check_correctness(
            task_id=task_id,
            completion_id=task_id,
            solution=solution,
            time_out=10,
        )
        exec_result.append(check_result)
        is_passed.append(check_result["passed"])

    result = {"pass_rate": sum(is_passed) / len(is_passed) if is_passed else 0, "exec_result": exec_result}
    with open(Path(save_path), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    return result.get("pass_rate", 0)
