"""
HumanEval dataset API - same structure as MBPP run_api.
"""
from prompt_template import PROMPT_WRAPPER_API
from utils import (
    get_humaneval_raw_problems,
    map_humaneval_problem,
    post_process_humaneval,
    write_jsonl,
)


DATASET_MAPPING = {
    "humaneval": (get_humaneval_raw_problems, map_humaneval_problem),
}


def build_message(prompt):
    return [
        {"role": "system", "content": "You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions."},
        {"role": "user", "content": prompt},
    ]
