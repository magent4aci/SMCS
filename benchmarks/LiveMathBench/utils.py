"""
LiveMathBench utility functions.
Answer extraction, data loading, etc.
"""
import json
import re
from grade_answer import extract_answer

ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)"


def extract_raw_answer(raw_data: str) -> str:
    """
    Extract answer from model raw output.
    1. If contains "Final Answer" and no \\boxed, use Answer pattern
    2. Otherwise extract from \\boxed{}
    3. Fallback: take last integer
    """
    if "Final Answer" in raw_data and "\\boxed" not in raw_data:
        matches = re.findall(ANSWER_PATTERN, raw_data)
        if matches:
            return matches[-1].strip()
    answer = extract_answer(raw_data)
    if answer is not None:
        return answer
    integers = re.findall(r'\b\d+\b', raw_data)
    if integers:
        return integers[-1]
    return ""


def load_livemathbench_data(path: str) -> list:
    """Load LiveMathBench data in JSONL format; auto-add question_id"""
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                ex = json.loads(line)
                if "question_id" not in ex:
                    ex["question_id"] = i
                examples.append(ex)
    return examples
