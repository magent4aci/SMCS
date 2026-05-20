"""
FinQA grading logic.
Uses local math-answer equivalence helpers; no external project dependency.
"""
import os
import re

_script_dir = os.path.dirname(os.path.abspath(__file__))

# Use local answer-equivalence helpers.
from math_grader_utils import grade_answer_mathd, grade_answer_sympy


def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx:right_brace_idx + 1] if right_brace_idx is not None else None


def _remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left and s[-1] == "}"
        return s[len(left):-1]
    except Exception:
        return None


def _extract_gt_from_boxed(text):
    """Extract ground_truth from \\boxed{}"""
    boxed = _last_boxed_only_string(text)
    return _remove_boxed(boxed) if boxed else None


def extract_number(text):
    """FinQA decimal tolerance: extract number and round to integer for comparison"""
    if text is None or text == "":
        return ""
    pattern = r'-?\d+(?:\.\d+)?'
    match = re.search(pattern, str(text))
    if match:
        try:
            number = float(match.group(0))
            number = round(number, 0)
            return str(int(number))
        except (ValueError, TypeError):
            return str(text).strip()
    return str(text).strip()


def grade_finqa_answer(prediction: str, ground_truth: str) -> bool:
    """
    Grading logic:

    1. If ground_truth contains \\boxed, extract its content first
    2. Use grade_answer_mathd or grade_answer_sympy for judgment
    3. If incorrect, use extract_number for FinQA decimal tolerance comparison
    """
    if ground_truth is None or prediction is None or prediction == "":
        return False

    pred = str(prediction).strip()
    gt = str(ground_truth).strip()

    if "\\boxed" in gt:
        extracted = _extract_gt_from_boxed(gt)
        gt = str(extracted).strip() if extracted else ""

    is_correct = grade_answer_mathd(pred, gt) or grade_answer_sympy(pred, gt)

    if not is_correct:
        formatted_pred = extract_number(pred)
        formatted_gt = extract_number(gt)
        is_correct = (formatted_pred == formatted_gt)

    return is_correct
