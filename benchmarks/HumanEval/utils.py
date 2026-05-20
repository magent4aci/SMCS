"""
HumanEval dataset utility functions.
Prefer local data: dataset/HumanEval.jsonl.
Fallback: evalplus or HuggingFace (requires network).
"""
import gzip
import json
import os
from pathlib import Path
from typing import Sequence, Mapping
from typing import Any, Dict

# Local data paths (supports multiple lookup methods)
_HUMANEVAL_DIR = Path(__file__).resolve().parent
_LOCAL_DATA_PATHS = [
    _HUMANEVAL_DIR / "dataset" / "HumanEval.jsonl",
    _HUMANEVAL_DIR / "dataset" / "HumanEval.jsonl.gz",
    Path("dataset") / "HumanEval.jsonl",
    Path("HumanEval") / "dataset" / "HumanEval.jsonl",
]

try:
    from evalplus.data import get_human_eval_plus
    USE_EVALPLUS = True
except ImportError:
    USE_EVALPLUS = False

try:
    from datasets import load_dataset
    USE_DATASETS = True
except ImportError:
    USE_DATASETS = False


def _find_local_data_path() -> Path | None:
    """Find local HumanEval data file"""
    for p in _LOCAL_DATA_PATHS:
        path = Path(p)
        if path.is_absolute() and path.exists():
            return path
        # Relative path: first relative to cwd, then to HumanEval dir
        for base in [Path.cwd(), _HUMANEVAL_DIR]:
            full = (base / path).resolve()
            if full.exists():
                return full
    return None


def _load_local_jsonl(path: Path) -> list[dict]:
    """Load from local JSONL file (supports .jsonl and .jsonl.gz)"""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_humaneval_raw_problems() -> list[dict]:
    """Get HumanEval raw problem list; prefer local data"""
    local_path = _find_local_data_path()
    if local_path is not None:
        problems = _load_local_jsonl(local_path)
        if problems:
            return problems
    if USE_EVALPLUS:
        problems = get_human_eval_plus()
        return list(problems.values())
    if USE_DATASETS:
        ds = load_dataset("openai/openai_humaneval", split="test")
        return list(ds)
    raise FileNotFoundError(
        "HumanEval data not found. Place HumanEval.jsonl in benchmarks/HumanEval/dataset/.\n"
        "Data sources:\n"
        "  - HuggingFace: openai/openai_humaneval (requires network)\n"
        "  - GitHub: https://github.com/openai/human-eval (data directory)\n"
        "  - Or install evalplus: pip install evalplus"
    )


def map_humaneval_problem(p: dict) -> Dict[str, Any]:
    """Map HumanEval raw problem to moa format"""
    task_id = p.get("task_id", p.get("id", 0))
    if isinstance(task_id, str) and "/" in str(task_id):
        task_id = str(task_id).split("/")[-1]
    prompt = p.get("prompt", "").strip()
    instruction = f"""Write a solution to the following problem:
```python
{prompt}
```"""
    response_prefix = f"""```python
{prompt}"""
    entry_point = p.get("entry_point", "")
    test = p.get("test", "")
    return dict(
        id=task_id,
        instruction=instruction,
        response_prefix=response_prefix,
        prompt=prompt,
        entry_point=entry_point,
        test=test,
    )


def post_process_humaneval(output: str) -> str:
    """Post-process model-generated code"""
    if "if __name__" in output:
        output = output.split("if __name__", 1)[0]
    elif "# Test" in output:
        output = output.split("# Test", 1)[0]
    elif "# Example" in output:
        output = output.split("# Example ", 1)[0]
    return output.strip()


# HumanEval code extraction and sanitize
IMPORTS = [
    "import math", "import re", "import sys", "import copy", "import datetime",
    "import itertools", "import collections", "import heapq", "import functools",
    "import hashlib", "import numpy", "import numpy as np", "import string",
    "from typing import *", "from collections import *",
]


def build_humaneval_solution(
    completion: str, prompt: str, test: str, entry_point: str, sanitize_fn=None
) -> str:
    """
    Build complete executable HumanEval solution.
    completion: model-generated code (function body)
    prompt: original prompt (with function signature)
    test: test code
    entry_point: entry function name
    """
    if sanitize_fn:
        completion = sanitize_fn(completion, entry_point)
    else:
        completion = post_process_humaneval(completion)
    code = "\n".join(IMPORTS) + "\n\n" + completion + "\n\n" + test + "\n\n" + f"check({entry_point})"
    return code


def write_jsonl(path: str | Path, data: Sequence[Mapping]):
    """Write JSONL file"""
    with Path(path).open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
