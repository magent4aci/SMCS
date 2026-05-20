"""
HumanEval code execution and correctness verification.
Uses subprocess to avoid multiprocessing failures in nested spawn environments (rjob/Pool worker, etc.).
"""
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Dict


def check_correctness(
    task_id,
    completion_id: int,
    solution: str,
    time_out: float = 10.0,
) -> Dict:
    """
    Evaluate functional correctness of code.
    HumanEval format: solution should contain imports + completion + test + check(entry_point).
    Uses subprocess to avoid multiprocessing failures in rjob and similar environments.
    """
    result_str = "failed"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(solution)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            timeout=time_out,
            cwd=tempfile.gettempdir(),
        )
        if result.returncode == 0:
            result_str = "passed"
        else:
            err = (result.stderr or result.stdout or b"").decode("utf-8", errors="replace")
            result_str = f"failed: {err.strip() or 'non-zero exit'}"
    except subprocess.TimeoutExpired:
        result_str = "timed out"
    except Exception as e:
        result_str = f"failed: {e}"
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass

    return dict(
        task_id=task_id,
        completion_id=completion_id,
        passed=result_str == "passed",
        result=result_str,
        solution=solution,
    )
