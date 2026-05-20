# LiveMathBench Dataset

LiveMathBench is a mathematical reasoning benchmark dataset, including problem-solving and fill-in-the-blank types.

## Data Acquisition

1. **Manual copy**:
   - Place `en_test.json` in the `benchmarks/LiveMathBench/dataset/` directory

## Data Format

- `en_test.json`: JSONL format, one JSON object per line:
  - `question`: Question text (LaTeX format)
  - `answer`: Standard answer (can be `$...$` or `\boxed{...}` format)
  - `question_type`: Question type (e.g., "problem-solving", "fill-in-the-blank")
  - `options`: Option list (can be empty)

## Evaluation Logic

- Uses `grade_answer_mathd` and `grade_answer_sympy` for answer equivalence checking
- Supports extracting answers from `\boxed{}` or "Answer:" pattern
- Dependencies: sympy, pylatexenc
