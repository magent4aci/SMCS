# FinQA Dataset

FinQA is a financial domain numerical reasoning dataset for question-answering tasks based on financial statements.

## Data Acquisition (run in networked environment, then copy to server)

1. **HuggingFace** (recommended):
   ```bash
   cd benchmarks/FinQA/dataset
   pip install datasets
   python download_finqa.py
   ```

## Data Format

- `FinQA_test.json`: JSON array, each item contains:
  - `pre_text`: Text list before the table
  - `post_text`: Text list after the table
  - `table`: Table data (2D list)
  - `question`: Question text
  - `answer`: Standard answer (numeric or string)

## Data Source

- HuggingFace: https://huggingface.co/datasets/ibm-research/finqa
- Paper: FinQA: A Dataset of Numerical Reasoning over Financial Data
