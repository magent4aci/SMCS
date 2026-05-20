# HumanEval Dataset

**Prefer local data**: Place `HumanEval.jsonl` in this directory; no network required.

## Data Acquisition (run in networked environment, then copy to server)

1. **HuggingFace** (recommended):
   ```bash
   pip install huggingface_hub
   python -c "
   from datasets import load_dataset
   ds = load_dataset('openai/openai_humaneval', split='test')
   with open('HumanEval.jsonl', 'w') as f:
       for item in ds:
           f.write(__import__('json').dumps(dict(item)) + '\n')
   "
   ```

2. **GitHub**:
   - Open https://github.com/openai/human-eval
   - Download `data/HumanEval.jsonl.gz`, decompress and rename to `HumanEval.jsonl`

3. **Direct download**:
   - Data files from https://huggingface.co/datasets/openai/openai_humaneval

## Data Format

One JSON object per line. Required fields: `task_id`, `prompt`, `entry_point`, `test`, `canonical_solution`

## Alternative (requires network)

If local data is unavailable, evalplus or HuggingFace will attempt automatic download.
