#!/usr/bin/env python3
"""
Run this script in a networked environment to download HumanEval.jsonl to the current directory.
After completion, copy HumanEval.jsonl to an offline server for use.
"""
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "HumanEval.jsonl")


def download_from_huggingface():
    """Download from HuggingFace"""
    from datasets import load_dataset
    print("Downloading openai/openai_humaneval from HuggingFace ...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in ds:
            f.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
    print(f"Saved to {OUTPUT_PATH}, {len(ds)} items")


if __name__ == "__main__":
    try:
        download_from_huggingface()
    except ImportError:
        print("Please install first: pip install datasets")
        raise
