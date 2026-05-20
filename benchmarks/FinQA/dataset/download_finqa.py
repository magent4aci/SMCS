#!/usr/bin/env python3
"""
Run this script in a networked environment to download and convert FinQA test set to moa format in current directory.
After completion, copy FinQA_test.json to an offline server for use.
"""
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "FinQA_test.json")


def download_from_huggingface():
    """Download from HuggingFace and convert format.

    HuggingFace ibm-research/finqa is flattened; each record has: id, pre_text, post_text, table, question, answer
    """
    from datasets import load_dataset

    print("Downloading ibm-research/finqa from HuggingFace ...")
    ds = load_dataset("ibm-research/finqa", split="test")

    examples = []
    for idx, record in enumerate(ds):
        record_dict = dict(record)
        pre_text = record_dict.get("pre_text", [])
        post_text = record_dict.get("post_text", [])
        table = record_dict.get("table", [])
        question = record_dict.get("question", "")
        answer = record_dict.get("answer", record_dict.get("final_result", ""))
        if isinstance(pre_text, str):
            pre_text = [pre_text]
        if isinstance(post_text, str):
            post_text = [post_text]
        item = {
            "question_id": idx,
            "pre_text": pre_text,
            "post_text": post_text,
            "table": table,
            "question": question,
            "answer": str(answer) if answer is not None else "",
        }
        examples.append(item)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    print(f"Saved to {OUTPUT_PATH}, {len(examples)} items")


if __name__ == "__main__":
    try:
        download_from_huggingface()
    except ImportError:
        print("Please install first: pip install datasets")
        raise
