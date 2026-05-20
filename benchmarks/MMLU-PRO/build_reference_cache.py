import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "global_utils").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluate_from_api import (
    get_the_subset,
    load_mmlu_pro,
    merge_result,
    save_res,
    save_summary,
    single_request,
    update_result,
)


DATA_PATH = "./dataset/MMLU-Pro"


def build_reference_cache(model, max_tokens=8192, data_path=DATA_PATH, output_dir=None):
    exp_name = "single_agent_8k" if max_tokens == 8192 else "single_agent"
    output_dir = output_dir or os.path.join("result", exp_name, model)
    test_df, dev_df = load_mmlu_pro(data_path)
    test_df = get_the_subset(test_df, parti_ratio=0.1)
    os.makedirs(output_dir, exist_ok=True)

    for subject, examples in test_df.items():
        output_res_path = os.path.join(output_dir, f"{subject}_result.json")
        output_summary_path = os.path.join(output_dir, f"{subject}_summary.json")
        res, category_record = update_result(output_res_path)
        done_question_ids = {item["question_id"] for item in res}

        for each in tqdm(examples, desc=subject):
            if each["question_id"] in done_question_ids:
                continue
            label = each["answer"]
            pred, response, _, mean_logprob = single_request(
                model,
                each,
                dev_df,
                res,
                max_tokens=max_tokens,
            )
            if response is None:
                continue
            if subject not in category_record:
                category_record[subject] = {"corr": 0.0, "wrong": 0.0}
            each["pred"] = pred
            each["model_outputs"] = response
            each["is_correct"] = pred == label
            if mean_logprob is not None:
                each["mean_logprob"] = mean_logprob
            merge_result(res, each)
            if pred == label:
                category_record[subject]["corr"] += 1
            else:
                category_record[subject]["wrong"] += 1
            save_res(res, output_res_path)
            save_summary(category_record, output_summary_path)
        save_res(res, output_res_path)
        save_summary(category_record, output_summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "--model_name", dest="model", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    build_reference_cache(
        args.model,
        max_tokens=args.max_tokens,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )
