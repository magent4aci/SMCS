import os
import re
import jsonlines
import json
import argparse
from tqdm import tqdm
from harness import check_correctness
from datasets import load_dataset, Dataset
import sys
from pathlib import Path
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "global_utils").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from run_api import DATASET_MAPPING, PROMPT_WRAPPER_API, make_request, post_process_humaneval, write_jsonl,\
    build_message
from global_utils.utils import (
    generate_openai,
    generate_with_references,
    DEBUG,
    generate_general,
)
MAX_PROCESSES = 8

def clean_answer(ans):
    ans = ans.split('```python', 1)[-1]
    return post_process_humaneval(ans[: index if (index := ans.find("```")) != -1 else len(ans)])

def single_agent_test(model, max_tokens=2048):
    exp_name = 'question_bank_8k' if max_tokens == 8192 else 'question_bank'
    result_dir = os.path.join('result', exp_name, model)
    result_file = os.path.join(result_dir, f'result.json')
    summary_file = os.path.join(result_dir, f'summary.json')
    ds = load_dataset("google-research-datasets/mbpp")
    raw_problems = list(ds['train']) + list(ds['validation'])
    raw_problem_fn, map_problem_fn = DATASET_MAPPING['mbpp']
    problems = list(map(map_problem_fn, raw_problems))
    problems_new = []
    for p in problems:
        p['question_id'] = p['id']
        problems_new.append(p)
    problems = problems_new
    final_res = []
    has_question_id_list = []
    cnt = 0
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    if os.path.exists(result_file):
        with jsonlines.Reader(open(result_file, 'r', encoding='utf-8')) as f:
            for pre_r_dict in f:
                has_question_id_list.append(pre_r_dict['question_id'])
                final_res.append(pre_r_dict['is_correct'])
        cnt = len(has_question_id_list)
    f_result_handle = jsonlines.Writer(open(result_file, 'a', encoding='utf-8'))
    llm_fn = generate_general
    llm_kwargs = {'temperature': 0.7, 'max_tokens': max_tokens, 'model': model}
    for example, raw_example in tqdm(zip(problems, raw_problems), total=len(problems)):
        question_id = example['question_id']
        if question_id in has_question_id_list:
            continue
        prompt = PROMPT_WRAPPER_API.format(
            instruction=example["instruction"], response=example["response_prefix"]
        )
        response = make_request(prompt, llm_fn, llm_kwargs)
        pred = clean_answer(response)
        check_result = check_correctness(problem=raw_example, completion=clean_answer(pred),
                                         timeout=500,
                                         completion_id=question_id)
        is_correct = check_result['passed']
        final_res.append(is_correct)
        cnt += 1
        res_dict = {'question_id': question_id, 'question': example['instruction'],
                     'pred': pred, 'model_response': response, 'is_correct': is_correct}
        sum_dict = {'corr': sum(final_res), 'wrong': len(final_res) - sum(final_res),
                    'acc': sum(final_res) / len(final_res), 'schedule': f'{cnt}/{len(problems)}'}
        with open(summary_file, "w") as fo:
            fo.write(json.dumps(sum_dict))
        f_result_handle.write(res_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192
    )
    args = parser.parse_args()
    single_agent_test(args.model, max_tokens=args.max_tokens)
