import os
import re
import jsonlines
import json
import argparse
from tqdm import tqdm
import sys
from pathlib import Path
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "global_utils").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from global_utils.data_utils import load_jsonl_records
from global_utils.utils import (
    generate_openai,
    generate_with_references,
    DEBUG,
    generate_general,
)
MAX_PROCESSES = 8
DATA_PATH = './dataset/AIME_1983_2023.jsonl'
LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
INDEX_TO_LETTER = {v: k for k, v in LETTER_TO_INDEX.items()}

def last_boxed_only_string(string):
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

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def build_prompt(question):
    messages = [{"role": "system",
                 "content": "Please reason step by step, and put your final answer within \\boxed{{}}."}]
    messages.append({"role": "user", "content": f"{question}"})
    return messages


def build_prompt_nemotron_thinking(question, thinking="on"):
    messages = [{"role": "system",
                 "content": f"detailed thinking {thinking}"}]
    messages.append({"role": "user", "content": f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."})
    return messages



def extract_answer(answer):
    if answer is None:
        return str(-114514)
    ext_ans = remove_boxed(last_boxed_only_string(answer))
    if ext_ans is None:
        # if not match, select 'A'
        print(f'Not match, use final match')
        patterns = [r'[Aa]nswer is ([+-]?\d*\.?\d+)', r'[Aa]nswer is (.*)', r'is ([+-]?\d*\.?\d+)', r'([+-]?\d*\.?\d+)']
        for pattern in patterns:
            match = re.findall(pattern, answer)
            if match:
                result = match[-1]
                if pattern == '[Aa]nswer is (.*)' and result[-1] == '.':
                    return result[:-1]
                return result
        return str(-114514)
    return ext_ans

def single_agent_test(model, max_tokens=2048, force_8k=False):
    examples = load_jsonl_records(DATA_PATH)
    if force_8k:
        max_tokens = 8192
    dir_name = 'question_bank_8k' if max_tokens == 8192 else 'question_bank'
    if 'Llama-3_3-Nemotron-Super-49B-v1' == model:
        model_name = model + '_thinking'
    else:
        model_name = model
    result_dir = os.path.join('result', dir_name, model_name)
    result_file = os.path.join(result_dir, f'result.json')
    summary_file = os.path.join(result_dir, f'summary.json')
    final_res = []
    has_question_id_list = []
    cnt = 0
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    if os.path.exists(result_file):
        with jsonlines.Reader(open(result_file, 'r', encoding='utf-8')) as f:
            for pre_r_dict in f:
                has_question_id_list.append(pre_r_dict['question_id'])
                final_res.append(pre_r_dict['answer'] == pre_r_dict['pred'])
        cnt = len(has_question_id_list)

    for question_id, example in tqdm(enumerate(examples), total=len(examples)):
        if question_id in has_question_id_list:
            continue
        if 'Llama-3_3-Nemotron-Super-49B-v1' == model:
            messages = build_prompt_nemotron_thinking(example['problem'])
        else:
            messages = build_prompt(example['problem'])
        response = generate_general(model, messages, max_tokens, 0.7)
        pred = extract_answer(response)
        gt = example['answer']
        is_correct = gt == pred
        final_res.append(is_correct)
        cnt += 1
        res_dict = {'question_id': question_id, 'question': example['problem'],
                    'answer': gt, 'pred': pred, 'is_correct': is_correct, 'model_response': response}
        sum_dict = {'corr': sum(final_res), 'wrong': len(final_res) - sum(final_res),
                    'acc': sum(final_res) / len(final_res), 'schedule': f'{cnt}/{len(examples)}'}
        with open(summary_file, "w") as fo:
            fo.write(json.dumps(sum_dict))
        with open(result_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(res_dict) + '\n')
        #f_result_handle.write(res_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force_8k",
        action='store_true',
        help="Deprecated alias for --max_tokens 8192.",
    )
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
    single_agent_test(args.model, max_tokens=args.max_tokens, force_8k=args.force_8k)
