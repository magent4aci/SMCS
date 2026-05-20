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
from math_equivalence import is_equiv
from global_utils.data_utils import load_jsonl_records
from global_utils.utils import (
    generate_openai,
    generate_with_references,
    DEBUG,
    generate_general,
)
MAX_PROCESSES = 8
DATA_PATH = './dataset/train.jsonl'

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

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


def build_prompt(question):
    messages = [{"role": "system",
                 "content": "You are a math problem solver. Please solve the following math problem. Be sure to explain your solution in detail. The numerical values in the answer should be surrounded by \\boxed{}. The final answer should start with 'The answer is' and give the conclusion directly. Do not add any extra content."}]
    messages.append({"role": "user", "content": f"question:{question}"})
    return messages

def extract_answer(answer):
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

def single_agent_test(model, max_tokens=2048):
    examples = load_jsonl_records(DATA_PATH)[:1000]
    exp_name = 'question_bank_8k' if max_tokens == 8192 else 'question_bank'
    result_dir = os.path.join('result', exp_name, model)
    result_file = os.path.join(result_dir, f'result.json')
    summary_file = os.path.join(result_dir, f'summary.json')
    final_res = []
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    f_result_handle = jsonlines.Writer(open(result_file, 'a', encoding='utf-8'))
    cnt = 0
    has_question_id_list = []
    if os.path.exists(result_file):
        with jsonlines.Reader(open(result_file, 'r', encoding='utf-8')) as f:
            for pre_r_dict in f:
                has_question_id_list.append(pre_r_dict['question_id'])
                final_res.append(is_equiv(pre_r_dict['answer'], pre_r_dict['pred']))
        cnt = len(has_question_id_list)
    for question_id, example in tqdm(enumerate(examples), total=len(examples)):
        if question_id in has_question_id_list:
            continue
        question, answer = example['problem'], example['answer']
        messages = build_prompt(question)
        response = generate_general(model, messages, max_tokens, 0.7)
        pred = extract_answer(response)
        cnt += 1
        is_correct = is_equiv(pred, answer)
        final_res.append(is_correct)
        res_dict = {'question_id': question_id, 'question': question,
                    'answer': answer, 'pred': pred, 'is_correct': is_correct, 'model_response': response,
                    'solution': example['solution'], 'level': example['level'],
                    'subject': example['subject']}
        sum_dict = {'corr': sum(final_res), 'wrong': len(final_res)-sum(final_res), 'acc':sum(final_res)/len(final_res)}
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
