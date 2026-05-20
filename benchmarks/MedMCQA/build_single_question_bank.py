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
from medmcqa_utils import load_medmcqa_question_bank_split
from global_utils.utils import (
    generate_openai,
    generate_with_references,
    DEBUG,
    generate_general,
)
MAX_PROCESSES = 8
DATA_PATH = './dataset/MedMCQA_test.json'
LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
INDEX_TO_LETTER = {v: k for k, v in LETTER_TO_INDEX.items()}

def extract_answer(answer):
    patterns = [r'[Aa]nswer is \((.)\)', r'[Aa]nswer: \((.)\)', r'[Aa]nswer: \((.)\)', r'[Aa]nswer \((.)\)', r'[Aa][nN][sS][wW][eE][rR]\s+is.*?([a-zA-Z])', r'\((.)\)']
    for pattern in patterns:
        match = re.findall(pattern, answer)
        if match and match[-1] in LETTER_TO_INDEX:
            return match[-1]
    # if not match, select 'A'
    print(f'Not match, select A \n {answer}')
    return 'A'

def build_prompt(question):
    prompt = f"Provide your step-by-step reasoning first, and then print \"The answer is (X)\", where X is the answer choice (one capital letter), at the end of your response. \n The Question is: {question}\n"
    return prompt

def single_agent_test(model, max_tokens=2048):
    examples = load_medmcqa_question_bank_split(DATA_PATH)
    exp_name = 'question_bank_8k' if max_tokens == 8192 else 'question_bank'
    result_dir = os.path.join('result', exp_name, model)
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
    f_result_handle = jsonlines.Writer(open(result_file, 'a', encoding='utf-8'))

    for question_id, example in tqdm(enumerate(examples), total=len(examples)):
        if question_id in has_question_id_list:
            continue
        prompt = build_prompt(example['question'])
        messages = [
            {"role": "system", "content": "You are a very intelligent assistant, who is very professional in medical science and biology like an expert doctor."},
            {"role": "user", "content": prompt}
        ]
        response = generate_general(model, messages, max_tokens, 0.7)
        pred = extract_answer(response)
        gt = example['gold_answer']
        is_correct = gt == pred
        final_res.append(is_correct)
        cnt += 1
        res_dict = {'question_id': question_id, 'question': example['question'],
                    'answer': gt, 'pred': pred, 'is_correct': is_correct, 'model_response': response}
        sum_dict = {'corr': sum(final_res), 'wrong': len(final_res) - sum(final_res),
                    'acc': sum(final_res) / len(final_res), 'schedule': f'{cnt}/{len(examples)}'}
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
