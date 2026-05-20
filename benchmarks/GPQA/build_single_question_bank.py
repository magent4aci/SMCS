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
from gpqa_utils import create_prompts, load_gpqa_question_bank_split
from global_utils.utils import generate_general
MAX_PROCESSES = 8
DATA_PATH = './dataset'
LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
INDEX_TO_LETTER = {v: k for k, v in LETTER_TO_INDEX.items()}


def extract_answer(answer):
    patterns = [r'answer is \((.)\)', r'Answer: \((.)\)', r'answer: \((.)\)', r'answer \((.)\)', r'[Aa][nN][sS][wW][eE][rR]\s+is.*?([a-zA-Z])', r'\((.)\)']
    for pattern in patterns:
        match = re.findall(pattern, answer)
        if match and match[-1] in LETTER_TO_INDEX:
            return match[-1]
    # if not match, select 'A'
    print(f'Not match, select A \n {answer}')
    return 'A'

def single_agent_test(model, max_tokens=2048):
    examples = load_gpqa_question_bank_split(DATA_PATH, seed=42)
    exp_name = 'question_bank_8k' if max_tokens == 8192 else 'question_bank'
    result_dir = os.path.join('result', exp_name, model)
    result_file = os.path.join(result_dir, f'result.json')
    summary_file = os.path.join(result_dir, f'summary.json')
    final_res = []
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    prompts, examples = create_prompts(examples, prompt_type='chain_of_thought', few_shot_n=None, model_name=model)
    f_result_handle = jsonlines.Writer(open(result_file, 'a', encoding='utf-8'))
    cnt = 0
    has_question_id_list = []
    if os.path.exists(result_file):
        with jsonlines.Reader(open(result_file, 'r', encoding='utf-8')) as f:
            for pre_r_dict in f:
                has_question_id_list.append(pre_r_dict['question_id'])
                final_res.append(pre_r_dict['answer'] == pre_r_dict['pred'])
        cnt = len(has_question_id_list)
    for question_id, (prompt, example) in tqdm(enumerate(zip(prompts, examples)), total=len(examples)):
        if question_id in has_question_id_list:
            continue
        messages = [
            {"role": "system", "content": "You are a very intelligent assistant, who follows instructions directly."},
            {"role": "user", "content": prompt}
        ]
        response = generate_general(model, messages, max_tokens, 0.7)
        pred = extract_answer(response)
        is_correct = pred == INDEX_TO_LETTER[example.correct_index]
        cnt += 1
        final_res.append(INDEX_TO_LETTER[example.correct_index] == pred)
        res_dict = {'question_id': question_id, 'question': example.question, 'option': example[1:5],
                    'answer': INDEX_TO_LETTER[example.correct_index], 'pred': pred, 'is_correct': is_correct,
                    'model_response': response}
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
