import asyncio
import random
import re
import os
import time

import inspect
import openai
import numpy as np
import traceback
import argparse
import jsonlines
import datetime
from datasets import load_from_disk, load_dataset
import json
import multiprocessing
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from openai import OpenAI
import multiprocessing
import sys
import copy
import math
import re
import itertools
from modelscope import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import torch
from pathlib import Path
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "global_utils").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from global_utils.utils import (
    generate_openai,
    generate_with_references,
    DEBUG,
    generate_general,
)
from moa_api import raw_moa_api, async_raw_moa_api
from pathlib import Path
from instruction_following_eval.evaluation_main import evaluate
from global_utils.data_utils import load_jsonl_records

import nltk

# Set NLTK data path
NLTK_DATA_PATH = str(Path(__file__).resolve().parent / 'nltk_data')
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

# Download required NLTK data if not exists
if not os.path.exists(os.path.join(NLTK_DATA_PATH, 'tokenizers/punkt')):
    nltk.download('punkt', download_dir=NLTK_DATA_PATH)
if not os.path.exists(os.path.join(NLTK_DATA_PATH, 'tokenizers/punkt_tab')):
    nltk.download('punkt_tab', download_dir=NLTK_DATA_PATH)

MAX_PROCESSES = 8
DATA_PATH = './dataset/train.jsonl'
MODEL_LIST_DICT = {
    '6_small': ['Qwen2.5-7B-Instruct', 'glm-4-9b-chat', 'Qwen2-7B-Instruct', 'Meta-Llama-3.1-8B-Instruct',
                  'internlm2_5-7b-chat', 'gemma-2-9b-it'],
    '4_mid': ['Meta-Llama-3.3-70B-Instruct', 'Qwen2.5-32b-Instruct', 'gemma_3_27b_it', 'QwQ-32B'],
    '6_mid': ['Meta-Llama-3.3-70B-Instruct', 'Qwen2.5-32b-Instruct', 'gemma_3_27b_it', 'QwQ-32B', 'EXAONE-Deep-32B', ],
    '4_mid+2_small': ['Meta-Llama-3.3-70B-Instruct', 'Qwen2.5-32b-Instruct', 'gemma_3_27b_it', 'QwQ-32B',
                      'Qwen2.5-7B-Instruct', 'internlm2_5-7b-chat']
}

def build_prompt(question):
    messages = [{"role": "user",
                 "content": question}]
    return messages

def build_prompt_nemotron_thinking(question, thinking="on"):
    messages = [{"role": "system",
                 "content": f"detailed thinking {thinking}"}]
    messages.append({"role": "user", "content": question})
    return messages

def remove_think(response):
    if response is None:
        return ''
    if '</think>' in response:
        return response.split('</think>')[-1].strip()
    elif '</thought>' in response:
        return response.split('</thought>')[-1].strip()
    elif re.search(r'Reasoned for .* seconds', response):
        return response.split(re.search(r'Reasoned for .* seconds', response).group())[-1].strip()
    
    return response


def transfer_format(result):
    new_result = [
        {
            'question_id': i,
            'question': r['prompt'],
            'pred': remove_think(r['response']),
            'model_response': r['response'],
            'is_correct': r['follow_all_instructions'],
            'follow_instruction_list': r['follow_instruction_list'],
            'instruction_id_list': r['instruction_id_list']
        }
        for i, r in enumerate(result)
    ]
    return new_result


def single_agent_test(model, max_tokens=2048, force_8k=False):
    if force_8k:
        max_tokens = 8192
    dir_name = 'question_bank_8k' if max_tokens == 8192 else 'question_bank'
    if 'Llama-3_3-Nemotron-Super-49B-v1' == model:
        model_name = model + '_thinking'
    else:
        model_name = model
        
    examples = load_jsonl_records(DATA_PATH)

    result_dir = os.path.join('result', dir_name, model_name)
    result_file = os.path.join(result_dir, f'result_cache.json')
    eval_result_file = os.path.join(result_dir, f'eval_results_strict.jsonl')
    eval_result_summary_file = os.path.join(result_dir, f'eval_results_strict_summary.json')
    result_format_file = os.path.join(result_dir, f'result.json')
    summary_file = os.path.join(result_dir, f'summary.json')
    final_res = []
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    #f_result_handle = jsonlines.Writer(open(result_file, 'a', encoding='utf-8'))

    has_question_id_list = []
    if os.path.exists(result_file):
        with jsonlines.Reader(open(result_file, 'r', encoding='utf-8')) as f:
            for pre_r_dict in f:
                #if pre_r_dict['response'] is not None:
                has_question_id_list.append(pre_r_dict['question_id'])
                #results.append(pre_r_dict)
                #final_res.append(is_equiv(pre_r_dict['answer'], pre_r_dict['pred']))
    
    for question_id, example in tqdm(enumerate(examples), total=len(examples)):
        if question_id in has_question_id_list:
            continue
        question = example['prompt']
        print(f"Question: {question}")
        if 'Llama-3_3-Nemotron-Super-49B-v1' == model:
            messages = build_prompt_nemotron_thinking(question)
        else:
            messages = build_prompt(question)
        response = generate_general(model, messages, max_tokens, 0.7)

        #response = remove_think(response)

        res_dict = {'question_id': question_id, 'prompt': question, 'response': response}
        with open(result_file, 'a') as f:
            f.write(json.dumps(res_dict) + '\n')

    evaluate(DATA_PATH, result_file, result_dir)

    with open(eval_result_file, 'r') as f:
        result = [json.loads(line) for line in f]
    new_result = transfer_format(result)
    with open(result_format_file, 'w') as f:
        for r in new_result:
            f.write(json.dumps(r) + '\n')

    with open(eval_result_summary_file, 'r') as f:
        summary = json.load(f)
    with open(summary_file, 'w') as f:
        json.dump(summary, f)

    print(1)


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
