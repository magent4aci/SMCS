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
from global_utils.utils import (
    generate_openai,
    generate_with_references,
    DEBUG,
    generate_general,
)

MAX_PROCESSES = 8
QUESTION_BANK_START_DATE = "2023-05-01"
QUESTION_BANK_END_DATE = "2024-07-31"
LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
INDEX_TO_LETTER = {v: k for k, v in LETTER_TO_INDEX.items()}

from lcb_runner.utils.scenarios import Scenario
from lcb_runner.lm_styles import LanguageModelStore
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset, load_code_generation_dataset_not_fast
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
from lcb_runner.prompts.code_generation import get_generic_question_template_answer
from lcb_runner.evaluation import extract_instance_results
from lcb_runner.runner.scenario_router import (
    combine_results,
    sort_and_extract_save_results,
)

MAX_PROCESSES = 8
MODEL_LIST_DICT = {
    '6_small': ['Qwen2.5-7B-Instruct', 'glm-4-9b-chat', 'Qwen2-7B-Instruct', 'Meta-Llama-3.1-8B-Instruct',
                'internlm2_5-7b-chat', 'gemma-2-9b-it'],
    '4_mid': ['Meta-Llama-3.3-70B-Instruct', 'Qwen2.5-32b-Instruct', 'gemma_3_27b_it', 'QwQ-32B'],
    '6_mid': ['Meta-Llama-3.3-70B-Instruct', 'Qwen2.5-32b-Instruct', 'gemma_3_27b_it', 'QwQ-32B', 'EXAONE-Deep-32B', ],
    '4_mid+2_small': ['Meta-Llama-3.3-70B-Instruct', 'Qwen2.5-32b-Instruct', 'gemma_3_27b_it', 'QwQ-32B',
                      'Qwen2.5-7B-Instruct', 'internlm2_5-7b-chat']
}


def build_prompt(question):
    messages = [{"role": "system",
                 "content": "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."}]
    messages.append({"role": "user", "content": get_generic_question_template_answer(question)})
    return messages


def build_prompt_nemotron_thinking(question, thinking="on"):
    messages = [{"role": "system",
                 "content": f"detailed thinking {thinking}"}]
    messages.append({"role": "user",
                     "content": f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n{get_generic_question_template_answer(question)}"})
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


def extract_code(model_output: str):
    outputlines = model_output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
        # return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])
    return "\n".join(outputlines[indexlines[-2] + 1: indexlines[-1]])


def transfer_format(result):
    new_result = [
        {
            'question_id': i,
            'question': r['question_content'],
            'pred': r['code_list'][0],
            'model_response': r['output_list'][0],
            'is_correct': r['graded_list'][0],
            'question_title': r['question_title'],
            'platform': r['platform'],
            'ori_question_id': r['question_id'],
            "contest_id": r['contest_id'],
            "contest_date": r['contest_date'],
            "starter_code": r['starter_code'],
            "difficulty": r['difficulty'],
            "metadata": r['metadata'][0]
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

    benchmark = load_code_generation_dataset(
        "release_v6",
        start_date=QUESTION_BANK_START_DATE,
        end_date=QUESTION_BANK_END_DATE,
    )

    benchmark = sorted(benchmark, key=lambda x: x.question_id)

    # benchmark = benchmark[:4]

    result_dir = os.path.join('result', dir_name, model_name)
    result_file = os.path.join(result_dir, f'result_cache.json')
    result_format_file = os.path.join(result_dir, f'result.json')
    eval_all_file = os.path.join(result_dir, f'result_eval_all.json')
    summary_file = os.path.join(result_dir, f'summary.json')
    final_res = []
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    # f_result_handle = jsonlines.Writer(open(result_file, 'w', encoding='utf-8'))

    old_save_results = []
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            old_save_results = json.load(f)

    seen_question_ids = set()
    old_save_results = [
        instance
        for instance in old_save_results
        if instance["output_list"] and instance["question_id"] not in seen_question_ids
           and not seen_question_ids.add(instance["question_id"])
    ]

    old_save_results_question_ids = [
        instance["question_id"] for instance in old_save_results
    ]
    remaining_benchmark = [
        instance
        for instance in benchmark
        if instance.question_id not in old_save_results_question_ids
    ]

    examples = remaining_benchmark
    results = []
    for id, example in tqdm(enumerate(examples), total=len(examples)):
        question = example
        print(f"Question: {question.question_content}")
        if 'Llama-3_3-Nemotron-Super-49B-v1' == model:
            messages = build_prompt_nemotron_thinking(question)
        else:
            messages = build_prompt(question)
        response = generate_general(model, messages, max_tokens, 0.7)
        # response = remove_think(response)
        # pred = extract_code(response)
        results.append([response])

        # save
        model_fake = LanguageModelStore['Qwen/QwQ-32B']

        combined_results = combine_results(
            Scenario.codegeneration, results, model_fake, False
        )

        save_results = [
            instance.insert_output(outputs_list, extracted_list)
            for instance, (outputs_list, extracted_list) in zip(
                remaining_benchmark, combined_results
            )
        ]

        save_results += old_save_results

        save_results, combined_results = sort_and_extract_save_results(
            Scenario.codegeneration, save_results
        )

        with open(result_file, "w") as f:
            json.dump(save_results, f, indent=4)

    # save
    model_fake = LanguageModelStore['Qwen/QwQ-32B']

    combined_results = combine_results(
        Scenario.codegeneration, results, model_fake, False
    )

    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(
            remaining_benchmark, combined_results
        )
    ]

    save_results += old_save_results

    save_results, combined_results = sort_and_extract_save_results(
        Scenario.codegeneration, save_results
    )

    with open(result_file, "w") as f:
        json.dump(save_results, f, indent=4)

    eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
    generations = [[remove_think(response) for response in extracted] for _, extracted in combined_results]

    assert len(eval_samples) == len(generations)

    metrics = codegen_metrics(
        eval_samples,
        generations,
        k_list=[1]
    )
    graded = extract_instance_results(metrics[1])

    if metrics:
        metadatas = metrics[2]
    else:
        metadatas = [[] for _ in benchmark]
    save_eval_results = [
        instance.insert_output_evaluation(
            outputs_list, extracted_list, graded_list, metadata=meta
        )
        for instance, (outputs_list, extracted_list), graded_list, meta in zip(
            benchmark, combined_results, graded, metadatas
        )
    ]

    with open(summary_file, "w") as f:
        json.dump(metrics, f, indent=4)

    with open(eval_all_file, "w") as f:
        json.dump(save_eval_results, f, indent=4)

    with open(result_format_file, "w") as f:
        result_format = transfer_format(save_eval_results)
        for r in result_format:
            f.write(json.dumps(r) + '\n')

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
