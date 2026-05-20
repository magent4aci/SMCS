import jsonlines
import os
from openai import OpenAI
from tqdm import tqdm
from collections import Counter
import json
import sys
from pathlib import Path
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "global_utils").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from global_utils import generate_general

client = OpenAI(
    api_key='your api key',  # Fill in authorization
    base_url="your base url"
)

model_list = ['Qwen2.5-32b-Instruct', 'Meta-Llama-3.3-70B-Instruct', 'HuatuoGPT-o1-72B',
              'Llama-3_3-Nemotron-Super-49B-v1', 'internlm2_5-20b-chat', 'R1-distill-llama32b',
              'Qwen2.5-72b-Instruct', 'Qwen2.5-Coder-32b-Instruct', 'QwQ-32B', 'Qwen3-32B', 'GLM-Z1-32B-0414',
              'R1-distill-llama70b', 'TeleChat2-35B-32K', 'EXAONE-Deep-32B', 'gemma_3_27b_it']

response_path = './result/'


def get_keywords(output):
    keywords = output.split("Keywords:")[-1].split(",")
    new_keywords = []
    for i in keywords:
        if len(i) <= 20:
            new_keywords.append(i)
    new_keywords = [i.strip().lower().replace(".", "") for i in new_keywords]
    return new_keywords


def create_keywords(model):
    path = os.path.join(response_path, 'main_exp_8k', 'Qwen2.5-7B-Instruct_15_large_majority_voting', 'result.json')
    save_path = os.path.join(response_path, 'question_keywords.jsonl')
    fw = jsonlines.open(save_path, 'w')
    with jsonlines.Reader(open(path, 'r', encoding='utf-8')) as reader:
        for q in tqdm(reader):
            sample = {}
            question = q['question']
            keyword_prompts = [
                f"Question: {question}\n"
                f"What are the core knowledge, subjects or skills needed to solve this problem? "
                f"List 2-5 keywords separated in comma. "
                f"Example keywords: psychology, virology, behavioral theory, microbiology, "
                f"diplomacy, political science, property law, finance, business. "
                f"Give ONLY the keywords, no other words or explanation. "
                f"Follow this format: Keywords: <keyword1>, <keyword2>..."
            ]
            messages = [{"role": "user", "content": str(keyword_prompts)}]
            for i in range(5):
                # response = client.chat.completions.create(
                #     model="Qwen/Qwen2.5-7B-Instruct",
                #     messages=messages,
                # )
                # keywords = response.choices[0].message.content
                keywords = generate_general('Qwen2.5-7B-Instruct', messages)
                if "keywords" not in sample:
                    sample["keywords"] = get_keywords(keywords)
                else:
                    sample["keywords"].extend(get_keywords(keywords))

            sample["keywords"] = [k for k in sample["keywords"] if len(k) <= 20]
            sample["keywords"] = [k for k, count in Counter(sample["keywords"]).items() if count > 1]
            sample['question'] = question
            sample['question_id'] = q['question_id']
            fw.write(sample)


def create_model_profiles():
    keywords_path = os.path.join(response_path, 'question_bank_keywords.jsonl')
    keywords_dict = {}
    all_keywords = {}
    with jsonlines.Reader(open(keywords_path, 'r', encoding='utf-8')) as reader:
        for q in tqdm(reader):
            keywords_dict[q['question_id']] = q['keywords']
            for k in q['keywords']:
                if k not in all_keywords.keys():
                    all_keywords[k] = 1
                else:
                    all_keywords[k] += 1

    ####### filter keywords ########
    save_keywords = []
    for k in all_keywords:
        if all_keywords[k] > 5:
            save_keywords.append(k)
    print(save_keywords)

    for model in model_list:
        model_profile = {}
        path = os.path.join(response_path, model, 'result.json')
        save_path = os.path.join(response_path, model, 'model_profile.json')
        with jsonlines.Reader(open(path, 'r', encoding='utf-8')) as reader:
            for q in tqdm(reader):
                question_id = q['question_id']
                keywords = keywords_dict[question_id]
                if keywords is []:
                    continue
                if 'is_correct' in q.keys():
                    is_correct = q['is_correct']
                else:
                    is_correct = q['pred'] == q['answer']
                for i in keywords:
                    if i not in save_keywords:
                        continue
                    if i not in model_profile:
                        model_profile[i] = 0
                    if is_correct:
                        model_profile[i] += 1
                    else:
                        model_profile[i] -= 1
        with open(save_path, 'w') as f:
            json.dump(model_profile, f)


if __name__ == '__main__':
    model = 'Qwen2.5-32b-Instruct'
    create_keywords(model)
