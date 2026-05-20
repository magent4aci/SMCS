import os
import re
import sys
import json
import multiprocessing
import jsonlines
from tqdm import tqdm

from pathlib import Path
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "global_utils").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from global_utils import MoaBase, async_generate_general, async_raw_moa_api, model_List_map
import argparse
from finqa_grader import grade_finqa_answer

data_path = './dataset/FinQA_test.json'

PROMPT_TEMPLATE = """Solve the following problem step by step. The last line of your response should only contain your final answer inside a \\boxed{{}} command.

{question_content}

Remember to put your final answer on the last line using the format \\boxed{{$ANSWER}} where $ANSWER is the answer to the problem."""


def list_to_simple_table(table_data):
    if not table_data or not isinstance(table_data, list):
        return ""
    table_str = ""
    for row in table_data:
        row_str = [str(cell).strip() for cell in row]
        table_str += "| " + " | ".join(row_str) + " |\n"
    return table_str


def build_prompt(each):
    pre_text = each.get('pre_text', [])
    post_text = each.get('post_text', [])
    table = each.get('table', [])
    question = each.get('question', '')

    if isinstance(pre_text, str):
        pre_text = [pre_text]
    if isinstance(post_text, str):
        post_text = [post_text]

    pre_str = ''.join(pre_text) if pre_text else ''
    post_str = ''.join(post_text) if post_text else ''
    table_str = list_to_simple_table(table)

    question_content = f"{pre_str}\n\nHere is the table:\n{table_str}\n\n{post_str}\n\nBased on the content and the table, please answer the following question: {question}"
    return PROMPT_TEMPLATE.format(question_content=question_content)


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
    return string[idx:right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left and s[-1] == "}"
        return s[len(left):-1]
    except Exception:
        return None


def _load_finqa_data(path):
    """Load FinQA data; supports JSON array or JSONL format"""
    with open(path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    with open(path, 'r', encoding='utf-8') as f:
        if first_line.strip().startswith('['):
            return json.load(f)
        examples = []
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
        return examples


class MoaFinQA(MoaBase):
    def build_dataset(self):
        examples = _load_finqa_data(self.data_path)
        examples_new = []
        for i, example in enumerate(examples):
            if 'question_id' not in example:
                example['question_id'] = i
            examples_new.append(example)
        self.test_data = examples_new
        self.val_data = None
        self.test_data_num = len(self.test_data)
        return examples_new

    def build_res_and_sum_file(self):
        output_res_path = os.path.join(self.result_dir, "result.json")
        output_summary_path = os.path.join(self.result_dir, "summary.json")
        self.output_res_path = output_res_path
        self.output_summary_path = output_summary_path
        return output_res_path, output_summary_path

    def build_cache(self):
        max_token_ref = self.max_tokens_list[-1]
        res_exp_name = 'single_agent_8k' if max_token_ref == 8192 else 'single_agent'
        if self.cache_dict is None:
            self.cache_dict = {}
            for m in self.model_list:
                result_path_m = self.benchmark_result_path(res_exp_name, m, 'result.json')
                if os.path.exists(result_path_m):
                    self.cache_dict[m] = {}
                    with jsonlines.Reader(open(result_path_m, 'r', encoding='utf-8')) as reader:
                        for q in reader:
                            self.cache_dict[m][q['question_id']] = q["model_response"]

    def build_messages(self, each):
        prompt = build_prompt(each)
        messages = [
            {"role": "system", "content": "You are a financial analysis expert. Solve the following numerical reasoning problem based on the given table and context. Provide step-by-step reasoning and put your final numerical answer in \\boxed{}."},
            {"role": "user", "content": prompt}
        ]
        return messages

    def extract_answer(self, response):
        ext_ans = remove_boxed(last_boxed_only_string(response))
        if ext_ans is not None:
            return ext_ans.strip()
        patterns = [
            r'[Aa]nswer\s*:\s*([^\n]+)',
            r'[Aa]nswer\s+is\s+([+-]?\d*\.?\d+)',
            r'[Aa]nswer\s+is\s+(.+)',
            r'[Ff]inal\s+[Aa]nswer\s*:\s*([^\n]+)',
            r'([+-]?\d*\.?\d+)'
        ]
        for pattern in patterns:
            match = re.findall(pattern, response)
            if match:
                result = match[-1].strip()
                if result and result[-1] == '.':
                    result = result[:-1]
                return result
        return ""

    def grade_answer(self, pred, gt):
        return grade_finqa_answer(pred, gt)

    def get_question_id(self, each):
        return each['question_id']

    def get_question(self, each):
        return each.get('question', '')

    async def async_generate_general_cache(self, question_id, model, messages, max_tokens, temperature, streaming):
        if self.cache_dict is None or model not in self.cache_dict or question_id not in self.cache_dict[model]:
            raise NotImplementedError
        return self.cache_dict[model][question_id]

    def prepare_continue(self):
        if os.path.exists(self.output_res_path):
            with jsonlines.Reader(open(self.output_res_path, 'r', encoding='utf-8')) as f:
                self.done_list = list(f)
            self.done_question_id = [q['question_id'] for q in self.done_list]
        if os.path.exists(self.output_summary_path):
            with open(self.output_summary_path, "r", encoding='utf-8') as fo:
                self.done_sum_dict = json.load(fo)

    def run(self):
        result_dir = self.build_result_dir()
        self.build_cache()
        test_df = self.build_dataset()
        final_res = []
        cnt = 0
        test_data = test_df
        test_data_num = len(self.test_data)
        output_res_path, output_summary_path = self.build_res_and_sum_file()
        data_index = list(range(0, test_data_num, self.max_process)) + [self.test_data_num]
        if self.mode == 'raw_moa':
            wrap_function = self.wrap_raw_moa_test
        elif 'rag_moa' in self.mode:
            wrap_function = self.wrap_rag_moa_test
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Supported: raw_moa, rag_moa_*")

        self.prepare_continue()
        test_data_question_id = [t['question_id'] for t in test_data]
        cnt = len(self.done_question_id)
        if len(self.done_question_id) != 0:
            have_done_corr = sum([q['is_correct'] for q in self.done_list])
            final_res.extend([False] * int(len(self.done_list) - have_done_corr) + [True] * int(have_done_corr))

        for i in tqdm(range(len(data_index) - 1)):
            data_id_range = list(range(data_index[i], data_index[i + 1]))
            data_id_range_filter = [d_i for d_i in data_id_range if test_data_question_id[d_i] not in self.done_question_id]
            if len(data_id_range_filter) == 0:
                continue
            tasks = [[data_id, test_data, None, self.model_list, self.model, self.max_tokens_list,
                      self.use_sc, self.ppl_coef, self.N, self.sc_posi, self.ref_sample] for data_id in data_id_range_filter]
            with multiprocessing.Pool(processes=min(len(tasks), self.max_process)) as pool:
                response_pred_dict_list = pool.starmap(wrap_function, tasks)

            cnt += len(data_id_range_filter)
            for response_pred_dict, data_id_range_i in zip(response_pred_dict_list, data_id_range_filter):
                each = test_data[data_id_range_i]
                gt = each.get('answer') or each.get('final_result', '')
                is_correct = self.grade_answer(response_pred_dict['pred'], gt)
                final_res.append(is_correct)
                res_dict = {
                    'question_id': each['question_id'], 'question': each.get('question', ''),
                    'answer': gt, 'pred': response_pred_dict['pred'], 'is_correct': is_correct,
                    'response': response_pred_dict['response'], 'sc_memory': response_pred_dict.get('sc_memory'),
                    'n_response': response_pred_dict.get('n_response'), 'ref_dict': response_pred_dict.get('ref_dict'),
                }
                for k in response_pred_dict:
                    if k not in res_dict:
                        res_dict[k] = response_pred_dict[k]

                sum_dict = {'corr': sum(final_res), 'wrong': len(final_res) - sum(final_res),
                            'acc': sum(final_res) / len(final_res), 'schedule': f'{cnt}/{test_data_num}'}
                with open(output_summary_path, "w", encoding='utf-8') as fo:
                    fo.write(json.dumps(sum_dict))
                with jsonlines.Writer(open(output_res_path, 'a', encoding='utf-8')) as f_result_handle:
                    f_result_handle.write(res_dict)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--agg_max_tokens", type=int, default=8192)
    parser.add_argument("--model_list", type=str, default="4_mid+2_small")
    parser.add_argument("--model", type=str, default="Meta-Llama-3.3-70B-Instruct")
    parser.add_argument("--sc_posi", type=str, default="agg")
    parser.add_argument("--use_sc", action='store_true')
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--ref_sample", type=str, default='all')
    parser.add_argument("--mode", type=str, default='raw_moa')
    parser.add_argument("--ppl_coef", type=float, default=0.0)
    parser.add_argument("--question_bank", type=str, default='8d')
    parser.add_argument("--cache_exp", type=str, default="")
    args = parser.parse_args()
    task = MoaFinQA(args.model, args.model_list, model_List_map, data_path, args.max_tokens, mode=args.mode,
                    use_sc=args.use_sc, N=args.N, max_process=8,
                    sc_posi=args.sc_posi, ref_sample=args.ref_sample, ppl_coef=args.ppl_coef,
                    agg_max_tokens=args.agg_max_tokens, question_bank=args.question_bank,
                    dataset='FinQA', cache_exp=args.cache_exp)
    task.run()
