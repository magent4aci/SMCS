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
from global_utils.data_utils import load_jsonl_records
import argparse
from math_equivalence import is_equiv

data_path = './dataset/test-500.jsonl'

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


class MoaMATH(MoaBase):
    def build_dataset(self):
        examples = load_jsonl_records(self.data_path)
        new_examples = []
        # record the question id
        for i, example in enumerate(examples):
            example['question_id'] = i
            new_examples.append(example)
        self.test_data = new_examples
        self.val_data = None
        self.test_data_num = len(self.test_data)
        return new_examples

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
        messages = [{"role": "system",
                     "content": "You are a math problem solver. Please solve the following math problem. Be sure to explain your solution in detail. The numerical values in the answer should be surrounded by \\boxed{}. The final answer should start with 'The answer is' and give the conclusion directly. Do not add any extra content."}]
        messages.append({"role": "user", "content": f"question:{each['problem']}"})
        return messages

    def extract_answer(self, response):
        ext_ans = remove_boxed(last_boxed_only_string(response))
        if ext_ans is None:
            # if not match, select 'A'
            print(f'Not match, use final match')
            patterns = [r'[Aa]nswer is ([+-]?\d*\.?\d+)', r'[Aa]nswer is (.*)', r'is ([+-]?\d*\.?\d+)',
                        r'([+-]?\d*\.?\d+)']
            for pattern in patterns:
                match = re.findall(pattern, response)
                if match:
                    result = match[-1]
                    if pattern == '[Aa]nswer is (.*)' and result[-1] == '.':
                        return result[:-1]
                    return result
            return str(-114514)
        return ext_ans

    def get_question_id(self, each):
        return each['question_id']

    def get_question(self, each):
        return each['problem']

    async def async_generate_general_cache(self, question_id, model, messages, max_tokens, temperature, streaming):
        # load the cache result
        if self.cache_dict is None:
            print(f"question_id: {question_id} is not prepared")
            # return await async_generate_general(model, messages, max_tokens, temperature, streaming)
            raise NotImplementedError
        elif model not in self.cache_dict:
            # return await async_generate_general(model, messages, max_tokens, temperature, streaming)
            raise NotImplementedError
        elif question_id not in self.cache_dict[model]:
            print(f"question_id: {question_id} is not prepared")
            # return await async_generate_general(model, messages, max_tokens, temperature, streaming)
            raise NotImplementedError
        else:
            return self.cache_dict[model][question_id]
        # return await async_generate_general(model, messages, max_tokens, temperature, streaming)

    def prepare_continue(self):
        if os.path.exists(self.output_res_path):
            with jsonlines.Reader(open(self.output_res_path, 'r', encoding='utf-8')) as f:
                self.done_list = list(f)
            self.done_question_id = [q['question_id'] for q in self.done_list]
        if os.path.exists(self.output_summary_path):
            with open(self.output_summary_path, "r") as fo:
                self.done_sum_dict = json.load(fo)


    def run(self):
        result_dir = self.build_result_dir()
        self.build_cache()
        test_df = self.build_dataset()
        final_res = []
        cnt = 0
        test_data = test_df
        test_data_num = len(test_data)
        output_res_path, output_summary_path = self.build_res_and_sum_file()
        # f_result_handle = jsonlines.Writer(open(output_res_path, 'a', encoding='utf-8'))
        data_index = list(range(0, test_data_num, self.max_process)) + [self.test_data_num]
        if self.mode == 'raw_moa':
            wrap_function = self.wrap_raw_moa_test
        elif 'rag_moa' in self.mode:
            wrap_function = self.wrap_rag_moa_test
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Supported: raw_moa, rag_moa_*")

        # for continue
        self.prepare_continue()
        test_data_question_id = [t['question_id'] for t in test_data]
        cnt = len(self.done_question_id)
        if len(self.done_question_id) != 0:
            have_done_corr = sum([q['is_correct'] for q in self.done_list])
            final_res.extend([False] * int(len(self.done_list) - have_done_corr) + [True] * int(have_done_corr))

        for i in tqdm(range(len(data_index) - 1)):
            # build the task for data parallel
            data_id_range = list(range(data_index[i], data_index[i + 1]))
            data_id_range_filter = [d_i for d_i in data_id_range if test_data_question_id[d_i] not in self.done_question_id]
            if len(data_id_range_filter) == 0:
                continue
            tasks = [[data_id, test_data, None, self.model_list, self.model, self.max_tokens_list,
                      self.use_sc, self.ppl_coef, self.N, self.sc_posi, self.ref_sample] for data_id in data_id_range_filter]
            with multiprocessing.Pool(processes=min(len(tasks), self.max_process)) as pool:
                response_pred_dict_list = pool.starmap(wrap_function, tasks)
            # log the result
            cnt += len(data_id_range_filter)
            for response_pred_dict, data_id_range_i in zip(response_pred_dict_list, data_id_range_filter):
                each = test_data[data_id_range_i]
                is_correct = is_equiv(response_pred_dict['pred'], each['answer'])
                final_res.append(is_correct)
                res_dict = {'question_id': each['question_id'], 'question': each['problem'],
                            'answer': each['answer'], 'pred': response_pred_dict['pred'], 'is_correct': is_correct,
                            'response': response_pred_dict['response'], 'sc_memory': response_pred_dict['sc_memory'],
                            'n_response': response_pred_dict['n_response'], 'ref_dict': response_pred_dict['ref_dict'],
                            }
                for k in response_pred_dict:
                    if k not in res_dict:
                        res_dict[k] = response_pred_dict[k]
                sum_dict = {'corr': sum(final_res), 'wrong': len(final_res) - sum(final_res),
                            'acc': sum(final_res) / len(final_res), 'schedule': f'{cnt}/{test_data_num}'}
                with open(output_summary_path, "w") as fo:
                    fo.write(json.dumps(sum_dict))
                with jsonlines.Writer(open(output_res_path, 'a', encoding='utf-8')) as f_result_handle:
                    f_result_handle.write(res_dict)




if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192
    )
    parser.add_argument(
        "--agg_max_tokens",
        type=int,
        default=8192
    )
    parser.add_argument(
        "--model_list",
        type=str,
        default="4_mid+2_small",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Meta-Llama-3.3-70B-Instruct",
    )
    parser.add_argument(
        "--sc_posi",
        type=str,
        default="agg",
    )
    parser.add_argument(
        "--use_sc",
        action='store_true',
    )
    parser.add_argument(
        "--N",
        type=int,
        default=8
    )
    parser.add_argument(
        "--ref_sample",
        type=str,
        default='all'
    )
    parser.add_argument(
        "--mode",
        type=str,
        default='raw_moa'
    )
    parser.add_argument(
        "--ppl_coef",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--question_bank",
        type=str,
        default='8d'
    )
    # for more experiments
    parser.add_argument(
        "--cache_exp",
        type=str,
        default="")
    args = parser.parse_args()
    model = args.model
    max_tokens = args.max_tokens
    agg_max_tokens = args.agg_max_tokens
    model_list_str = args.model_list
    use_sc = args.use_sc
    sc_posi = args.sc_posi
    N = args.N
    ppl_coef = args.ppl_coef
    ref_sample = args.ref_sample
    question_bank = args.question_bank
    mode = args.mode
    cache_exp = args.cache_exp
    task = MoaMATH(model, model_list_str, model_List_map, data_path, max_tokens, mode=mode, use_sc=use_sc,
                 N=N, max_process=8, sc_posi=sc_posi, ref_sample=ref_sample,
                   ppl_coef=ppl_coef, agg_max_tokens=agg_max_tokens,
                   question_bank=question_bank, dataset='MATH', cache_exp=cache_exp)
    task.run()
