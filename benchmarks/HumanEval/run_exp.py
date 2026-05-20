"""
HumanEval dataset MOA experiment entry.
Follows MedMCQA/MBPP format; supports main run_exp.py invocation.
"""
import os
import json
import multiprocessing
import jsonlines
from tqdm import tqdm

import sys
from pathlib import Path
_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "global_utils").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from global_utils import MoaBase, async_generate_general, async_raw_moa_api, model_List_map, generate_general_em
from run_api import DATASET_MAPPING, PROMPT_WRAPPER_API, build_message
from utils import post_process_humaneval, build_humaneval_solution
from execution import check_correctness

data_path = None  # HumanEval loads dynamically from evalplus/HF


def clean_answer(ans):
    """Extract code from model response"""
    ans = ans.split("```python", 1)[-1]
    return post_process_humaneval(ans[: index if (index := ans.find("```")) != -1 else len(ans)])


class MoaHumanEval(MoaBase):
    def build_dataset(self):
        raw_problem_fn, map_problem_fn = DATASET_MAPPING["humaneval"]
        raw_problems = raw_problem_fn()
        problems = list(map(map_problem_fn, raw_problems))
        for p in problems:
            p["question_id"] = p["id"]
        self.test_data = problems
        self.val_data = None
        self.test_data_num = len(self.test_data)
        self.raw_test_data = raw_problems
        return problems

    def build_res_and_sum_file(self):
        output_res_path = os.path.join(self.result_dir, "result.json")
        output_summary_path = os.path.join(self.result_dir, "summary.json")
        self.output_res_path = output_res_path
        self.output_summary_path = output_summary_path
        return output_res_path, output_summary_path

    def build_cache(self):
        max_token_ref = self.max_tokens_list[-1]
        res_exp_name = "single_agent_8k" if max_token_ref == 8192 else "single_agent"
        if self.cache_dict is None:
            self.cache_dict = {}
            for m in self.model_list:
                result_path_m = self.benchmark_result_path(res_exp_name, m, "results.jsonl")
                if os.path.exists(result_path_m):
                    self.cache_dict[m] = {}
                    with jsonlines.Reader(open(result_path_m, "r", encoding="utf-8")) as reader:
                        for q in reader:
                            self.cache_dict[m][q["task_id"]] = q.get("response", q.get("solution", ""))

    def build_if_correct_cache(self):
        max_token_ref = self.max_tokens_list[-1]
        res_exp_name = "single_agent_8k" if max_token_ref == 8192 else "single_agent"
        self.correct_cache_dict = {}
        for m in self.model_list:
            result_path_m = self.benchmark_result_path(res_exp_name, m, "eval_acc.json")
            if os.path.exists(result_path_m):
                with open(result_path_m, "r", encoding="utf-8") as f:
                    res_dict = json.load(f)
                    exec_result = res_dict.get("exec_result", res_dict)
                    self.correct_cache_dict[m] = {}
                    for q in exec_result:
                        tid = q.get("task_id", q.get("completion_id"))
                        self.correct_cache_dict[m][tid] = q.get("passed", False)

    def build_messages(self, each):
        prompt = PROMPT_WRAPPER_API.format(
            instruction=each["instruction"],
            response=each["response_prefix"],
        )
        return build_message(prompt)

    def extract_answer(self, response):
        return response

    def get_question_id(self, each):
        return each["question_id"]

    def get_question(self, each):
        return each["instruction"]

    async def async_generate_general_cache(self, question_id, model, messages, max_tokens, temperature, streaming):
        if self.cache_dict is None:
            raise NotImplementedError
        elif model not in self.cache_dict:
            raise NotImplementedError
        elif question_id not in self.cache_dict[model]:
            raise NotImplementedError
        else:
            return self.cache_dict[model][question_id]

    def prepare_continue(self):
        if os.path.exists(self.output_res_path):
            with jsonlines.Reader(open(self.output_res_path, "r", encoding="utf-8")) as f:
                self.done_list = list(f)
            self.done_question_id = [q["question_id"] for q in self.done_list]
        if os.path.exists(self.output_summary_path):
            with open(self.output_summary_path, "r") as fo:
                self.done_sum_dict = json.load(fo)

    def run(self):
        result_dir = self.build_result_dir()
        self.build_cache()
        self.build_if_correct_cache()
        test_df = self.build_dataset()
        final_res = []
        cnt = 0
        test_data = test_df
        test_data_num = len(test_data)
        output_res_path, output_summary_path = self.build_res_and_sum_file()

        data_index = list(range(0, test_data_num, self.max_process)) + [self.test_data_num]
        if self.mode == "raw_moa":
            wrap_function = self.wrap_raw_moa_test
        elif "rag_moa" in self.mode:
            wrap_function = self.wrap_rag_moa_test
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Supported: raw_moa, rag_moa_*")

        self.prepare_continue()
        test_data_question_id = [t["question_id"] for t in test_data]
        cnt = len(self.done_question_id)
        if len(self.done_question_id) != 0:
            have_done_corr = sum([q["is_correct"] for q in self.done_list])
            final_res.extend([False] * (len(self.done_list) - have_done_corr) + [True] * have_done_corr)

        for i in tqdm(range(len(data_index) - 1)):
            data_id_range = list(range(data_index[i], data_index[i + 1]))
            data_id_range_filter = [d_i for d_i in data_id_range if test_data_question_id[d_i] not in self.done_question_id]
            if len(data_id_range_filter) == 0:
                continue
            tasks = [
                [data_id, test_data, None, self.model_list, self.model, self.max_tokens_list,
                 self.use_sc, self.ppl_coef, self.N, self.sc_posi, self.ref_sample]
                for data_id in data_id_range_filter
            ]
            with multiprocessing.Pool(processes=min(len(tasks), self.max_process)) as pool:
                response_pred_dict_list = pool.starmap(wrap_function, tasks)

            cnt += len(data_id_range_filter)
            for response_pred_dict, data_id_range_i in zip(response_pred_dict_list, data_id_range_filter):
                each = test_data[data_id_range_i]
                raw_each = self.raw_test_data[data_id_range_i]

                # HumanEval correctness verification
                completion_clean = clean_answer(response_pred_dict["response"])
                solution = build_humaneval_solution(
                    completion_clean,
                    each["prompt"],
                    each["test"],
                    each["entry_point"],
                )
                if hasattr(self, "correct_cache_dict") and self.correct_cache_dict:
                    is_correct = None
                    for m in self.correct_cache_dict:
                        cached = self.cache_dict.get(m, {}).get(each["question_id"], "")
                        if cached == response_pred_dict["response"]:
                            is_correct = self.correct_cache_dict[m].get(each["question_id"])
                            break
                    if is_correct is None:
                        check_result = check_correctness(
                            task_id=each["question_id"],
                            completion_id=each["question_id"],
                            solution=solution,
                            time_out=10,
                        )
                        is_correct = check_result["passed"]
                else:
                    check_result = check_correctness(
                        task_id=each["question_id"],
                        completion_id=each["question_id"],
                        solution=solution,
                        time_out=10,
                    )
                    is_correct = check_result["passed"]

                final_res.append(is_correct)
                res_dict = {
                    "question_id": each["question_id"],
                    "question": each["instruction"],
                    "answer": "code",
                    "pred": response_pred_dict["pred"],
                    "is_correct": is_correct,
                    "response": response_pred_dict["response"],
                    "sc_memory": response_pred_dict["sc_memory"],
                    "n_response": response_pred_dict["n_response"],
                    "ref_dict": response_pred_dict["ref_dict"],
                }
                for k in response_pred_dict:
                    if k not in res_dict:
                        res_dict[k] = response_pred_dict[k]

                sum_dict = {
                    "corr": sum(final_res),
                    "wrong": len(final_res) - sum(final_res),
                    "acc": sum(final_res) / len(final_res),
                    "schedule": f"{cnt}/{test_data_num}",
                }
                with open(output_summary_path, "w") as fo:
                    fo.write(json.dumps(sum_dict))
                with jsonlines.Writer(open(output_res_path, "a", encoding="utf-8")) as f_result_handle:
                    f_result_handle.write(res_dict)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--agg_max_tokens", type=int, default=8192)
    parser.add_argument("--model_list", type=str, default="7_large")
    parser.add_argument("--model", type=str, default="Meta-Llama-3.3-70B-Instruct")
    parser.add_argument("--sc_posi", type=str, default="agg")
    parser.add_argument("--use_sc", action="store_true")
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--ref_sample", type=str, default="all")
    parser.add_argument("--mode", type=str, default="raw_moa")
    parser.add_argument("--ppl_coef", type=float, default=0.0)
    parser.add_argument("--question_bank", type=str, default="8d")
    parser.add_argument("--cache_exp", type=str, default="")
    args = parser.parse_args()

    task = MoaHumanEval(
        args.model,
        args.model_list,
        model_List_map,
        data_path,
        args.max_tokens,
        mode=args.mode,
        use_sc=args.use_sc,
        N=args.N,
        max_process=8,
        sc_posi=args.sc_posi,
        ref_sample=args.ref_sample,
        ppl_coef=args.ppl_coef,
        agg_max_tokens=args.agg_max_tokens,
        question_bank=args.question_bank,
        dataset="HumanEval",
        cache_exp=args.cache_exp,
    )
    task.run()
