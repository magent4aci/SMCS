import argparse
import multiprocessing
import importlib
import importlib.util
from pathlib import Path
import sys


def _load_runtime_config_module():
    runtime_config_path = Path(__file__).resolve().parent / "global_utils" / "runtime_config.py"
    spec = importlib.util.spec_from_file_location("smcs_runtime_config", runtime_config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load runtime config helpers from `{runtime_config_path}`.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_runtime_config = _load_runtime_config_module()
repo_root = _runtime_config.repo_root
benchmark_dir = _runtime_config.benchmark_dir
resolve_benchmark_relative_path = _runtime_config.resolve_benchmark_relative_path


def load_dataset_module(dataset_name):
    dataset_root = benchmark_dir(dataset_name)
    dataset_module_path = dataset_root / "run_exp.py"
    if not dataset_module_path.exists():
        raise FileNotFoundError(f"Cannot find dataset entrypoint at `{dataset_module_path}`.")
    for path in (repo_root(), dataset_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    # Use a normal importable module name so spawned Pool workers can unpickle
    # dataset task instances by importing the same module in a fresh process.
    return importlib.import_module(f"benchmarks.{dataset_name}.run_exp")

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
        default="15_large",
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
    # prior_x (remain x)
    parser.add_argument(
        "--ref_sample",
        type=str,
        default='all'
    )
    # [raw_moa, rag_moa_x]
    parser.add_argument(
        "--mode",
        type=str,
        default='raw_moa'
    )
    #
    parser.add_argument(
        "--ppl_coef",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--k",
        type=int,
        default=400,
        help="RAG top-k for question bank retrieval"
    )
    parser.add_argument(
        "--question_bank",
        type=str,
        default='8d'
    )
    parser.add_argument(
        "--question_bank_path",
        type=str,
        default='',
        help="Optional explicit path to a question bank JSON file. Overrides the alias lookup.",
    )
    parser.add_argument(
        "--dataset",
        choices=['MMLU-PRO', 'AIME', 'GPQA', 'IFEval', 'LiveCodeBench', 'LiveMathBench', 'MATH', 'MBPP', 'MedMCQA', 'HumanEval', 'FinQA'],
        type=str,
        default='MMLU-PRO'
    )
    parser.add_argument(
        "--exp_suffix",
        type=str,
        default=''
    )
    # for more experiments
    parser.add_argument(
        "--cache_exp",
        type=str,
        default="")
    args = parser.parse_args()
    dataset_name = args.dataset if args.dataset != 'MMLU-PRO' else 'MMLU'
    dataset_functions = load_dataset_module(args.dataset)
    data_path = resolve_benchmark_relative_path(args.dataset, dataset_functions.data_path)
    model_List_map = dataset_functions.model_List_map
    task_class = getattr(dataset_functions, f'Moa{dataset_name}')
    model = args.model
    max_tokens = args.max_tokens
    agg_max_tokens = args.agg_max_tokens
    model_list_str = args.model_list
    use_sc = args.use_sc
    sc_posi = args.sc_posi
    N = args.N
    ppl_coef = args.ppl_coef
    k = args.k
    ref_sample = args.ref_sample
    question_bank = args.question_bank
    question_bank_path = args.question_bank_path or None
    exp_suffix = args.exp_suffix
    mode = args.mode
    cache_exp = args.cache_exp
    task = task_class(model, model_list_str, model_List_map, data_path, max_tokens, mode=mode, use_sc=use_sc,
                 N=N, max_process=8, sc_posi=sc_posi, ref_sample=ref_sample,
                      ppl_coef=ppl_coef, agg_max_tokens=agg_max_tokens,
                      question_bank=question_bank, dataset=args.dataset, exp_suffix=exp_suffix, cache_exp=cache_exp,
                      k=k, question_bank_path=question_bank_path)
    task.run()
