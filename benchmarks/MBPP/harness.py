import json
import os
import re  
import subprocess  
import argparse

from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from execute import check_correctness


def read_data(path):
    with open(path, 'r') as f:
        ds = f.readlines()
    data = [json.loads(d) for d in ds]
    return data


def harness_mbpp(source_path, save_path):
    """
    Main function to handle command-line arguments and run the evaluation process.
    """
    
    # # Read the problems data
    ds = load_dataset("google-research-datasets/mbpp")
    problems = list(ds['test'])
    problems_dict = {p['task_id']: p for p in problems}
    # # Read the generated code
    gen_code_file =read_data(Path(source_path))
    # gen_code_file = [{'task_id': d['task_id'], 'solution':[d['solution']]} for d in gen_code_file]
    exec_result = []
    is_passed = []
    for gen_code in tqdm(gen_code_file):
        completion, task_id = gen_code['solution'], gen_code['task_id']
        # if task_id == 31:
        #     print(1)
        check_result = check_correctness(problem=problems_dict[task_id], completion=completion, timeout=500,
                                         completion_id=task_id)
        exec_result.append(check_result)
        is_passed.append(check_result['passed'])
    result = {'pass_rate': sum(is_passed) / len(is_passed), 'exec_result': exec_result}
    with open(Path(save_path), 'w') as f:
        json.dump(result, f, indent=4)
    return sum(is_passed) / len(is_passed)