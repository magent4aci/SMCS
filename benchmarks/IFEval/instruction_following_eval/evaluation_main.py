# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import os
from typing import Sequence

from instruction_following_eval import evaluation_lib
import json

def evaluate(input_data, input_response_data, output_dir):
  inputs = evaluation_lib.read_prompt_list(input_data)
  prompt_to_response = evaluation_lib.read_prompt_to_response_dict(input_response_data)

  # get instruction following results
  for func, output_file_name in [
      (evaluation_lib.test_instruction_following_strict, "eval_results_strict"),
      (evaluation_lib.test_instruction_following_loose, "eval_results_loose"),
  ]:
    outputs = []
    for inp in inputs:
      outputs.append(func(inp, prompt_to_response))
    follow_all_instructions = [o.follow_all_instructions for o in outputs]
    accuracy = sum(follow_all_instructions) / len(outputs)

    output_file_name = os.path.join(
        output_dir, output_file_name + ".jsonl"
    )
    evaluation_lib.write_outputs(output_file_name, outputs)

    # Prints instruction following accuracy report.
    print("=" * 64)
    print(f"{output_file_name} Accuracy Scores:")
    
    prompt_level_acc, instruction_level_acc = evaluation_lib.print_report(outputs)
    result_path = output_file_name.replace(".jsonl", '_summary.json')
    with open(result_path, 'w') as f:
        json.dump({"prompt_level_acc": prompt_level_acc, "instruction_level_acc": instruction_level_acc}, f)

