"""
The fast and local file system implementation of LLM
Using VLLM framework to adapt for multiple mainstream LLM architectures
"""
import argparse
import fcntl
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
import datetime
import torch
import time
import uuid
import sys
import os

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# the

# Command line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--request_folder", type=str, default="your path to queue_example/shared/requests")
parser.add_argument("--response_folder", type=str, default="your path to queue_example/shared/responses")
parser.add_argument("--online_log_path", type=str, default="your path to queue_example/shared/online_server.json")
parser.add_argument("--fps", type=float, default=0.3)
parser.add_argument("--model_name_or_path", type=str, default="your path to Qwen2-7B-Instruct")
parser.add_argument("--batch_size", type=int, default=2)

args = parser.parse_args()
all_lock = []
gpu_num = torch.cuda.device_count()

# GPU cleanup function
def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# Acquire file lock
def acquire_lock(file_path):
    lock_file = open(file_path, 'r+')  # Open file in append mode
    fcntl.flock(lock_file, fcntl.LOCK_EX)  # Lock file exclusively
    return lock_file

# Release file lock
def release_lock(lock_file):
    fcntl.flock(lock_file, fcntl.LOCK_UN)  # Unlock file
    lock_file.close()

# Function to process response
def get_response(model, sampling_params, tokenizer, request):
    """
    :param model: str
    :param sampling_params: sampling params
    :param tokenizer:
    :param request: List[dict]
    :return: List[dict]
    """
    messages = [r.get('messages') for r in request]
    if 'max_tokens' in request[0].keys() or 'temperature' in request[0].keys():
        sampling_params = SamplingParams(max_tokens=request[0]['max_tokens'], presence_penalty=1.05,
                                         temperature=request[0]['temperature'])
    # use the same sampling params for all requests
    model_inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response_raw = model.generate(model_inputs, sampling_params=sampling_params)
    answer = [{"response": r.outputs[0].text} for r in response_raw]
    return answer

# Initialize the model
def init_model():
    model_name_or_path = args.model_name_or_path
    model = LLM(model_name_or_path, task="generate", dtype='bfloat16', tensor_parallel_size=gpu_num)
    sampling_params = SamplingParams(max_tokens=4096, presence_penalty=1.05, temperature=0.7)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    return model, sampling_params, tokenizer

# Main server function
def main():
    model, sampling_params, tokenizer = init_model()
    model_name = os.path.basename(args.model_name_or_path)
    server_name = model_name + '-' + str(uuid.uuid4()).split('-')[0]
    args.server_name = server_name
    print(f'Server {server_name} started and waiting for requests!')

    # Register the server
    if not os.path.exists(args.online_log_path):
        with open(args.online_log_path, 'w') as f:
            json.dump([], f)

    with open(args.online_log_path, 'r') as f:
        all_online_server = json.load(f)
    with open(args.online_log_path, 'w') as f:
        all_online_server.append(args.server_name)
        json.dump(all_online_server, f)
        print(f"Added online server {args.server_name}!")
    # begin the main loop
    while True:
        # Read requests
        request_id_list, request_content_list = [], []
        request_file_list = list(filter(lambda x: x.endswith('json') and model_name in x, os.listdir(args.request_folder)))
        for k in request_file_list:
            with open(os.path.join(args.request_folder, k), 'r') as f:
                request_content_list.append(json.load(f))
            request_id_list.append('.'.join(k.split('.')[:-1]))

        if len(request_id_list) != 0:
            # Process each request
            all_requests_num = len(request_id_list)
            if all_requests_num < args.batch_size:
                batch_index = [0, all_requests_num]
            else:
                batch_index = list(range(0, all_requests_num, args.batch_size)) + [all_requests_num]
            batch_num = len(batch_index) - 1
            for i in range(batch_num):
                st_id, end_id = batch_index[i], batch_index[i+1]
                # Generate response
                response_content = get_response(model, sampling_params, tokenizer, request_content_list[st_id: end_id])
                # Write response and remove the request
                for j, request_id_j in enumerate(request_id_list[st_id: end_id]):
                    with open(os.path.join(args.response_folder, str(request_id_j) + '.json'), 'w') as f:
                        json.dump(response_content[j], f)
                    os.remove(os.path.join(args.request_folder, str(request_id_j) + '.json'))
        time.sleep(args.fps)


if __name__ == '__main__':
    main()
