"""
The fast api implementation of LLM
Using VLLM framework to adapt for multiple mainstream LLM architectures
"""
from fastapi import FastAPI, Request
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
import uvicorn
import subprocess
import re



os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# the

# Command line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--port", default=6006)
parser.add_argument("--max_tokens", type=int, default=32768)
parser.add_argument("--rope_scaling", action='store_true')

args = parser.parse_args()
gpu_num = torch.cuda.device_count()
# Create the FastAPI application
app = FastAPI()

# GPU cleanup function
def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

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
    for k in request[0]:
        if k != 'messages' and k in dir(sampling_params):
            setattr(sampling_params, k, request[0][k])
    # use the same sampling params for all requests
    model_inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response_raw = model.generate(model_inputs, sampling_params=sampling_params)
    if sampling_params.logprobs is not None:
        answer = [{
            "response": r.outputs[0].text,
            # This is a mean token logprob. Keep the legacy key for backward compatibility.
            "mean_logprob": r.outputs[0].cumulative_logprob / (len(r.outputs[0].token_ids) + 1e-8),
            "cumulative_logprob": r.outputs[0].cumulative_logprob / (len(r.outputs[0].token_ids) + 1e-8),
        } for r in response_raw]
    else:
        answer = [{"response": r.outputs[0].text} for r in response_raw]
    return answer

# Initialize the model
def init_model():
    model_name_or_path = args.model_name_or_path
    if args.rope_scaling:
        engine_args = {"rope_scaling": {
                    "factor": 4.0,
                    "original_max_position_embeddings": 32768,
                    "type": "yarn",
                    "rope_type": "yarn"
                    }}
    else:
        engine_args = {}
    try:
        model = LLM(model_name_or_path, task="generate", dtype='bfloat16', tensor_parallel_size=gpu_num, trust_remote_code=True, **engine_args)
    except:
        model = LLM(model_name_or_path, dtype='bfloat16', tensor_parallel_size=gpu_num,
                    trust_remote_code=True, **engine_args)
    sampling_params = SamplingParams(max_tokens=args.max_tokens, presence_penalty=1.05, temperature=0.7)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    # For robustness
    if isinstance(tokenizer, bool):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)
    return model, sampling_params, tokenizer


# The main function for POST request
@app.post("/")
async def create_item(request: Request):
    request_content_list = []
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    response_content = get_response(args.model, args.sampling_params, args.tokenizer, [json_post_list])
    return response_content


# Main server function
def main():
    model, sampling_params, tokenizer = init_model()
    args.model = model
    args.sampling_params = sampling_params
    args.tokenizer = tokenizer
    model_name = os.path.basename(args.model_name_or_path)
    server_name = model_name + '-' + str(uuid.uuid4()).split('-')[0]
    args.server_name = server_name
    print(f'Server {server_name} started and waiting for requests!')
    # Start FastAPI
    # 6006 endpoint
    uvicorn.run(app, host='0.0.0.0', port=int(args.port), workers=1)  # Start app on specified host and port


if __name__ == '__main__':
    main()
