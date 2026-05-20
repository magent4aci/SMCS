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
import numpy as np
import torch.nn.functional as F
from global_utils.embedding_models import auto_get_em, get_em_model_path

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

# the

# Command line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='Linq-Embed-Mistral')
parser.add_argument("--port", default=6006)

args = parser.parse_args()
gpu_num = torch.cuda.device_count()
# Create the FastAPI application
app = FastAPI()

# GPU cleanup function
def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# The main function for POST request
@app.post("/")
async def create_item(request: Request):
    request_content_list = []
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    # Generate response
    response_content = args.model.obtain_embedding(json_post_list['sentences'], json_post_list['tasks'], json_post_list['batch_size'], json_post_list['max_length'])
    return response_content


# Main server function
def main():
    model = auto_get_em(args.model_name)(get_em_model_path(args.model_name), args.model_name, device='auto')
    args.model = model
    model_name = args.model_name
    server_name = model_name + '-' + str(uuid.uuid4()).split('-')[0]
    args.server_name = server_name
    print(f'Server {server_name} started and waiting for requests! at port {args.port}')
    # Start FastAPI
    uvicorn.run(app, host='0.0.0.0', port=int(args.port), workers=1)  # Start app on specified host and port


if __name__ == '__main__':
    main()
