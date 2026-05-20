from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import argparse
import json
import subprocess
import os
import re
import uuid
import asyncio
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import uvicorn

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str)
parser.add_argument("--batch_size", type=int, default=64,
                    help="Number of requests to batch together")
parser.add_argument("--port", type=int, default=6006)
parser.add_argument("--max_tokens", type=int, default=4096)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=-1)
parser.add_argument("--presence_penalty", type=float, default=1.05)
parser.add_argument("--temperature", type=float, default=0.7)


parser.add_argument("--rope_scaling", action="store_true")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.4)
args = parser.parse_args()
gpu_num = torch.cuda.device_count()

# ---------------------------------------------------------------------------
# FastAPI app & globals
# ---------------------------------------------------------------------------
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
app = FastAPI()

model: LLM = None
sampling_params: SamplingParams = None
tokenizer: AutoTokenizer = None

# Async batching machinery ---------------------------------------------------
pending: List[Dict[str, Any]] = []  # {"data": payload, "future": Future}
lock = asyncio.Lock()
batch_ready = asyncio.Event()  # Wake worker immediately when batch_size is reached
FIRST_REQUEST_TS: float | None = None  # Monotonic time of first item in queue
CHECK_INTERVAL = 0.05   # Worker polling interval (seconds); too large delays batch trigger
TIMEOUT_SECONDS = 2.0   # Timeout after first request (seconds); flush current queue on timeout
_executor = None  # Thread pool; get_response runs in executor to avoid blocking event loop


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_response(model_: LLM, sampling_params_: SamplingParams,
                 tokenizer_: AutoTokenizer,
                 requests: List[Dict[str, Any]]):
    """Run one forward pass over *requests* and return list of answer dicts."""
    messages = [r["messages"] for r in requests]

    first = requests[0]
    for k in first:
        if k != "messages" and hasattr(sampling_params_, k):
            setattr(sampling_params_, k, first[k])

    inputs = tokenizer_.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
    outputs = model_.generate(inputs, sampling_params=sampling_params_)

    if sampling_params_.logprobs is not None:
        return [{
            "response": o.outputs[0].text,
            # This is a mean token logprob. Keep the legacy key for backward compatibility.
            "mean_logprob": o.outputs[0].cumulative_logprob /
                             (len(o.outputs[0].token_ids) + 1e-8),
            "cumulative_logprob": o.outputs[0].cumulative_logprob /
                                   (len(o.outputs[0].token_ids) + 1e-8)
        } for o in outputs]
    return [{"response": o.outputs[0].text} for o in outputs]


async def batch_worker():
    """Background coroutine that delivers batched inference."""
    global FIRST_REQUEST_TS
    loop = asyncio.get_event_loop()
    while True:
        # batch_ready is set when batch_size is reached, wake immediately; otherwise wait at most CHECK_INTERVAL
        try:
            await asyncio.wait_for(batch_ready.wait(), timeout=CHECK_INTERVAL)
        except asyncio.TimeoutError:
            pass
        batch_ready.clear()

        async with lock:
            if not pending:
                FIRST_REQUEST_TS = None
                continue

            queue_age = loop.time() - (FIRST_REQUEST_TS or loop.time())
            ready_by_size = len(pending) >= args.batch_size
            ready_by_time = queue_age >= TIMEOUT_SECONDS
            if not (ready_by_size or ready_by_time):
                continue

            if ready_by_size:
                batch = pending[:args.batch_size]
                del pending[:args.batch_size]
            else:
                batch = list(pending)
                pending.clear()
            if not pending:
                FIRST_REQUEST_TS = None
        # --------------- lock released --------------------------------------
        # Run in executor to avoid blocking event loop (otherwise cannot receive new requests)
        try:
            answers = await loop.run_in_executor(
                _executor,
                lambda: get_response(model, sampling_params, tokenizer,
                                    [item["data"] for item in batch]),
            )
            for item, ans in zip(batch, answers):
                item["future"].set_result([ans])
        except Exception as exc:
            for item in batch:
                item["future"].set_exception(exc)
        finally:
            torch_gc()


@app.on_event("startup")
async def _startup():
    global _executor
    _executor = ThreadPoolExecutor(max_workers=1)  # Single-threaded inference to avoid blocking event loop
    asyncio.create_task(batch_worker())
    server_name = getattr(args, 'server_name', 'unknown')
    print(f"Server {server_name} (batch={args.batch_size}) ready on port {args.port}")


@app.post("/")
async def handle_request(request: Request):
    try:
        payload = await request.json()
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()

        async with lock:
            global FIRST_REQUEST_TS
            pending.append({"data": payload, "future": fut})
            if len(pending) == 1:
                FIRST_REQUEST_TS = loop.time()
            if len(pending) >= args.batch_size:
                batch_ready.set()  # Wake worker immediately, no need to wait for CHECK_INTERVAL

        result = await fut
        # Ensure result is a list
        if not isinstance(result, list):
            result = [result]
        return JSONResponse(content=result)
    except Exception as e:
        # Log the error for debugging
        import traceback
        print(f"Error in handle_request: {str(e)}")
        print(traceback.format_exc())
        # Ensure we always return valid JSON even on error
        error_response = [{"response": f"Error: {str(e)}", "error": True}]
        return JSONResponse(content=error_response, status_code=500)

# ---------------------------------------------------------------------------
# Model initialisation & server start
# ---------------------------------------------------------------------------

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
    # max_num_seqs: vLLM default is small (e.g. 256 or lower); set explicitly for large batch
    # Otherwise even with batch_size=64, vLLM internally processes only a few sequences
    engine_args["max_num_seqs"] = max(args.batch_size, 64)
    model = LLM(model_name_or_path, dtype='bfloat16', tensor_parallel_size=gpu_num,
                trust_remote_code=True, gpu_memory_utilization=args.gpu_memory_utilization, **engine_args)
    sampling_params = SamplingParams(max_tokens=args.max_tokens, presence_penalty=args.presence_penalty,
                                     temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    # For robustness
    if isinstance(tokenizer, bool):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)
    return model, sampling_params, tokenizer


def main():
    global model, sampling_params, tokenizer
    model, sampling_params, tokenizer = init_model()

    model_name = os.path.basename(args.model_name_or_path)
    args.server_name = f"{model_name}-{uuid.uuid4().hex[:6]}"
    uvicorn.run(app, host="0.0.0.0", port=args.port, workers=1)


if __name__ == "__main__":
    main()
