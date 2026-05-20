import os
import json
import time
import requests
import copy
from openai import OpenAI
from loguru import logger
import asyncio
from global_utils.runtime_config import (
    get_section,
    is_placeholder_value,
    is_unconfigured_endpoint,
    resolve_config_path,
)

"""For the global configs"""
DEBUG = int(os.environ.get("DEBUG", "0"))
TIMEOUT = 900


class OpenAIConfig:
    def __init__(self, api_key, model_name, base_url="", api_key_env="OPENAI_API_KEY"):
        if isinstance(api_key, str):
            api_key = [api_key] if api_key else []
        self.api_key = [key for key in list(api_key or []) if not is_placeholder_value(key)]
        self.model_name = model_name
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.cnt = 0

    def get_api_key(self):
        if self.api_key:
            if self.cnt >= len(self.api_key):
                self.cnt = 0
            return_api_key = self.api_key[self.cnt]
            self.cnt += 1
            return return_api_key
        env_api_key = os.environ.get(self.api_key_env, "")
        if env_api_key:
            return env_api_key
        if self.base_url:
            return "EMPTY"
        return ""

    def has_api_key(self):
        return bool(self.api_key or os.environ.get(self.api_key_env, ""))


class FastApiConfig:
    def __init__(self, port, ip):
        self.port = port
        self.ip = ip
        self.full_ip = "http://" + ip + ":" + port


def _build_model_config(name, raw_config, default_port="6006"):
    config_type = raw_config.get("type", "fastapi")
    if config_type == "fastapi":
        return FastApiConfig(port=str(raw_config.get("port", default_port)), ip=str(raw_config.get("ip", "0.0.0.0")))
    if config_type == "openai":
        return OpenAIConfig(
            api_key=raw_config.get("api_key", raw_config.get("api_keys", [])),
            model_name=raw_config.get("model_name", raw_config.get("full_model_name", name)),
            base_url=raw_config.get("base_url", ""),
            api_key_env=raw_config.get("api_key_env", "OPENAI_API_KEY"),
        )
    raise ValueError(
        f"Unsupported model config type `{config_type}` for `{name}`. "
        "Supported types are `fastapi` and `openai`."
    )


def _build_model_config_map(default_names, section_name):
    config_map = {name: FastApiConfig(port="6006", ip="0.0.0.0") for name in default_names}
    for name, raw_config in get_section(section_name).items():
        if not isinstance(raw_config, dict):
            raise ValueError(f"Config section `{section_name}` entry `{name}` must be a JSON object.")
        config_map[name] = _build_model_config(name, raw_config)
    return config_map


_DEFAULT_MODEL_NAMES = [
    "QwQ-32B",
    "Qwen2.5-32b-Instruct",
    "Qwen2.5-Coder-32b-Instruct",
    "EXAONE-Deep-32B",
    "internlm2_5-20b-chat",
    "TeleChat2-35B-32K",
    "Qwen2.5-72b-Instruct",
    "R1-distill-llama70b",
    "R1-distill-llama32b",
    "Meta-Llama-3.3-70B-Instruct",
    "GLM-Z1-32B-0414",
    "Qwen3-32B",
    "Llama-3_3-Nemotron-Super-49B-v1",
    "HuatuoGPT-o1-72B",
    "gemma_3_27b_it",
]
_DEFAULT_EM_MODEL_NAMES = ["Linq-Embed-Mistral"]


# Model runtime configs.
MODELCONFIG = _build_model_config_map(_DEFAULT_MODEL_NAMES, "model_configs")

# Embedding runtime configs.
EMMODELCONFIG = _build_model_config_map(_DEFAULT_EM_MODEL_NAMES, "embedding_model_configs")


def _get_model_config(config_map, model_name, section_name):
    if model_name not in config_map:
        raise KeyError(
            f"Unknown model `{model_name}`. Add it to `{resolve_config_path()}` section `{section_name}`."
        )
    return config_map[model_name]


def _validate_runtime_config(model_name, model_config, kind):
    if isinstance(model_config, FastApiConfig):
        if is_unconfigured_endpoint(model_config.ip, model_config.port):
            raise ValueError(
                f"{kind} `{model_name}` is not configured. Update `{resolve_config_path()}` "
                f"with a reachable endpoint instead of `{model_config.ip}:{model_config.port}`."
            )
    elif isinstance(model_config, OpenAIConfig):
        configured_key = model_config.has_api_key()
        has_custom_base_url = bool(str(model_config.base_url).strip())
        if has_custom_base_url and is_placeholder_value(model_config.base_url):
            raise ValueError(
                f"{kind} `{model_name}` is not configured. Update `{resolve_config_path()}` "
                "with a valid OpenAI-compatible base URL."
            )
        if not has_custom_base_url and not configured_key:
            raise ValueError(
                f"{kind} `{model_name}` is not configured. Update `{resolve_config_path()}` "
                f"with an API key or set `{model_config.api_key_env}`."
            )


def generate_openai(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    output = output.strip()

    return output


def inject_references_to_messages(
    messages,
    references,
):


    messages = copy.deepcopy(messages)

    system = f"""You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""
#     system = f"""You have been provided with a set of responses from various open-source models to the latest user query. Your task is to ealuate these responses following their stepsz, synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. You should not try to answer the question by your-self. You should mostly rely on the evaluation of the given responses to summary the final response. Remember that The truth may be in the hands of a few. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy.
#     Responses from models:"""

    for i, reference in enumerate(references):

        system += f"\n{i+1}. {reference}"

    if messages[0]["role"] == "system":

        messages[0]["content"] += "\n\n" + system

    else:

        messages = [{"role": "system", "content": system}] + messages

    return messages


def generate_general(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
    streaming=False,
    logprobs=None
):

    '''General model-generation interface for OpenAI-compatible and local FastAPI backends.'''

    output = None

    for sleep_time in [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]:

        try:
            model_config = _get_model_config(MODELCONFIG, model, "model_configs")
            _validate_runtime_config(model, model_config, "Model")
            if isinstance(model_config, OpenAIConfig):
                client_kwargs = {
                    "api_key": model_config.get_api_key(),
                    "timeout": TIMEOUT,
                }
                if model_config.base_url:
                    client_kwargs["base_url"] = model_config.base_url
                client = OpenAI(**client_kwargs)
                request_kwargs = {
                    "model": model_config.model_name,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                }
                if logprobs is not None:
                    request_kwargs["logprobs"] = True
                res = client.chat.completions.create(**request_kwargs)
                response_text = res.choices[0].message.content
                if logprobs is not None and getattr(res.choices[0], "logprobs", None):
                    token_logprobs = [
                        item.logprob
                        for item in (res.choices[0].logprobs.content or [])
                        if getattr(item, "logprob", None) is not None
                    ]
                    if token_logprobs:
                        mean_logprob = sum(token_logprobs) / len(token_logprobs)
                        output = {
                            "response": response_text,
                            "mean_logprob": mean_logprob,
                            "cumulative_logprob": mean_logprob,
                        }
                    else:
                        output = response_text
                else:
                    output = response_text
            elif isinstance(model_config, FastApiConfig):
                headers = {'Content-Type': 'application/json'}
                data = {"messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "logprobs": logprobs}
                try:
                    response = requests.post(url=model_config.full_ip, headers=headers, data=json.dumps(data), timeout=TIMEOUT)
                except requests.exceptions.RequestException as e:
                    raise ValueError(f"Request failed to {model_config.full_ip}: {str(e)}")
                # Check status code
                if response.status_code != 200:
                    raise ValueError(f"Server returned status {response.status_code}. Response: {response.text[:200]}")
                # Check if response body is empty
                if not response.text or len(response.text.strip()) == 0:
                    raise ValueError(f"Empty response from server. Status: {response.status_code}, URL: {model_config.full_ip}")
                try:
                    response_json = response.json()
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON response from server. Status: {response.status_code}, Response text: {response.text[:500]}, Error: {str(e)}")
                if not response_json or len(response_json) == 0:
                    raise ValueError("Empty response JSON from server")
                if logprobs is not None:
                    output = response_json[0]
                else:
                    if 'response' not in response_json[0]:
                        raise ValueError(f"Missing 'response' key in response. Got: {response_json[0]}")
                    output = response_json[0]['response']
            else:
                raise NotImplementedError(f"Model Config {type(model_config)} is not implemented!!!")
            break

        except Exception as e:
            logger.info(f"Detecting error when using model {model}..")
            logger.error(e)
            if DEBUG:
                logger.debug(f"Msgs: `{messages}`")

            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:

        return output
    if isinstance(output, str):
        output = output.strip()
    else:
        output['response'] = output['response'].strip()

    if DEBUG:
        preview = output if isinstance(output, str) else output.get('response', '')
        logger.debug(f"Output: `{preview[:20]}...`.")

    return output


async def async_generate_general(
        model,
        messages,
        max_tokens=2048,
        temperature=0.7,
        streaming=False,
        logprobs=None
):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, generate_general, model, messages, max_tokens, temperature, streaming, logprobs)

def generate_general_em(
    model,
    sentences,
    tasks,
    max_length,
    batch_size
):

    '''General embedding interface for the local FastAPI backend.'''

    output = None

    for sleep_time in [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]:

        try:
            model_config = _get_model_config(EMMODELCONFIG, model, "embedding_model_configs")
            _validate_runtime_config(model, model_config, "Embedding model")
            if isinstance(model_config, FastApiConfig):
                headers = {'Content-Type': 'application/json'}
                data = {"sentences": sentences,
                        'tasks': tasks,
                        'batch_size': batch_size,
                        'max_length': max_length}
                response = requests.post(url=model_config.full_ip, headers=headers, data=json.dumps(data))
                output = response.json()
            elif isinstance(model_config, OpenAIConfig):
                raise NotImplementedError(
                    "Embedding models currently use the local `fastapi` backend only."
                )
            else:
                raise NotImplementedError(f"Model Config {type(model_config)} is not implemented!!!")
            break

        except Exception as e:
            logger.info(f"Detecting error when using model {model}..")
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:

        return output

    if DEBUG:
        logger.debug(f"Output: `{output}...`.")

    return output


def generate_with_references(
    model,
    messages,
    references=[],
    max_tokens=2048,
    temperature=0.7,
    generate_fn=generate_general,
    logprobs=None
):

    if len(references) > 0:
        messages = inject_references_to_messages(messages, references)
    return generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=logprobs
    )

if __name__ == '__main__':
    prompt = "what's the best city. "
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Summary your thinking process in the final response."},
        {"role": "user", "content": prompt}
    ]
    print(generate_general(model='gpt-4.1', messages=messages))
    print(1)
