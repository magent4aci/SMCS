import json
import os
from functools import lru_cache
from pathlib import Path


CONFIG_ENV_VAR = "SMCS_CONFIG_PATH"
DEFAULT_CONFIG_FILENAME = "smcs_config.json"
PLACEHOLDER_SNIPPETS = (
    "your path",
    "your api",
    "api_key",
    "the base url",
    "the name of the model",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def benchmarks_root() -> Path:
    return repo_root() / "benchmarks"


def benchmark_dir(dataset: str) -> Path:
    return benchmarks_root() / dataset


def resolve_benchmark_relative_path(dataset: str, path_value: str | None) -> str | None:
    if path_value is None:
        return None
    raw_path = str(path_value).strip()
    if not raw_path:
        return raw_path
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = benchmark_dir(dataset) / path
    return str(path.resolve())


def is_cwd_benchmark_dir(dataset: str) -> bool:
    return Path.cwd().resolve() == benchmark_dir(dataset).resolve()


def benchmark_result_path(dataset: str, *parts: str) -> str:
    if is_cwd_benchmark_dir(dataset):
        return str(Path("result", *parts))
    return str(benchmark_dir(dataset) / "result" / Path(*parts))


def default_config_path() -> Path:
    return repo_root() / DEFAULT_CONFIG_FILENAME


def resolve_config_path() -> Path:
    config_path = os.environ.get(CONFIG_ENV_VAR)
    if config_path:
        return Path(config_path).expanduser().resolve()
    return default_config_path().resolve()


@lru_cache(maxsize=1)
def load_runtime_config() -> dict:
    config_path = resolve_config_path()
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"SMCS config at `{config_path}` must contain a JSON object at the top level."
        )
    return data


def get_section(section_name: str) -> dict:
    section = load_runtime_config().get(section_name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"SMCS config section `{section_name}` must be a JSON object.")
    return section


def resolve_repo_relative_path(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    raw_path = str(path_value).strip()
    if not raw_path:
        return raw_path
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    return str(path.resolve())


def is_placeholder_value(value) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    if not normalized:
        return True
    return any(snippet in normalized for snippet in PLACEHOLDER_SNIPPETS)


def is_unconfigured_endpoint(ip, port=None) -> bool:
    if is_placeholder_value(ip):
        return True
    ip_normalized = str(ip).strip()
    if ip_normalized in {"0.0.0.0", "http://0.0.0.0", "https://0.0.0.0"}:
        return True
    if port is None:
        return False
    return not str(port).strip()


def ensure_configured_path(path_value: str | None, name: str, kind: str) -> str:
    resolved_path = resolve_repo_relative_path(path_value)
    if is_placeholder_value(resolved_path):
        config_path = resolve_config_path()
        raise ValueError(
            f"{kind} `{name}` is not configured. Set it in `{config_path}` "
            f"or via `{CONFIG_ENV_VAR}`."
        )
    return resolved_path
