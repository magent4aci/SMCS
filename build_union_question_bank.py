from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm


DEFAULT_DATASETS = (
    "GPQA",
    "MedMCQA",
    "MATH",
    "MBPP",
    "AIME",
    "IFEval",
    "LiveCodeBench",
    "MMLU-PRO",
)
DEFAULT_MMLU_SUBJECTS = (
    "business",
    "law",
    "psychology",
    "biology",
    "chemistry",
    "history",
    "other",
    "health",
    "economics",
    "math",
    "physics",
    "computer science",
    "philosophy",
    "engineering",
)
DEFAULT_MODEL_LIST = "15_large"
DEFAULT_RESULT_EXP = "question_bank_8k"
DEFAULT_OUTPUT = "union_question_bank_8d.json"
DEFAULT_EMBEDDING_MODEL = "Linq-Embed-Mistral"


@dataclass(frozen=True)
class SourceSpec:
    dataset: str
    subject: str | None = None

    @property
    def label(self) -> str:
        if self.subject is None:
            return self.dataset
        return f"{self.dataset}/{self.subject}"


@dataclass(frozen=True)
class QuestionKey:
    dataset: str
    question_id: str
    subject: str | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def benchmark_result_path(dataset: str, *parts: str) -> Path:
    return repo_root() / "benchmarks" / dataset / "result" / Path(*parts)


def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module `{module_name}` from `{path}`.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model_list(alias: str) -> list[str]:
    common_config = load_module_from_path(
        "smcs_common_config",
        repo_root() / "global_utils" / "common_config.py",
    )
    try:
        return list(common_config.model_List_map[alias])
    except KeyError as exc:
        known = ", ".join(sorted(common_config.model_List_map))
        raise ValueError(f"Unknown model list alias `{alias}`. Known aliases: {known}") from exc


def load_generate_general_em():
    utils = load_module_from_path(
        "smcs_global_utils",
        repo_root() / "global_utils" / "utils.py",
    )
    return utils.generate_general_em


def parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or value.strip().lower() == "all":
        return list(default)
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Comma-separated argument cannot be empty.")
    return items


def result_file_for(spec: SourceSpec, model: str, result_exp: str) -> Path:
    if spec.subject is None:
        return benchmark_result_path(spec.dataset, result_exp, model, "result.json")
    return benchmark_result_path(
        spec.dataset,
        result_exp,
        model,
        f"{spec.subject}_result.json",
    )


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing question-bank result file: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}.")
            records.append(record)
        return records

    if isinstance(parsed, list):
        if not all(isinstance(record, dict) for record in parsed):
            raise ValueError(f"Expected `{path}` to contain a list of JSON objects.")
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    raise ValueError(f"Unsupported result file shape in `{path}`.")


def normalize_question(value: Any) -> str:
    return " ".join(str(value).split())


def parse_is_correct(record: dict[str, Any]) -> bool:
    if "is_correct" in record:
        value = record["is_correct"]
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return bool(value)
    return record.get("pred") == record.get("answer")


def question_key(spec: SourceSpec, record: dict[str, Any]) -> QuestionKey:
    if "question_id" not in record:
        raise KeyError(f"Record from {spec.label} is missing `question_id`.")
    return QuestionKey(
        dataset=spec.dataset,
        question_id=str(record["question_id"]),
        subject=spec.subject,
    )


def add_model_records(
    question_bank: dict[QuestionKey, dict[str, Any]],
    spec: SourceSpec,
    model: str,
    records: list[dict[str, Any]],
) -> None:
    for record in records:
        if "question" not in record:
            raise KeyError(f"Record from {spec.label}/{model} is missing `question`.")
        key = question_key(spec, record)
        question = record["question"]
        if key not in question_bank:
            item = {
                "question": question,
                "src_question_id": record["question_id"],
                "src": spec.dataset,
                "model_res": {},
            }
            if spec.subject is not None:
                item["src_subject"] = spec.subject
            question_bank[key] = item
        else:
            previous = normalize_question(question_bank[key]["question"])
            current = normalize_question(question)
            if previous != current:
                raise ValueError(
                    f"Question mismatch for {spec.label} question_id={record['question_id']} "
                    f"while reading model `{model}`."
                )
        if model in question_bank[key]["model_res"]:
            raise ValueError(f"Duplicate record for {spec.label} question_id={record['question_id']} model={model}.")
        question_bank[key]["model_res"][model] = {"is_correct": parse_is_correct(record)}


def source_specs(datasets: list[str], mmlu_subjects: list[str]) -> list[SourceSpec]:
    specs = []
    for dataset in datasets:
        if dataset == "MMLU-PRO":
            specs.extend(SourceSpec(dataset, subject) for subject in mmlu_subjects)
        else:
            specs.append(SourceSpec(dataset))
    return specs


def validate_model_coverage(
    question_bank: dict[QuestionKey, dict[str, Any]],
    models: list[str],
    allow_partial: bool,
) -> None:
    missing = []
    for key, item in question_bank.items():
        missing_models = [model for model in models if model not in item["model_res"]]
        if missing_models:
            missing.append((key, missing_models))
    if missing and not allow_partial:
        preview = "; ".join(
            f"{key.dataset}/{key.subject or '-'}:{key.question_id} missing {models}"
            for key, models in missing[:5]
        )
        raise ValueError(
            f"{len(missing)} question(s) are missing one or more model results. "
            f"Use --allow_partial to write an incomplete bank. Preview: {preview}"
        )


def load_existing_embeddings(paths: list[Path]) -> dict[str, dict[tuple[str, ...] | str, Any]]:
    cache: dict[str, dict[tuple[str, ...] | str, Any]] = {
        "source_subject": {},
        "source": {},
        "question": {},
    }
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Existing question bank `{path}` must contain a JSON object.")
        for item in data.values():
            if not isinstance(item, dict) or "embedding" not in item:
                continue
            src = str(item.get("src", ""))
            src_question_id = str(item.get("src_question_id", ""))
            src_subject = str(item.get("src_subject", ""))
            question = normalize_question(item.get("question", ""))
            if src and src_question_id:
                cache["source"][(src, src_question_id)] = item["embedding"]
                if src_subject:
                    cache["source_subject"][(src, src_subject, src_question_id)] = item["embedding"]
            if question:
                cache["question"][question] = item["embedding"]
    return cache


def find_reusable_embedding(item: dict[str, Any], cache: dict[str, dict[tuple[str, ...] | str, Any]]):
    src = str(item.get("src", ""))
    src_question_id = str(item.get("src_question_id", ""))
    src_subject = str(item.get("src_subject", ""))
    question = normalize_question(item.get("question", ""))
    if src_subject:
        embedding = cache["source_subject"].get((src, src_subject, src_question_id))
        if embedding is not None:
            return embedding
    embedding = cache["source"].get((src, src_question_id))
    if embedding is not None:
        return embedding
    return cache["question"].get(question)


def make_json_safe(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def attach_embeddings(
    items: list[dict[str, Any]],
    reuse_paths: list[Path],
    embedding_model: str,
    batch_size: int,
    max_length: int,
) -> None:
    cache = load_existing_embeddings(reuse_paths)
    missing_indices = []
    for index, item in enumerate(items):
        embedding = find_reusable_embedding(item, cache)
        if embedding is None:
            missing_indices.append(index)
        else:
            item["embedding"] = make_json_safe(embedding)

    if not missing_indices:
        print("All embeddings were reused from existing question bank files.")
        return

    generate_general_em = load_generate_general_em()
    print(f"Generating {len(missing_indices)} missing embedding(s) with `{embedding_model}`.")
    for start in tqdm(range(0, len(missing_indices), batch_size), desc="Embedding"):
        batch_indices = missing_indices[start:start + batch_size]
        sentences = [items[index]["question"] for index in batch_indices]
        tasks = [""] * len(sentences)
        embeddings = generate_general_em(
            embedding_model,
            sentences=sentences,
            tasks=tasks,
            max_length=max_length,
            batch_size=batch_size,
        )
        if embeddings is None or len(embeddings) != len(batch_indices):
            raise RuntimeError(
                f"Embedding model returned {0 if embeddings is None else len(embeddings)} "
                f"embedding(s) for {len(batch_indices)} sentence(s)."
            )
        for index, embedding in zip(batch_indices, embeddings):
            items[index]["embedding"] = make_json_safe(embedding)


def build_question_bank(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    datasets = parse_csv(args.datasets, DEFAULT_DATASETS)
    mmlu_subjects = parse_csv(args.mmlu_subjects, DEFAULT_MMLU_SUBJECTS)
    models = parse_csv(args.models, tuple()) if args.models else load_model_list(args.model_list)
    specs = source_specs(datasets, mmlu_subjects)
    question_bank: dict[QuestionKey, dict[str, Any]] = {}

    for spec in specs:
        for model in models:
            path = result_file_for(spec, model, args.result_exp)
            records = read_records(path)
            if not records:
                raise ValueError(f"No records found in `{path}`.")
            add_model_records(question_bank, spec, model, records)

    validate_model_coverage(question_bank, models, args.allow_partial)
    items = list(question_bank.values())
    print(f"Loaded {len(items)} unique question(s) from {len(specs)} source split(s).")
    print(f"Validated {len(models)} model(s): {', '.join(models)}")

    if args.dry_run:
        print("Dry run complete; no embeddings generated and no output file written.")
        return {str(index): item for index, item in enumerate(items)}

    reuse_paths = [Path(path).expanduser().resolve() for path in args.reuse_embeddings_from]
    attach_embeddings(
        items,
        reuse_paths=reuse_paths,
        embedding_model=args.embedding_model,
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
    )
    return {str(index): item for index, item in enumerate(items)}


def default_reuse_paths(output_path: Path) -> list[str]:
    candidates = [output_path, repo_root() / "union_question_bank.json"]
    return [str(path) for path in candidates]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge benchmark-local question-bank results into one SMCS question bank.",
    )
    parser.add_argument("--datasets", default="all", help="Comma-separated datasets, or `all`.")
    parser.add_argument("--mmlu_subjects", default="all", help="Comma-separated MMLU-PRO subjects, or `all`.")
    parser.add_argument("--model_list", default=DEFAULT_MODEL_LIST, help="Alias in global_utils/common_config.py.")
    parser.add_argument("--models", default="", help="Comma-separated explicit model names. Overrides --model_list.")
    parser.add_argument("--result_exp", default=DEFAULT_RESULT_EXP, help="Per-benchmark result experiment directory.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Merged question-bank JSON output path.")
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding_batch_size", type=int, default=8)
    parser.add_argument("--embedding_max_length", type=int, default=8192)
    parser.add_argument(
        "--reuse_embeddings_from",
        action="append",
        default=None,
        help="Existing question-bank JSON to reuse embeddings from. Can be passed multiple times.",
    )
    parser.add_argument("--allow_partial", action="store_true", help="Allow questions missing some model results.")
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs without generating embeddings or writing output.")
    args = parser.parse_args()
    output_path = Path(args.output).expanduser().resolve()
    if args.reuse_embeddings_from is None:
        args.reuse_embeddings_from = default_reuse_paths(output_path)
    return args


def main() -> None:
    args = parse_args()
    question_bank = build_question_bank(args)
    if args.dry_run:
        return
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(question_bank, f, ensure_ascii=False)
    print(f"Wrote {len(question_bank)} question(s) to `{output_path}`.")


if __name__ == "__main__":
    main()
