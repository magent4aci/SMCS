import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
DATA_PATH = BENCHMARK_DIR / "dataset" / "MedMCQA_test.json"
TEST_SIZE = 1200
QUESTION_BANK_SIZE = 1000


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return BENCHMARK_DIR / path


def load_medmcqa_records(path: str | Path = DATA_PATH) -> list[dict]:
    with _resolve_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _with_question_ids(records: list[dict], start_index: int = 0) -> list[dict]:
    result = []
    for offset, record in enumerate(records):
        item = dict(record)
        item["question_id"] = start_index + offset
        result.append(item)
    return result


def _question_key(record: dict) -> str:
    return " ".join(record["question"].split())


def validate_non_overlapping_splits(test_records: list[dict], bank_records: list[dict]) -> None:
    test_questions = {_question_key(record) for record in test_records}
    bank_questions = {_question_key(record) for record in bank_records}
    overlap = test_questions & bank_questions
    if overlap:
        sample = next(iter(overlap))
        raise ValueError(
            "MedMCQA split leakage detected between test split and question-bank split: "
            f"{len(overlap)} overlapping question(s). Example: {sample}"
        )


def load_medmcqa_test_split(path: str | Path = DATA_PATH) -> list[dict]:
    records = load_medmcqa_records(path)
    test_records = records[:TEST_SIZE]
    bank_records = records[-QUESTION_BANK_SIZE:]
    validate_non_overlapping_splits(test_records, bank_records)
    return _with_question_ids(test_records, start_index=0)


def load_medmcqa_question_bank_split(path: str | Path = DATA_PATH) -> list[dict]:
    records = load_medmcqa_records(path)
    test_records = records[:TEST_SIZE]
    bank_records = records[-QUESTION_BANK_SIZE:]
    validate_non_overlapping_splits(test_records, bank_records)
    return _with_question_ids(bank_records, start_index=0)
