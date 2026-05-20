import json
import random
from collections import namedtuple
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm


BENCHMARK_DIR = Path(__file__).resolve().parent
DATASET_DIR = BENCHMARK_DIR / "dataset"
PROMPT_DIR = BENCHMARK_DIR / "prompts"

Example = namedtuple(
    "Example",
    ["question", "choice1", "choice2", "choice3", "choice4", "correct_index"],
)
ExampleWithId = namedtuple(
    "ExampleWithId",
    [
        "question",
        "choice1",
        "choice2",
        "choice3",
        "choice4",
        "correct_index",
        "question_id",
    ],
)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return BENCHMARK_DIR / path


def _question_key(question: str) -> str:
    return " ".join(str(question).split())


def validate_non_overlapping_splits(
    left_examples: Iterable[Example],
    right_examples: Iterable[Example],
    left_name: str,
    right_name: str,
) -> None:
    left_questions = {_question_key(example.question) for example in left_examples}
    right_questions = {_question_key(example.question) for example in right_examples}
    overlap = left_questions & right_questions
    if overlap:
        sample = next(iter(overlap))
        raise ValueError(
            f"GPQA split leakage detected between {left_name} and {right_name}: "
            f"{len(overlap)} overlapping question(s). Example: {sample}"
        )


def load_examples(path: str | Path, seed: int) -> list[Example]:
    question_df = pd.read_csv(_resolve_path(path))
    rng = random.Random(seed)

    def shuffle_choices_and_create_example(row) -> Example:
        choices = [
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
            row["Correct Answer"],
        ]
        rng.shuffle(choices)
        return Example(
            row.Question,
            choices[0],
            choices[1],
            choices[2],
            choices[3],
            choices.index(row["Correct Answer"]),
        )

    return [shuffle_choices_and_create_example(row) for _, row in question_df.iterrows()]


def load_gpqa_test_split(data_dir: str | Path = DATASET_DIR, seed: int = 42) -> list[ExampleWithId]:
    data_dir = _resolve_path(data_dir)
    examples_main = load_examples(data_dir / "gpqa_main.csv", seed=seed)
    examples_diamond = load_examples(data_dir / "gpqa_diamond.csv", seed=seed)
    main_question_ids = {
        _question_key(example.question): question_id
        for question_id, example in enumerate(examples_main)
    }
    missing_questions = [
        example.question
        for example in examples_diamond
        if _question_key(example.question) not in main_question_ids
    ]
    if missing_questions:
        raise ValueError(
            "GPQA diamond test split must be a subset of gpqa_main.csv. "
            f"Missing {len(missing_questions)} question(s)."
        )
    return [
        ExampleWithId(
            **example._asdict(),
            question_id=main_question_ids[_question_key(example.question)],
        )
        for example in examples_diamond
    ]


def load_gpqa_question_bank_split(
    data_dir: str | Path = DATASET_DIR,
    seed: int = 42,
) -> list[Example]:
    data_dir = _resolve_path(data_dir)
    question_bank_examples = load_examples(data_dir / "gpqa_question_bank.csv", seed=seed)
    test_examples = load_gpqa_test_split(data_dir, seed=seed)
    validate_non_overlapping_splits(
        question_bank_examples,
        test_examples,
        "question bank",
        "diamond test",
    )
    return question_bank_examples


def load_from_json(path: str | Path) -> dict:
    with _resolve_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_prompt_from_examples(json_data, with_explanations=True) -> str:
    output = ""
    for q in json_data["questions"]:
        output += f'Question: {q["question"]}\nChoices:\n'
        for choice, value in q["choices"].items():
            output += f"({choice}) {value}\n"
        if with_explanations:
            output += f"Let's think step by step: \n{q['explanation']}\n"
        output += f'The correct answer is ({q["correct_answer"]})\n'
    return output


def chain_of_thought_prompt(example: Example) -> str:
    prompt = (
        "Here are some example questions from experts. An explanation is given before "
        "the final answer. Answer the final question yourself, giving your reasoning "
        "beforehand.\n"
    )
    json_data = load_from_json(PROMPT_DIR / "chain_of_thought_examples.json")
    prompt += generate_prompt_from_examples(json_data, with_explanations=True)
    prompt += f"Question: {example.question}"
    prompt += (
        f"\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}\n"
        f"(C) {example.choice3}\n(D) {example.choice4}"
    )
    prompt += (
        "\nGive step by step reasoning before you answer, and when you're ready to "
        'answer, please use the format "The correct answer is (insert answer here)":\n'
    )
    return prompt


def create_prompts(
    examples: list[Example],
    prompt_type: str,
    few_shot_n: int | None,
    model_name: str,
) -> tuple[list[str], list[Example]]:
    if prompt_type != "chain_of_thought":
        raise ValueError(
            "The SMCS GPQA flow only supports chain_of_thought prompts. "
            f"Got: {prompt_type}"
        )
    if few_shot_n is not None:
        raise ValueError("GPQA chain_of_thought prompts do not use few_shot_n.")
    return [chain_of_thought_prompt(example) for example in tqdm(examples)], examples
