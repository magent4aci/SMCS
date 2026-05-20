from pathlib import Path

from datasets import load_dataset


BENCHMARK_DIR = Path(__file__).resolve().parent
DATASET_DIR = BENCHMARK_DIR / "dataset" / "LiveCodeBench_dataset"

dataset = load_dataset("livecodebench/code_generation_lite", 
                          split="test", 
                          version_tag="release_v6", 
                          trust_remote_code=True, 
                          cache_dir=str(BENCHMARK_DIR / "dataset" / "hf_cache"))
dataset.save_to_disk(str(DATASET_DIR))
