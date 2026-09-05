"""Fetch the private Q&A dataset from Hugging Face and save it locally."""

import json
import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

from src.dataset.models import HFSample
from src.settings import DATASET_SPLIT, DEFAULT_MAX_SAMPLES, PRIVATE_REPO
from src.settings import DEFAULT_QUESTIONS_PATH as DEFAULT_OUTPUT_PATH


def load_private_dataset(
    max_samples: int = DEFAULT_MAX_SAMPLES,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> list[HFSample]:
    """Load Q&A records and save up to ``max_samples`` as JSON."""
    load_dotenv()

    token = os.getenv("HF_TOKEN")
    if not token:
        message = "Set HF_TOKEN in your environment or .env file."
        raise ValueError(message)

    dataset = load_dataset(PRIVATE_REPO, token=token, split=DATASET_SPLIT)
    sample_count = min(max_samples, len(dataset))
    samples = [
        HFSample.model_validate(sample)
        for sample in dataset.select(range(sample_count))
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            [sample.model_dump() for sample in samples], ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    return samples


if __name__ == "__main__":
    print("Fetching private Q&A dataset...")
    raw_qa = load_private_dataset()
    print(f"Saved {len(raw_qa)} Q&A pairs to {DEFAULT_OUTPUT_PATH}.")
