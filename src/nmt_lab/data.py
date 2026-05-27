from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import random
from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer, normalize_text


@dataclass(frozen=True)
class ParallelExample:
    source: str
    target: str


class TranslationDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[ParallelExample],
        source_tokenizer: CharTokenizer,
        target_tokenizer: CharTokenizer,
        max_length: int,
    ) -> None:
        self.examples = list(examples)
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        source_ids = self.source_tokenizer.encode(example.source)[: self.max_length]
        target_ids = self.target_tokenizer.encode(example.target)[: self.max_length]

        if source_ids[-1] != self.source_tokenizer.eos_id:
            source_ids[-1] = self.source_tokenizer.eos_id
        if target_ids[-1] != self.target_tokenizer.eos_id:
            target_ids[-1] = self.target_tokenizer.eos_id

        return torch.tensor(source_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


def collate_batch(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor]],
    source_pad_id: int,
    target_pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sources, targets = zip(*batch)
    source_batch = pad_sequence(sources, batch_first=True, padding_value=source_pad_id)
    target_batch = pad_sequence(targets, batch_first=True, padding_value=target_pad_id)
    return source_batch, target_batch


def load_tsv(path: Path, source_column: str, target_column: str, limit: int | None = None) -> list[ParallelExample]:
    examples: list[ParallelExample] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if source_column not in reader.fieldnames or target_column not in reader.fieldnames:
            raise ValueError(
                f"Expected TSV columns '{source_column}' and '{target_column}', "
                f"got {reader.fieldnames!r} in {path}."
            )
        for row in reader:
            source = normalize_text(row[source_column])
            target = normalize_text(row[target_column])
            if source and target:
                examples.append(ParallelExample(source, target))
            if limit is not None and len(examples) >= limit:
                break
    return examples


def load_huggingface_dataset(
    dataset_name: str,
    source_column: str,
    target_column: str,
    split: str,
    limit: int | None = None,
) -> list[ParallelExample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the optional dependency: pip install datasets") from exc

    dataset = load_dataset(dataset_name, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    examples: list[ParallelExample] = []
    for row in dataset:
        source = normalize_text(str(row[source_column]))
        target = normalize_text(str(row[target_column]))
        if source and target:
            examples.append(ParallelExample(source, target))
    return examples


def train_valid_split(
    examples: Sequence[ParallelExample],
    valid_ratio: float,
    seed: int,
) -> tuple[list[ParallelExample], list[ParallelExample]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    valid_size = max(1, int(len(shuffled) * valid_ratio)) if len(shuffled) > 1 else 0
    valid = shuffled[:valid_size]
    train = shuffled[valid_size:]
    if not train:
        raise ValueError("Need at least two parallel examples to create a training split.")
    return train, valid
