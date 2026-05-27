from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import random
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    ParallelExample,
    TranslationDataset,
    collate_batch,
    load_huggingface_dataset,
    load_tsv,
    train_valid_split,
)
from .model import Seq2SeqTransformer, create_padding_mask, generate_square_subsequent_mask
from .tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small Transformer NMT model.")
    parser.add_argument("--tsv", type=Path, default=Path("data/sample_ru_low.tsv"))
    parser.add_argument("--dataset-name", default=None, help="Example: slone/myv_ru_2022")
    parser.add_argument("--split", default="train")
    parser.add_argument("--source-column", default="src")
    parser.add_argument("--target-column", default="tgt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embedding-size", type=int, default=192)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--feedforward-size", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bleu-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ru_myv_transformer.pt"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_examples(args: argparse.Namespace) -> list[ParallelExample]:
    if args.dataset_name:
        return load_huggingface_dataset(
            dataset_name=args.dataset_name,
            source_column=args.source_column,
            target_column=args.target_column,
            split=args.split,
            limit=args.limit,
        )
    return load_tsv(args.tsv, args.source_column, args.target_column, args.limit)


def build_dataloader(
    examples: list[ParallelExample],
    source_tokenizer: CharTokenizer,
    target_tokenizer: CharTokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TranslationDataset(examples, source_tokenizer, target_tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_batch(batch, source_tokenizer.pad_id, target_tokenizer.pad_id),
    )


def train_epoch(
    model: Seq2SeqTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    source_pad_id: int,
    target_pad_id: int,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for source, target in tqdm(dataloader, desc="train", leave=False):
        source = source.to(device)
        target = target.to(device)
        decoder_input = target[:, :-1]
        expected_output = target[:, 1:]

        target_mask = generate_square_subsequent_mask(decoder_input.size(1), device)
        source_padding_mask = create_padding_mask(source, source_pad_id)
        target_padding_mask = create_padding_mask(decoder_input, target_pad_id)

        logits = model(source, decoder_input, target_mask, source_padding_mask, target_padding_mask)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), expected_output.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate_loss(
    model: Seq2SeqTransformer,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    source_pad_id: int,
    target_pad_id: int,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for source, target in dataloader:
        source = source.to(device)
        target = target.to(device)
        decoder_input = target[:, :-1]
        expected_output = target[:, 1:]

        target_mask = generate_square_subsequent_mask(decoder_input.size(1), device)
        source_padding_mask = create_padding_mask(source, source_pad_id)
        target_padding_mask = create_padding_mask(decoder_input, target_pad_id)
        logits = model(source, decoder_input, target_mask, source_padding_mask, target_padding_mask)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), expected_output.reshape(-1))

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def translate(
    model: Seq2SeqTransformer,
    sentence: str,
    source_tokenizer: CharTokenizer,
    target_tokenizer: CharTokenizer,
    max_length: int,
    device: torch.device,
) -> str:
    model.eval()
    source = torch.tensor([source_tokenizer.encode(sentence)[:max_length]], dtype=torch.long, device=device)
    if source[0, -1].item() != source_tokenizer.eos_id:
        source[0, -1] = source_tokenizer.eos_id

    source_padding_mask = create_padding_mask(source, source_tokenizer.pad_id)
    memory = model.encode(source, source_padding_mask)
    generated = torch.tensor([[target_tokenizer.bos_id]], dtype=torch.long, device=device)

    for _ in range(max_length - 1):
        target_mask = generate_square_subsequent_mask(generated.size(1), device)
        decoder_output = model.decode(generated, memory, target_mask, source_padding_mask)
        logits = model.generator(decoder_output[:, -1, :])
        next_token = int(logits.argmax(dim=-1).item())
        generated = torch.cat(
            [generated, torch.tensor([[next_token]], dtype=torch.long, device=device)],
            dim=1,
        )
        if next_token == target_tokenizer.eos_id:
            break

    return target_tokenizer.decode(generated.squeeze(0).tolist())


def corpus_bleu(hypotheses: Iterable[str], references: Iterable[str]) -> float:
    hypotheses = list(hypotheses)
    references = list(references)
    try:
        import sacrebleu

        return float(sacrebleu.corpus_bleu(hypotheses, [references]).score)
    except ImportError:
        matches = 0
        total = 0
        for hypothesis, reference in zip(hypotheses, references):
            hyp_tokens = hypothesis.split()
            ref_tokens = set(reference.split())
            matches += sum(1 for token in hyp_tokens if token in ref_tokens)
            total += len(hyp_tokens)
        return 100.0 * matches / max(total, 1)


@torch.no_grad()
def evaluate_bleu(
    model: Seq2SeqTransformer,
    examples: list[ParallelExample],
    source_tokenizer: CharTokenizer,
    target_tokenizer: CharTokenizer,
    max_length: int,
    device: torch.device,
    sample_count: int,
) -> float:
    sampled = examples[:sample_count]
    hypotheses = [
        translate(model, example.source, source_tokenizer, target_tokenizer, max_length, device)
        for example in sampled
    ]
    references = [example.target for example in sampled]
    return corpus_bleu(hypotheses, references)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    examples = load_examples(args)
    if len(examples) < 2:
        raise ValueError("Need at least two parallel sentence pairs.")

    train_examples, valid_examples = train_valid_split(examples, args.valid_ratio, args.seed)
    source_tokenizer = CharTokenizer.build((example.source for example in train_examples), args.min_freq)
    target_tokenizer = CharTokenizer.build((example.target for example in train_examples), args.min_freq)

    train_loader = build_dataloader(
        train_examples,
        source_tokenizer,
        target_tokenizer,
        args.max_length,
        args.batch_size,
        shuffle=True,
    )
    valid_loader = build_dataloader(
        valid_examples,
        source_tokenizer,
        target_tokenizer,
        args.max_length,
        args.batch_size,
        shuffle=False,
    )

    model = Seq2SeqTransformer(
        source_vocab_size=len(source_tokenizer),
        target_vocab_size=len(target_tokenizer),
        embedding_size=args.embedding_size,
        nhead=args.heads,
        num_encoder_layers=args.encoder_layers,
        num_decoder_layers=args.decoder_layers,
        dim_feedforward=args.feedforward_size,
        dropout=args.dropout,
        max_length=args.max_length,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=target_tokenizer.pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(
        f"examples: train={len(train_examples)}, valid={len(valid_examples)}, "
        f"src_vocab={len(source_tokenizer)}, tgt_vocab={len(target_tokenizer)}, device={device}"
    )

    best_valid_loss = float("inf")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            source_tokenizer.pad_id,
            target_tokenizer.pad_id,
            device,
        )
        valid_loss = evaluate_loss(
            model,
            valid_loader,
            loss_fn,
            source_tokenizer.pad_id,
            target_tokenizer.pad_id,
            device,
        )
        bleu = evaluate_bleu(
            model,
            valid_examples,
            source_tokenizer,
            target_tokenizer,
            args.max_length,
            device,
            args.bleu_samples,
        )

        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} bleu={bleu:.2f}")

        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            checkpoint = {
                "model_state": model.state_dict(),
                "source_tokenizer": source_tokenizer.to_dict(),
                "target_tokenizer": target_tokenizer.to_dict(),
                "model_config": {
                    "source_vocab_size": len(source_tokenizer),
                    "target_vocab_size": len(target_tokenizer),
                    "embedding_size": args.embedding_size,
                    "nhead": args.heads,
                    "num_encoder_layers": args.encoder_layers,
                    "num_decoder_layers": args.decoder_layers,
                    "dim_feedforward": args.feedforward_size,
                    "dropout": args.dropout,
                    "max_length": args.max_length,
                },
                "training_args": vars(args),
                "sample_examples": [asdict(example) for example in valid_examples[:5]],
            }
            torch.save(checkpoint, args.checkpoint)

    print(f"saved best checkpoint to {args.checkpoint}")
    for example in valid_examples[:3]:
        print(f"ru:  {example.source}")
        print(f"ref: {example.target}")
        print(f"hyp: {translate(model, example.source, source_tokenizer, target_tokenizer, args.max_length, device)}")


if __name__ == "__main__":
    main()
