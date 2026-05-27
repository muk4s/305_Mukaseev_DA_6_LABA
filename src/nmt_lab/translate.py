from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import Seq2SeqTransformer
from .tokenizer import CharTokenizer
from .train import translate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate a sentence with a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ru_myv_transformer.pt"))
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-length", type=int, default=None)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)

    source_tokenizer = CharTokenizer.from_dict(checkpoint["source_tokenizer"])
    target_tokenizer = CharTokenizer.from_dict(checkpoint["target_tokenizer"])
    model_config = checkpoint["model_config"]
    model = Seq2SeqTransformer(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])

    max_length = args.max_length or model_config["max_length"]
    print(translate(model, args.text, source_tokenizer, target_tokenizer, max_length, device))


if __name__ == "__main__":
    main()
