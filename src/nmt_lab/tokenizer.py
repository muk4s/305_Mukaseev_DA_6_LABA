from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable
import re


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


def normalize_text(text: str) -> str:
    """Keep the script intact while making whitespace predictable."""
    text = text.strip().lower().replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text)


@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1) -> "CharTokenizer":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(normalize_text(text))

        tokens = SPECIAL_TOKENS + [
            token
            for token, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
            if count >= min_freq and token not in SPECIAL_TOKENS
        ]
        return cls({token: index for index, token in enumerate(tokens)}, tokens)

    @classmethod
    def from_dict(cls, payload: dict) -> "CharTokenizer":
        itos = list(payload["itos"])
        return cls({token: index for index, token in enumerate(itos)}, itos)

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [self.stoi.get(char, self.unk_id) for char in normalize_text(text)]
        if add_special_tokens:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        chars: list[str] = []
        for token_id in ids:
            if token_id < 0 or token_id >= len(self.itos):
                continue
            token = self.itos[token_id]
            if token in SPECIAL_TOKENS:
                continue
            chars.append(token)
        return "".join(chars).strip()
