from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size: int, dropout: float, max_length: int = 5000) -> None:
        super().__init__()
        den = torch.exp(-torch.arange(0, embedding_size, 2) * math.log(10000) / embedding_size)
        pos = torch.arange(0, max_length).reshape(max_length, 1)
        encoding = torch.zeros(max_length, embedding_size)
        encoding[:, 0::2] = torch.sin(pos * den)
        encoding[:, 1::2] = torch.cos(pos * den)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        sequence_length = token_embedding.size(1)
        return self.dropout(token_embedding + self.encoding[:, :sequence_length, :])


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        embedding_size: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_length: int = 256,
    ) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.source_embedding = nn.Embedding(source_vocab_size, embedding_size)
        self.target_embedding = nn.Embedding(target_vocab_size, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size, dropout, max_length=max_length)
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(embedding_size, target_vocab_size)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
        source_padding_mask: torch.Tensor,
        target_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        source_emb = self.positional_encoding(self.source_embedding(source) * math.sqrt(self.embedding_size))
        target_emb = self.positional_encoding(self.target_embedding(target) * math.sqrt(self.embedding_size))
        output = self.transformer(
            source_emb,
            target_emb,
            tgt_mask=target_mask,
            src_key_padding_mask=source_padding_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=source_padding_mask,
        )
        return self.generator(output)

    def encode(self, source: torch.Tensor, source_padding_mask: torch.Tensor) -> torch.Tensor:
        source_emb = self.positional_encoding(self.source_embedding(source) * math.sqrt(self.embedding_size))
        return self.transformer.encoder(source_emb, src_key_padding_mask=source_padding_mask)

    def decode(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        target_mask: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_emb = self.positional_encoding(self.target_embedding(target) * math.sqrt(self.embedding_size))
        return self.transformer.decoder(
            target_emb,
            memory,
            tgt_mask=target_mask,
            memory_key_padding_mask=memory_padding_mask,
        )


def generate_square_subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)


def create_padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    return tokens == pad_id
