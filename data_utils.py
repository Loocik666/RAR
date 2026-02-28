"""Утилиты для токенизации и подготовки батчей (BPE через tiktoken)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch


class BPETokenizer:
    """BPE-токенайзер на базе tiktoken (подходит для русского языка и кода)."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "Установите tiktoken: pip install tiktoken"
            ) from exc

        if encoding_name not in {"gpt2", "cl100k_base"}:
            raise ValueError("Допустимые encoding_name: 'gpt2' или 'cl100k_base'.")

        self.encoding_name = encoding_name
        self.enc = tiktoken.get_encoding(encoding_name)

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)


@dataclass
class DatasetBundle:
    train_ids: torch.Tensor
    val_ids: torch.Tensor
    tokenizer: BPETokenizer


def load_text(path: str | Path) -> str:
    """Чтение текстового корпуса из файла UTF-8."""
    return Path(path).read_text(encoding="utf-8")


def build_dataset(
    text_path: str,
    val_ratio: float = 0.1,
    encoding_name: str = "cl100k_base",
) -> DatasetBundle:
    """Загрузка текста, BPE-токенизация, разбиение на train/val."""
    text = load_text(text_path)
    tokenizer = BPETokenizer(encoding_name=encoding_name)

    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_idx = int((1.0 - val_ratio) * len(ids))
    train_ids = ids[:split_idx]
    val_ids = ids[split_idx:]

    return DatasetBundle(train_ids=train_ids, val_ids=val_ids, tokenizer=tokenizer)


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Сэмплирует случайный батч (x, y), где y — x, сдвинутый на 1 токен."""
    if len(data) <= block_size:
        raise ValueError("Длина данных должна быть больше block_size.")

    starts = [random.randint(0, len(data) - block_size - 1) for _ in range(batch_size)]
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])

    return x.to(device), y.to(device)
