"""Утилиты для токенизации и подготовки батчей."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch


class CharTokenizer:
    """Простой посимвольный токенайзер (учебный вариант)."""

    def __init__(self, text: str) -> None:
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class TikTokenTokenizer:
    """Токенайзер через tiktoken (опционально)."""

    def __init__(self, encoding_name: str = "gpt2") -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "Для TikTokenTokenizer установите пакет: pip install tiktoken"
            ) from exc

        self.enc = tiktoken.get_encoding(encoding_name)
        self.stoi = None
        self.itos = None

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
    tokenizer: CharTokenizer | TikTokenTokenizer


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_dataset(
    text_path: str,
    val_ratio: float = 0.1,
    use_tiktoken: bool = False,
    tiktoken_encoding: str = "gpt2",
) -> DatasetBundle:
    """Загружает текст, токенизирует и делит на train/val."""
    text = load_text(text_path)

    if use_tiktoken:
        tokenizer: CharTokenizer | TikTokenTokenizer = TikTokenTokenizer(tiktoken_encoding)
    else:
        tokenizer = CharTokenizer(text)

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
