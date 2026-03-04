"""Утилиты токенизации и загрузки данных из txt/xml/json/md/yaml/кода."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch


SUPPORTED_EXTENSIONS = {
    ".py",
    ".txt",
    ".xml",
    ".json",
    ".md",
    ".yaml",
    ".yml",
    ".sql",
    ".c",
    ".cpp",
}


class BPETokenizer:
    """BPE-токенайзер на базе tiktoken (подходит для русского языка и кода)."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError("Установите tiktoken: pip install tiktoken") from exc

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


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def _is_preformatted(text: str) -> bool:
    return "[FILE:" in text and "[CONTENT]" in text and "<|endoftext|>" in text


def _format_block(path: Path, text: str) -> str:
    return f"\n[FILE: {path.name}]\n[CONTENT]\n{text}\n<|endoftext|>\n"


def collect_corpus(data_path: str, recursive: bool = True) -> str:
    """Собирает корпус из файла/папки в единый текст.

    - Если передан файл: читается как есть (с разметкой или без).
    - Если папка: собираются только SUPPORTED_EXTENSIONS, каждый файл оборачивается
      в мета-блок [FILE]/[CONTENT]/<|endoftext|>, если у него нет готовой разметки.
    """
    path = Path(data_path)
    if not path.exists():
        raise ValueError(f"Путь не существует: {path}")

    if path.is_file():
        text = _read_text(path)
        if text is None:
            raise ValueError(f"Не удалось прочитать файл как UTF-8: {path}")
        return text

    parts: list[str] = []

    if recursive:
        walker = os.walk(path)
        candidates = [Path(root) / name for root, _, files in walker for name in files]
    else:
        candidates = [p for p in path.iterdir() if p.is_file()]

    candidates.sort()

    for file_path in candidates:
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        text = _read_text(file_path)
        if text is None:
            continue

        if _is_preformatted(text):
            parts.append(text.strip() + "\n")
        else:
            parts.append(_format_block(file_path, text.strip()))

    merged = "\n".join(parts).strip()
    if not merged:
        raise ValueError("После фильтрации не найдено пригодных текстовых данных.")
    return merged


def build_dataset(
    data_path: str,
    val_ratio: float = 0.1,
    encoding_name: str = "cl100k_base",
    recursive: bool = True,
) -> DatasetBundle:
    """Загрузка корпуса из файла/папки, BPE-токенизация, разбиение на train/val."""
    text = collect_corpus(data_path, recursive=recursive)
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
