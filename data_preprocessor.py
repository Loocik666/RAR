"""Сбор и очистка текстовых данных в единый data.txt для обучения Looped Transformer."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


ALLOWED_EXTENSIONS = {
    ".py",
    ".txt",
    ".xml",
    ".json",
    ".md",
    ".yaml",
    ".sql",
    ".c",
    ".cpp",
}


def clean_content(text: str, max_line_length: int) -> str:
    """Очищает текст, сохраняя отступы в начале строк (важно для Python-кода)."""
    # Удаляем непечатные символы, оставляя полезные whitespace.
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t\r")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        # Сохраняем отступы слева, но убираем лишние пробелы/табы справа.
        rstrip_line = line.rstrip()

        # Пропускаем подозрительно длинные строки (минификация/base64 и т.п.).
        if len(rstrip_line) > max_line_length:
            continue

        cleaned_lines.append(rstrip_line)

    # Удаляем серии из 3+ пустых строк.
    normalized = "\n".join(cleaned_lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    return normalized


def read_text_file(path: Path) -> str | None:
    """Пытается прочитать текстовый файл в UTF-8; при ошибке возвращает None."""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def collect_data(
    root_dir: Path,
    output_path: Path,
    min_chars: int,
    max_line_length: int,
    max_total_mb: int,
) -> dict[str, dict[str, int]]:
    """Рекурсивно собирает данные и пишет единый датасет в output_path."""
    max_total_bytes = max_total_mb * 1024 * 1024

    total_written_bytes = 0
    stats: dict[str, dict[str, int]] = {
        ext: {"files": 0, "bytes": 0} for ext in sorted(ALLOWED_EXTENSIONS)
    }

    with output_path.open("w", encoding="utf-8") as out:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                path = Path(dirpath) / filename
                ext = path.suffix.lower()

                if ext not in ALLOWED_EXTENSIONS:
                    continue

                raw_text = read_text_file(path)
                if raw_text is None:
                    continue

                cleaned = clean_content(raw_text, max_line_length=max_line_length)
                if len(cleaned) < min_chars:
                    continue

                block = f"\n[FILE: {path.name}]\n[CONTENT]\n{cleaned}\n<|endoftext|>\n"
                block_bytes = len(block.encode("utf-8"))

                if total_written_bytes + block_bytes > max_total_bytes:
                    return stats

                out.write(block)
                total_written_bytes += block_bytes
                stats[ext]["files"] += 1
                stats[ext]["bytes"] += block_bytes

    return stats


def print_stats(stats: dict[str, dict[str, int]]) -> None:
    """Печать статистики по расширениям."""
    print("\nСтатистика собранных данных по расширениям:")
    total_files = 0
    total_bytes = 0

    for ext, metric in stats.items():
        files = metric["files"]
        size_b = metric["bytes"]
        if files == 0:
            continue
        total_files += files
        total_bytes += size_b
        print(f"  {ext:>5}: files={files:6d}, bytes={size_b:12d}")

    print(f"\nИТОГО: files={total_files}, bytes={total_bytes}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Подготовка data.txt из исходной директории")
    parser.add_argument("--root_dir", type=str, required=True, help="Корневая директория для сканирования")
    parser.add_argument("--output", type=str, default="data.txt", help="Путь к итоговому файлу")
    parser.add_argument("--min_chars", type=int, default=100, help="Минимальная длина контента файла")
    parser.add_argument("--max_line_length", type=int, default=500, help="Максимальная длина строки")
    parser.add_argument("--max_total_mb", type=int, default=200, help="Лимит общего объема данных в МБ")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root_dir = Path(args.root_dir)
    if not root_dir.exists() or not root_dir.is_dir():
        raise ValueError(f"Некорректная директория: {root_dir}")

    output_path = Path(args.output)
    stats = collect_data(
        root_dir=root_dir,
        output_path=output_path,
        min_chars=args.min_chars,
        max_line_length=args.max_line_length,
        max_total_mb=args.max_total_mb,
    )

    print(f"Готово. Данные записаны в: {output_path.resolve()}")
    print_stats(stats)


if __name__ == "__main__":
    main()
