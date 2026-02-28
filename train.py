"""Скрипт обучения мини-GPT с сохранением чекпоинтов."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data_utils import build_dataset, get_batch
from model import GPT, GPTConfig


def estimate_loss(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Оценка loss на train/val без градиентов."""
    out = {}
    model.eval()
    for split_name, split_data in {"train": train_data, "val": val_data}.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split_data, batch_size, block_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split_name] = losses.mean().item()
    model.train()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение GPT-подобной модели")

    parser.add_argument("--text_path", type=str, required=True, help="Путь к .txt данным")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Папка для чекпоинтов")

    # Данные
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--use_tiktoken", action="store_true")
    parser.add_argument("--tiktoken_encoding", type=str, default="gpt2")

    # Модель
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Обучение
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA запрошена, но недоступна.")
        device = torch.device(args.device)

    data = build_dataset(
        args.text_path,
        val_ratio=args.val_ratio,
        use_tiktoken=args.use_tiktoken,
        tiktoken_encoding=args.tiktoken_encoding,
    )

    config = GPTConfig(
        vocab_size=data.tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )

    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Устройство: {device}")
    print(f"Размер словаря: {data.tokenizer.vocab_size}")

    for step in range(args.max_iters + 1):
        if step % args.eval_interval == 0:
            metrics = estimate_loss(
                model,
                data.train_ids,
                data.val_ids,
                args.eval_iters,
                args.batch_size,
                args.block_size,
                device,
            )
            print(
                f"step {step:5d} | train_loss={metrics['train']:.4f} | val_loss={metrics['val']:.4f}"
            )

            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(config),
                "tokenizer": data.tokenizer,
                "step": step,
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / "latest.pt")

        xb, yb = get_batch(data.train_ids, args.batch_size, args.block_size, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    print(f"Обучение завершено. Чекпоинт: {out_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
