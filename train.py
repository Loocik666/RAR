"""Скрипт обучения мини-GPT с сохранением чекпоинтов и cosine scheduler."""

from __future__ import annotations

import argparse
import math
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


def build_optimizer(model: GPT, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW с decay только на матрицы весов, без decay на bias и нормализации."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Не применяем decay к смещениям, одномерным параметрам (обычно norm/scale),
        # а также к позиционным и токенным эмбеддингам.
        if name.endswith("bias") or param.ndim == 1 or "emb" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)


def get_lr(step: int, warmup_iters: int, lr_decay_iters: int, min_lr: float, max_lr: float) -> float:
    """Linear warmup + cosine decay."""
    if step < warmup_iters:
        return max_lr * float(step + 1) / float(max(1, warmup_iters))

    if step > lr_decay_iters:
        return min_lr

    decay_ratio = (step - warmup_iters) / max(1, (lr_decay_iters - warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение GPT-подобной модели")

    parser.add_argument("--text_path", type=str, required=True, help="Путь к .txt данным")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Папка для чекпоинтов")

    # Данные
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--encoding_name", type=str, default="cl100k_base", choices=["gpt2", "cl100k_base"])

    # Модель
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Обучение
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--lr_decay_iters", type=int, default=5000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
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
        encoding_name=args.encoding_name,
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
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Устройство: {device}")
    print(f"Размер словаря: {data.tokenizer.vocab_size}")

    for step in range(args.max_iters + 1):
        lr = get_lr(
            step=step,
            warmup_iters=args.warmup_iters,
            lr_decay_iters=args.lr_decay_iters,
            min_lr=args.min_lr,
            max_lr=args.learning_rate,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

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
                f"step {step:5d} | lr={lr:.6e} | train_loss={metrics['train']:.4f} | val_loss={metrics['val']:.4f}"
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
