"""Скрипт обучения Looped mini-GPT: checkpoint/resume, multi-exit loss и cosine scheduler."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from data_utils import build_dataset, get_batch
from model import GPT, GPTConfig


def build_optimizer(model: GPT, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW с decay только на матрицы весов, без decay на bias и нормализации."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

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


def compute_losses(
    model: GPT,
    xb: torch.Tensor,
    yb: torch.Tensor,
    entropy_reg_weight: float,
    early_exit_penalty_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Считает лосс обучения с акцентом на рассуждение по всем loop-этапам."""
    outputs = model(xb, yb)
    per_loop_ce = outputs["per_loop_ce"]
    gate_probs = outputs["gate_probs"]

    # 1) Основной objective из замечания: учим быть правой на каждом loop-этапе.
    ce_loss = per_loop_ce.mean()

    gate_stack = torch.stack(gate_probs, dim=0)  # [L, B]

    # 2) KL(P||U), где P — распределение использования loop-шагов.
    usage_dist = torch.softmax(gate_stack.mean(dim=1), dim=0)
    n_loops = usage_dist.numel()
    uniform = torch.full_like(usage_dist, 1.0 / n_loops)
    kl_to_uniform = torch.sum(usage_dist * (torch.log(usage_dist + 1e-8) - torch.log(uniform + 1e-8)))

    # 3) Штраф за «ленивый» ранний выход при высокой ошибке.
    # Чем раньше цикл, тем выше его коэффициент (1.0 -> 0.0).
    early_factor = torch.linspace(1.0, 0.0, steps=n_loops, device=xb.device)
    gate_mean_by_loop = gate_stack.mean(dim=1)
    ce_norm = per_loop_ce.detach() / (per_loop_ce.detach().mean() + 1e-8)
    early_exit_penalty = (gate_mean_by_loop * early_factor * ce_norm).mean()

    total_loss = ce_loss + entropy_reg_weight * kl_to_uniform + early_exit_penalty_weight * early_exit_penalty

    logs = {
        "loss": total_loss.item(),
        "ce_loss": ce_loss.item(),
        "kl_gate": kl_to_uniform.item(),
        "early_exit_penalty": early_exit_penalty.item(),
        "gate_mean": gate_stack.mean().item(),
    }
    return total_loss, logs


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: torch.device,
    entropy_reg_weight: float,
    early_exit_penalty_weight: float,
) -> dict[str, float]:
    """Оценка composite loss на train/val без градиентов."""
    out = {}
    model.eval()

    for split_name, split_data in {"train": train_data, "val": val_data}.items():
        losses = torch.zeros(eval_iters)
        ce_losses = torch.zeros(eval_iters)
        kl_losses = torch.zeros(eval_iters)
        early_penalties = torch.zeros(eval_iters)

        for k in range(eval_iters):
            xb, yb = get_batch(split_data, batch_size, block_size, device)
            loss, logs = compute_losses(model, xb, yb, entropy_reg_weight, early_exit_penalty_weight)
            losses[k] = loss.item()
            ce_losses[k] = logs["ce_loss"]
            kl_losses[k] = logs["kl_gate"]
            early_penalties[k] = logs["early_exit_penalty"]

        out[f"{split_name}_loss"] = losses.mean().item()
        out[f"{split_name}_ce"] = ce_losses.mean().item()
        out[f"{split_name}_kl"] = kl_losses.mean().item()
        out[f"{split_name}_early"] = early_penalties.mean().item()

    model.train()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучение Looped GPT-подобной модели")

    parser.add_argument("--text_path", type=str, required=True, help="Путь к .txt данным")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Папка для чекпоинтов")

    # Данные
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--encoding_name", type=str, default="cl100k_base", choices=["gpt2", "cl100k_base"])

    # Модель
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_layer", type=int, default=2, help="Глубина shared-группы блоков")
    parser.add_argument("--n_loops", type=int, default=4, help="Число loop-итераций latent reasoning")
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
    parser.add_argument("--entropy_reg_weight", type=float, default=0.01)
    parser.add_argument("--early_exit_penalty_weight", type=float, default=0.05)

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

    data = build_dataset(args.text_path, val_ratio=args.val_ratio, encoding_name=args.encoding_name)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = out_dir / "latest.pt"

    resume_step = 0
    checkpoint = None

    if latest_ckpt_path.exists():
        checkpoint = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        saved_config = checkpoint["config"]
        config = GPTConfig(
            vocab_size=data.tokenizer.vocab_size,
            block_size=saved_config["block_size"],
            n_embd=saved_config["n_embd"],
            n_head=saved_config["n_head"],
            n_layer=saved_config["n_layer"],
            n_loops=saved_config.get("n_loops", args.n_loops),
            dropout=saved_config.get("dropout", args.dropout),
        )
        resume_step = int(checkpoint.get("step", 0))
        print(f"Обнаружен чекпоинт, продолжаю обучение с шага {resume_step}")
    else:
        config = GPTConfig(
            vocab_size=data.tokenizer.vocab_size,
            block_size=args.block_size,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            n_loops=args.n_loops,
            dropout=args.dropout,
        )
        print("Чекпоинт не найден, начинаю обучение с нуля")

    model = GPT(config).to(device)
    optimizer = build_optimizer(model, args.learning_rate, args.weight_decay)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Устройство: {device}")
    print(f"Размер словаря: {data.tokenizer.vocab_size}")
    print(f"Loops: {config.n_loops}")

    for step in range(resume_step, args.max_iters + 1):
        lr = get_lr(step, args.warmup_iters, args.lr_decay_iters, args.min_lr, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if step % args.eval_interval == 0:
            metrics = estimate_loss(
                model,
                data.train_ids,
                data.val_ids,
                args.eval_iters,
                args.batch_size,
                config.block_size,
                device,
                args.entropy_reg_weight,
                args.early_exit_penalty_weight,
            )
            print(
                f"step {step:5d} | lr={lr:.6e} | "
                f"train_loss={metrics['train_loss']:.4f} (ce={metrics['train_ce']:.4f}, kl={metrics['train_kl']:.4f}, early={metrics['train_early']:.4f}) | "
                f"val_loss={metrics['val_loss']:.4f} (ce={metrics['val_ce']:.4f}, kl={metrics['val_kl']:.4f}, early={metrics['val_early']:.4f})"
            )

            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": vars(config),
                "tokenizer": data.tokenizer,
                "step": step,
                "args": vars(args),
            }
            torch.save(ckpt, latest_ckpt_path)

        xb, yb = get_batch(data.train_ids, args.batch_size, config.block_size, device)
        loss, train_logs = compute_losses(
            model,
            xb,
            yb,
            args.entropy_reg_weight,
            args.early_exit_penalty_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.eval_interval == 0:
            print(
                f"train_step_metrics | loss={train_logs['loss']:.4f} | ce={train_logs['ce_loss']:.4f} | "
                f"kl={train_logs['kl_gate']:.4f} | early={train_logs['early_exit_penalty']:.4f} | "
                f"gate_mean={train_logs['gate_mean']:.4f}"
            )

    print(f"Обучение завершено. Чекпоинт: {latest_ckpt_path}")


if __name__ == "__main__":
    main()
