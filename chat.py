"""CLI-скрипт для генерации текста из обученного чекпоинта с looped reasoning."""

from __future__ import annotations

import argparse

import torch

from model import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Инференс Looped mini-GPT")
    parser.add_argument("--checkpoint", type=str, required=True, help="Путь к latest.pt")
    parser.add_argument("--prompt", type=str, default="Привет")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--gate_threshold", type=float, default=0.9)
    parser.add_argument("--max_loops", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA запрошена, но недоступна.")
        device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    config = GPTConfig(**checkpoint["config"])
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = checkpoint["tokenizer"]
    prompt_ids = tokenizer.encode(args.prompt)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            gate_threshold=args.gate_threshold,
            max_loops=args.max_loops,
        )

    generated = tokenizer.decode(out[0].tolist())
    print(generated)


if __name__ == "__main__":
    main()
