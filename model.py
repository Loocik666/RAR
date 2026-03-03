"""Looped decoder-only Transformer (GPT-подобная) с latent reasoning на PyTorch."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Гиперпараметры модели."""

    vocab_size: int
    block_size: int = 256
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 2  # глубина shared-группы блоков в одном loop-шаге
    n_loops: int = 4  # число рекурсивных прогонов shared-блоков
    dropout: float = 0.1


class RMSNorm(nn.Module):
    """RMSNorm для более стабильного обучения."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class CausalSelfAttention(nn.Module):
    """Многоголовое каузальное внимание через scaled_dot_product_attention."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd должен делиться на n_head без остатка.")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, emb_dim = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, emb_dim)
        y = self.resid_dropout(self.out_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward блок."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class Block(nn.Module):
    """Pre-Norm блок: RMSNorm -> Attention -> RMSNorm -> MLP."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    """Looped mini-GPT с shared-блоками и exit gate."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Shared-группа блоков, которая многократно применяется в loop-режиме.
        self.loop_blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Шлюз выхода: оценивает готовность завершить рассуждение на текущем loop.
        self.exit_gate = nn.Linear(config.n_embd, 1)

        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _forward_loops(self, idx: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Возвращает logits и gate_prob для каждого loop-шага."""
        bsz, seq_len = idx.size()
        if seq_len > self.config.block_size:
            raise ValueError("Длина последовательности превышает block_size.")

        positions = torch.arange(0, seq_len, device=idx.device).unsqueeze(0)
        latent = self.drop(self.tok_emb(idx) + self.pos_emb(positions))

        loop_logits: list[torch.Tensor] = []
        gate_probs: list[torch.Tensor] = []

        for _ in range(self.config.n_loops):
            for block in self.loop_blocks:
                latent = block(latent)

            hidden = self.norm_f(latent)
            logits = self.lm_head(hidden)

            # Gate считаем по состоянию последнего токена в последовательности.
            gate_logit = self.exit_gate(hidden[:, -1, :])
            gate_prob = torch.sigmoid(gate_logit).squeeze(-1)

            loop_logits.append(logits)
            gate_probs.append(gate_prob)

        return loop_logits, gate_probs

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Возвращает результаты по всем loop-итерациям."""
        loop_logits, gate_probs = self._forward_loops(idx)

        out: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "logits": loop_logits[-1],
            "loop_logits": loop_logits,
            "gate_probs": gate_probs,
        }

        if targets is not None:
            # Базовый per-loop CE без весов (финальная композиция в train.py).
            per_loop_ce = [
                F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                for logits in loop_logits
            ]
            out["per_loop_ce"] = torch.stack(per_loop_ce)

        return out

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        gate_threshold: float = 0.9,
        max_loops: int | None = None,
    ) -> torch.Tensor:
        """Генерация с размышлением: ранний выход по gate или по max_loops."""
        loops_cap = self.config.n_loops if max_loops is None else max(1, min(max_loops, self.config.n_loops))

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            loop_logits, gate_probs = self._forward_loops(idx_cond)

            chosen_idx = loops_cap - 1
            for loop_i in range(loops_cap):
                if torch.all(gate_probs[loop_i] >= gate_threshold):
                    chosen_idx = loop_i
                    break

            logits = loop_logits[chosen_idx][:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx
