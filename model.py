"""Looped Transformer ядро для Project Root: shared рекуррентный блок + weight tying."""

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
    n_layer: int = 1  # legacy-параметр для совместимости чекпоинтов
    n_loops: int = 4
    dropout: float = 0.1


class RMSNorm(nn.Module):
    """RMSNorm для устойчивого обучения."""

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


class ReasoningBlock(nn.Module):
    """Один shared-блок рассуждения, который используется рекурсивно."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm_mlp = RMSNorm(config.n_embd)
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm_attn(x))
        mlp_out = self.fc2(F.gelu(self.fc1(self.norm_mlp(x))))
        mlp_out = self.dropout(mlp_out)
        return attn_out + mlp_out


class GPT(nn.Module):
    """Looped mini-GPT с рекурсией, weight tying и self-consistency exit head."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # Именование в стиле transformer.wte / transformer.wpe для явного weight tying.
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
            }
        )

        self.reasoning_block = ReasoningBlock(config)
        self.loop_norm = RMSNorm(config.n_embd)
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Self-consistency head: оценивает стабильность h между итерациями.
        # Вход: [mean_abs_delta, mean_sq_delta] -> вероятность выхода.
        self.exit_head = nn.Linear(2, 1)

        # Weight Tying (критично по заданию).
        self.lm_head.weight = self.transformer["wte"].weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _forward_loops(self, idx: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Рекурсивный проход по n_loops с per-loop logits и exit probability."""
        bsz, seq_len = idx.size()
        if seq_len > self.config.block_size:
            raise ValueError("Длина последовательности превышает block_size.")

        positions = torch.arange(0, seq_len, device=idx.device).unsqueeze(0)
        h = self.transformer["drop"](self.transformer["wte"](idx) + self.transformer["wpe"](positions))

        loop_logits: list[torch.Tensor] = []
        gate_probs: list[torch.Tensor] = []

        for _ in range(self.config.n_loops):
            h_prev = h

            # Требуемая формула:
            # h_{i+1} = LayerNorm(h_i + Block(h_i))
            h = self.loop_norm(h + self.reasoning_block(h))

            hidden = self.norm_f(h)
            logits = self.lm_head(hidden)

            delta = h - h_prev
            mean_abs_delta = delta.abs().mean(dim=(1, 2), keepdim=True)
            mean_sq_delta = delta.pow(2).mean(dim=(1, 2), keepdim=True)
            gate_in = torch.cat([mean_abs_delta, mean_sq_delta], dim=-1).squeeze(1)
            gate_prob = torch.sigmoid(self.exit_head(gate_in)).squeeze(-1)

            loop_logits.append(logits)
            gate_probs.append(gate_prob)

        return loop_logits, gate_probs

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        loop_logits, gate_probs = self._forward_loops(idx)

        out: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "logits": loop_logits[-1],
            "loop_logits": loop_logits,
            "gate_probs": gate_probs,
        }

        if targets is not None:
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
