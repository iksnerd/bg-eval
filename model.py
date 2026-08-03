"""Standalone loader for the Glassbox Bulgarian checkpoints. No dependency on the training code.

The checkpoints on the Hub are plain PyTorch state dicts that carry their own config, which
makes them self-describing but not self-loading: you still need the class definitions. The
training repo's model package is ~1,150 lines because it carries every architecture toggle the
ablations needed (four norms, five FFN families, fast-weight attention, MoE routing, growth
ops). Every *published* checkpoint uses the same stack:

    rmsnorm · rope · swiglu · softmax attention · weight tying

with one axis of variation — the language models gate their residual branches
(`highway-scalar`), the character restorers do not (`plain`). This file implements those two
and refuses anything else rather than silently loading a model it would compute incorrectly.
Verified against the training implementation: identical logits to 0.0 max absolute difference
on every published checkpoint (see verify.py).

    from model import load
    model, tok = load("ckpt_gpt2bg.pt", "tokenizer_gpt2.json")
    ids = tok.encode("Столицата на България е").ids
    logits = model(torch.tensor([ids]))          # (1, T, vocab)

Checkpoints load under `weights_only=True`, so a downloaded `.pt` cannot execute code on the
way in. See `read_checkpoint`.

Requires: torch, and tokenizers if you want the tokenizer loaded for you.
MIT licensed, like the checkpoints.
"""

from __future__ import annotations

import pathlib
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.serialization import safe_globals

# What this file implements. A checkpoint asking for anything else is rejected: computing the
# wrong thing quietly is worse than not loading.
SUPPORTED = {
    "norm": ("rmsnorm",),
    "pos": ("rope",),
    "ffn": ("swiglu",),
    "attn": ("softmax",),
    # The language models gate their residual branches; the character restorers do not.
    "residual": ("highway-scalar", "plain"),
}

# What a checkpoint means by staying silent. These are the training code's own defaults, so an
# older checkpoint that predates a toggle rebuilds the way it was trained — the restorers omit
# `residual` entirely and are plain, while every language model names it.
DEFAULTS = {"norm": "rmsnorm", "pos": "rope", "ffn": "swiglu", "attn": "softmax", "residual": "plain"}

# A `.pt` is a pickle, and unpickling runs whatever the file says to run. `weights_only=True`
# restricts that to tensors and plain data, which is what these checkpoints are — with one
# exception: the training config stores the tokenizer path as a `pathlib.Path`, so the two
# concrete Path classes are allowlisted by name. Nothing else is, and nothing here needs to be.
SAFE_GLOBALS = [pathlib.PosixPath, pathlib.WindowsPath]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE's pairwise rotation, on *interleaved* pairs (x0,x1), (x2,x3), ...

    Note this is the interleaved convention, not the split-halves one used by Llama's reference
    implementation. Both are valid RoPE; they are not interchangeable against fixed weights.
    """
    return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        if n_embd % n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = n_head
        self.head_size = n_embd // n_head
        if self.head_size % 2:
            raise ValueError("RoPE needs an even head size")
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_size, 2).float() / self.head_size))
        freqs = torch.repeat_interleave(torch.outer(torch.arange(block_size).float(), inv_freq), 2, -1)
        self.register_buffer("rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("rope_sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        shape = (b, t, self.n_head, self.head_size)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        cos, sin = self.rope_cos[:t].to(x.dtype), self.rope_sin[:t].to(x.dtype)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).contiguous().view(b, t, c))


class SwiGLU(nn.Module):
    def __init__(self, n_embd: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or max(1, (8 * n_embd) // 3)  # ~matches a 4x ReLU FFN's parameter count
        self.gate = nn.Linear(n_embd, hidden, bias=False)
        self.up = nn.Linear(n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Pre-norm block, with or without a scalar highway gate on each residual branch.

    The gate is one sigmoid value per token (Linear -> 1), so it costs ~0.02% of parameters.
    The language models use it because the ablation found it beats plain residuals at matched
    parameters, with the margin widening with depth; the restorers predate that result and are
    plain, which is why both paths live here.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, ffn_hidden: int | None = None,
                 residual: str = "highway-scalar"):
        super().__init__()
        self.residual = residual
        self.sa = Attention(n_embd, n_head, block_size)
        self.ffwd = SwiGLU(n_embd, ffn_hidden)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)
        if residual == "highway-scalar":
            self.gate1 = nn.Linear(n_embd, 1)
            self.gate2 = nn.Linear(n_embd, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n1 = self.ln1(x)
        x = x + (self.sa(n1) if self.residual == "plain"
                 else torch.sigmoid(self.gate1(n1)) * self.sa(n1))
        n2 = self.ln2(x)
        return x + (self.ffwd(n2) if self.residual == "plain"
                    else torch.sigmoid(self.gate2(n2)) * self.ffwd(n2))


class GPT(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int,
                 block_size: int, ffn_hidden: int | None = None, tie: bool = True,
                 residual: str = "highway-scalar"):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, ffn_hidden, residual) for _ in range(n_layer)]
        )
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        if tie:
            self.lm_head.weight = self.token_embedding_table.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """(B, T) int64 token ids -> (B, T, vocab) logits. Truncates to the context window."""
        x = self.token_embedding_table(idx[:, -self.block_size :])
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = self(idx)[:, -1] / max(temperature, 1e-6)
            if top_k:
                cutoff = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1:]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1) if temperature > 0 else logits.argmax(-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def read_checkpoint(checkpoint: str | Path, trust_pickle: bool = False) -> dict:
    """Unpickle a checkpoint without letting it execute code.

    Every published checkpoint loads under `weights_only=True` with `SAFE_GLOBALS` allowlisted.
    One that does not contains something these files have never contained, and the honest
    response is to say so rather than to fall back to arbitrary execution: an eval harness
    people are asked to trust should not silently run code out of a downloaded `.pt`.
    """
    if trust_pickle:
        return torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    try:
        with safe_globals(SAFE_GLOBALS):
            return torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    # Narrow on purpose. A bare `except` here reports a bug in this file as a suspicious
    # checkpoint, which is the most misleading thing an integrity check can do.
    except pickle.UnpicklingError as exc:
        raise ValueError(
            f"{checkpoint} did not load under weights_only=True: {exc}\n"
            "Every published checkpoint does. This one holds a pickled object beyond tensors, "
            "plain data and paths — inspect it before trusting it. To load it anyway, and only "
            "if you trust where it came from, pass trust_pickle=True (which permits arbitrary "
            "code execution)."
        ) from exc


def load(checkpoint: str | Path, tokenizer: str | Path | None = None, device: str = "cpu",
         trust_pickle: bool = False):
    """Rebuild a published checkpoint and (optionally) its tokenizer.

    Returns `(model, tokenizer_or_None)`. The model is in eval mode with grads off: these
    checkpoints are for inference and scoring, and the training repo owns the training path.
    """
    blob = read_checkpoint(checkpoint, trust_pickle)
    # Training runs store the config under "args"; the fine-tuning script stores it under
    # "config". Both appear among the published checkpoints, so accept either.
    cfg = blob.get("args") or blob.get("config")
    state = blob["model_state"]
    if cfg is None:
        raise ValueError(f"{checkpoint}: no config found under 'args' or 'config'")

    arch = {}
    for key, allowed in SUPPORTED.items():
        got = cfg.get(key) or DEFAULTS[key]
        if got not in allowed:
            raise ValueError(
                f"This loader implements {key} in {allowed}; the checkpoint asks for {got!r}. "
                "It is a fixed set of configurations on purpose - see the module docstring."
            )
        arch[key] = got

    vocab = state["token_embedding_table.weight"].shape[0]
    model = GPT(
        vocab_size=vocab,
        n_embd=cfg["n_embd"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        block_size=cfg["block_size"],
        ffn_hidden=cfg.get("ffn_hidden"),
        tie=cfg.get("tie", True) in (True, "on"),
        residual=arch["residual"],
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Tied weights make lm_head.weight a duplicate entry; anything else missing is a real
    # mismatch and the caller should hear about it rather than get a half-initialised model.
    missing = [k for k in missing if k != "lm_head.weight"]
    unexpected = [k for k in unexpected if k != "lm_head.weight"]
    if missing or unexpected:
        raise ValueError(f"state dict mismatch: missing={missing}, unexpected={unexpected}")

    model.eval().requires_grad_(False)
    model.to(device)

    tok = None
    if tokenizer is not None:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(tokenizer))
    return model, tok
