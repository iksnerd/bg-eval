"""Bits per character on held-out Bulgarian — the number the model cards lose on.

Per-token cross-entropy cannot be compared across tokenizers, because two models segment the
same text differently. Nats over the same *characters* can be:

    bpc = total_nats / (characters_covered * ln 2)

so this is the one measurement here that puts a 14M character-hungry model, a 91M one and a
2.6B model from someone else on the same axis.

    python bpc.py --checkpoints ckpt_gpt2bg.pt --tokenizer tokenizer_gpt2.json

Published reference (`expected/bpc.json`): **1.3041** for the 91M flagship against **1.1600**
for INSAIT's BgGPT-Gemma-2-2.6B-IT and **1.1683** for BgGPT-Gemma-3-4B-IT. A 28x larger,
continued-pretrained model is 0.144 bpc better; this repo's own harness is where that loss is
recorded, which is the point of shipping it.

Known bias, identical across models at equal block size, so rankings are unaffected: the first
token of each window is never predicted, because no context crosses a window boundary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from data import held_out_text
from model import load


@torch.no_grad()
def total_nats(model, tokenizer, text: str, device: str, batch: int = 8) -> tuple[float, int, int]:
    """Teacher-forced nats over non-overlapping windows -> (nats, tokens predicted, chars covered)."""
    ids = tokenizer.encode(text).ids
    block = model.block_size
    n_windows = len(ids) // block
    if not n_windows:
        raise ValueError(f"text is shorter than one {block}-token window")
    windows = torch.tensor(ids[: n_windows * block], dtype=torch.long).view(n_windows, block)

    nats, predicted = 0.0, 0
    for i in range(0, n_windows, batch):
        x = windows[i : i + batch].to(device)
        logits = model(x).float()
        nats += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                x[:, 1:].reshape(-1), reduction="sum").item()
        predicted += x[:, 1:].numel()
    # Normalize by the characters the scored tokens actually cover, not by the text length:
    # the window clip drops a tail, and counting it would flatter every model equally but wrongly.
    covered = len(tokenizer.decode(ids[: n_windows * block]))
    return nats, predicted, covered


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--tokenizer", type=Path, required=True, help="tokenizer.json shipped with the model")
    p.add_argument("--max-chars", type=int, default=50_000,
                   help="held-out characters to score (50000 reproduces the published numbers)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--cache", type=Path, default=Path("cache"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    text = held_out_text(args.cache, args.max_chars)
    print(f"held-out: {len(text)} characters of thebogko `correct` sentences\n")

    results = {}
    for checkpoint in args.checkpoints:
        model, tokenizer = load(checkpoint, args.tokenizer, args.device)
        nats, predicted, covered = total_nats(model, tokenizer, text, args.device)
        params = sum(x.numel() for x in model.parameters())
        results[checkpoint.stem] = {
            "params": params,
            "bpc": round(nats / (covered * math.log(2)), 4),
            "ce_per_token": round(nats / predicted, 4),
            "tokens_predicted": predicted,
            "chars_covered": covered,
        }
        r = results[checkpoint.stem]
        print(f"{checkpoint.stem:22s} {params / 1e6:6.2f}M  bpc {r['bpc']:.4f}  "
              f"(CE/token {r['ce_per_token']:.4f}, {covered} chars)")

    expected = Path(__file__).parent / "expected" / "bpc.json"
    if expected.exists():
        reference = json.loads(expected.read_text())
        print(f"\npublished reference ({reference['measured_on']}):")
        for name, value in reference["bpc"].items():
            got = results.get(name, {}).get("bpc")
            delta = f"  yours {got:.4f}  Δ {got - value:+.4f}" if got is not None else ""
            print(f"  {name:34s} {value:.4f}{delta}")
        print(f"  tolerance: {reference['tolerance']}")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
