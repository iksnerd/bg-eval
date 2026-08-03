"""Shlyokavitsa restoration: exact match by phrase length, on text the model never saw.

Bulgarian typed on a Latin keyboard, turned back into Cyrillic one character at a time. The
romanization is lossy — `a` spells both **а** and **ъ**, `sht` collapses to **щ** — so this is
not a table lookup that got automated, and the model is genuinely guessing where the keyboard
destroyed the distinction.

    python restore_eval.py --checkpoint restore_big.pt

Published reference (`expected/restore_eval.json`, n = 200 per length): **0.67** exact match at
20 words on the Wikipedia-built test split, against **0.92** on text from the model's own
training distribution. The gap is the finding, not a footnote: Wikipedia is dense with proper
nouns and abbreviations, which is exactly where the irrecoverable collisions live.

**This scores plain greedy decoding, and that is the 0.67 row.** The model cards lead with
**0.72**, which is the same checkpoint decoded under the romanization's own constraint — legal
next characters only, zero extra parameters. That constraint is not implemented here, so this
script checks itself against `exact_match_unconstrained` and every number it prints is one it
measured. `expected/restore_eval.json` carries both rows.

Greedy decoding, so the run is deterministic and a rerun that disagrees means something real
changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data import restoration_pairs
from model import load

# The restorer's whole world: 30 Cyrillic letters out, 26 Latin letters in, a space, the `=`
# that separates question from answer, and a newline that ends it. Inlined here because these
# checkpoints carry no tokenizer file — the vocabulary *is* this list, in this order.
CYRILLIC = "абвгдежзийклмнопрстуфхцчшщъьюя"
LATIN = "abcdefghijklmnopqrstuvwxyz"
END = "\n"
VOCAB = ["<pad>"] + list(CYRILLIC) + list(LATIN) + [" ", "=", END]
STOI = {char: i for i, char in enumerate(VOCAB)}
ALLOWED = set(LATIN) | {" "}


def clean(latin: str) -> str:
    """Strip an input to what the model has ever seen: lowercase Latin letters and spaces."""
    return "".join(char for char in latin.lower() if char in ALLOWED).strip()


@torch.no_grad()
def restore(model, latin: str, device: str) -> str:
    """Greedy-decode one line of shlyokavitsa into Cyrillic."""
    latin = clean(latin)
    if not latin:
        return ""
    idx = torch.tensor([[STOI[c] for c in latin + "="]], dtype=torch.long, device=device)
    out = []
    for _ in range(2 * len(latin) + 16):
        nxt = int(model(idx)[:, -1].argmax(-1))
        if VOCAB[nxt] == END:
            break
        out.append(VOCAB[nxt])
        idx = torch.cat([idx, torch.tensor([[nxt]], device=device)], dim=1)
    return "".join(out)


def char_error_rate(pred: str, gold: str) -> float:
    """Levenshtein distance over characters, normalized by the gold length."""
    if not gold:
        return 0.0
    prev = list(range(len(gold) + 1))
    for i, pc in enumerate(pred, 1):
        cur = [i]
        for j, gc in enumerate(gold, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (pc != gc)))
        prev = cur
    return prev[-1] / len(gold)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path("restore_big.pt"))
    p.add_argument("--per-length", type=int, default=200,
                   help="pairs to score at each phrase length. 200 is the published protocol, so "
                        "the default reproduces expected/restore_eval.json; at 40 the sampling "
                        "noise reaches 11 points and puts 10 words above 5, which is why the "
                        "n=40 table is kept only as provenance")
    p.add_argument("--device", default="cpu")
    p.add_argument("--cache", type=Path, default=Path("cache"))
    p.add_argument("--pairs", type=Path, default=None, help="local JSONL instead of the Hub dataset")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    by_length = restoration_pairs(args.cache, args.per_length, args.pairs)
    model, _ = load(args.checkpoint, None, args.device)
    print(f"{args.checkpoint.name}: {sum(x.numel() for x in model.parameters()) / 1e6:.2f}M params, "
          f"{sum(len(v) for v in by_length.values())} pairs\n")

    results, failures = {}, []
    for n_words in sorted(by_length):
        rows = by_length[n_words]
        exact, cer = 0, 0.0
        for row in rows:
            pred = restore(model, row["latin"], args.device).strip()
            gold = row["cyrillic"].strip()
            if pred == gold:
                exact += 1
            else:
                cer += char_error_rate(pred, gold)
                if len(failures) < 20:
                    failures.append({"n_words": n_words, "latin": row["latin"],
                                     "gold": gold, "pred": pred})
        # CER averages over every item, correct ones contributing zero
        results[n_words] = {"n": len(rows), "exact_match": round(exact / len(rows), 4),
                            "cer": round(cer / len(rows), 4)}
        print(f"  {n_words:2d} words  n={len(rows):3d}  exact {exact / len(rows):.3f}  "
              f"CER {cer / len(rows):.4f}")

    expected = Path(__file__).parent / "expected" / "restore_eval.json"
    if expected.exists():
        reference = json.loads(expected.read_text())
        # Two ways a delta here would compare the wrong things, so neither is printed blind.
        # `restore()` above decodes unconstrained, and `exact_match` is the constrained table the
        # cards lead with: diffing against it prints -0.05 on a run that reproduced perfectly.
        # And the reference is one checkpoint — the 3.16M sibling has a 348-token window and a
        # ~6-word frontier, so its collapse past 10 words is its shape, not a failed reproduction.
        measured = reference["protocol"]["checkpoint"].split()[0]
        same_model = args.checkpoint.name == measured
        print(f"\npublished reference ({reference['measured_on']}), {measured}, "
              f"unconstrained greedy:")
        for n_words, value in reference["exact_match_unconstrained"].items():
            got = results.get(int(n_words), {}).get("exact_match")
            delta = f"  yours {got:.3f}  Δ {got - value:+.3f}" if got is not None and same_model else ""
            constrained = reference["exact_match"][n_words]
            print(f"  {n_words:>3s} words  {value:.3f}{delta}   (constrained {constrained:.3f})")
        print(f"  tolerance: {reference['tolerance']}")
        if not same_model:
            print(f"\n  scored {args.checkpoint.name} against a {measured} reference — the rows "
                  f"above are not yours, and no delta is printed")

    if failures:
        print("\nfirst failures — read these before believing the score:")
        for f in failures[:3]:
            print(f"  [{f['n_words']}w] {f['latin'][:66]}")
            print(f"      gold {f['gold'][:66]}")
            print(f"      pred {f['pred'][:66]}")

    if args.out:
        args.out.write_text(json.dumps({"checkpoint": args.checkpoint.name, "by_length": results,
                                        "failures": failures}, indent=2, ensure_ascii=False) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
