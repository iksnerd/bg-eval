"""Fluency judging: does the model give the correct sentence a lower per-character loss?

Real Bulgarian (erroneous, correct) pairs, same pairs and same order for every model. Chance is
50%. This is what the flagship is actually good at, and the one benchmark here where a 91M model
trained from scratch outscores a 2.6B Bulgarian-specialized instruct model.

    python judge.py --checkpoint ckpt_gpt2bg.pt --tokenizer tokenizer_gpt2.json --out ours.json
    python judge.py --mcnemar ours.json theirs.json

Two accuracy figures side by side settle nothing, so per-item outcomes are recorded and the
comparison is a paired test. Published reference (`expected/judge.json`), 2000 pairs: **0.9305**
(ours, 91M) against **0.9175** (BgGPT-Gemma-2-2.6B-IT) and **0.846** (BgGPT-Gemma-3-4B-IT).
Against Gemma-2: both right 1782, both wrong 86, only ours right 79, only theirs right 53, exact
McNemar over the 132 discordant pairs **p = 0.0292**. That clears 0.05 and only just — two
comparisons were run, so under a Bonferroni correction for two (0.025) it does not. Against
Gemma-3 it clears anything.

An earlier run over 500 pairs read 0.942 against 0.918 with p = 0.0961 and was called a tie.
The source rows are not shuffled and the first 500 are easier than the corpus, so that tie was
undermeasured rather than real; `expected/judge.json` keeps the reasoning under
`why_2000_and_not_500`.

**N means N real pairs.** Some rows in the source datasets have `erroneous` identical to
`correct` — not pairs, since every model scores them the same and the tie is counted as a miss.
They are skipped and reading continues, rather than being filtered out of a fixed head-slice:
they cluster near the front (65 of the first 500, 517 of all 30,533), so filtering a slice
afterwards would silently cost 13% of the sample. `--keep-degenerate` takes the raw first N.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import torch
import torch.nn.functional as F

from data import error_pairs
from model import load


@torch.no_grad()
def per_char_loss(model, tokenizer, text: str, device: str) -> float:
    """Mean nats per *character*, so sentences of different lengths stay comparable."""
    ids = torch.tensor([tokenizer.encode(text).ids], dtype=torch.long, device=device)
    if ids.shape[1] < 2:
        return float("inf")
    logits = model(ids).float()
    nats = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                           ids[:, 1:].reshape(-1), reduction="sum").item()
    return nats / max(1, len(text))


def mcnemar(ours: list[int], theirs: list[int]) -> dict:
    """Exact two-sided McNemar over the pairs the two models disagree on.

    The discordant pairs are the whole test: agreements carry no information about which model
    is better, which is exactly why comparing two accuracy numbers overstates a difference.
    """
    if len(ours) != len(theirs):
        raise ValueError(f"different pair counts: {len(ours)} vs {len(theirs)}")
    both = sum(a and b for a, b in zip(ours, theirs))
    neither = sum(not a and not b for a, b in zip(ours, theirs))
    only_ours = sum(a and not b for a, b in zip(ours, theirs))
    only_theirs = sum(b and not a for a, b in zip(ours, theirs))
    n = only_ours + only_theirs
    hi = max(only_ours, only_theirs)
    p = 2 * sum(comb(n, k) for k in range(hi, n + 1)) / 2**n if n else 1.0
    return {"both_right": both, "both_wrong": neither, "only_first": only_ours,
            "only_second": only_theirs, "discordant": n, "p_value": round(min(1.0, p), 4)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--pairs", type=int, default=2000,
                   help="usable pairs to score. 2000 is the published protocol, so the default "
                        "reproduces expected/judge.json; 500 reads the superseded 0.942")
    p.add_argument("--keep-degenerate", action="store_true",
                   help="keep the rows where erroneous == correct, which every model scores "
                        "identically and this counts as misses. With --pairs 500 that reproduces "
                        "the superseded 0.822 / 0.802 figures against a raw denominator of 500")
    p.add_argument("--device", default="cpu")
    p.add_argument("--cache", type=Path, default=Path("cache"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--mcnemar", type=Path, nargs=2, metavar=("FIRST", "SECOND"),
                   help="two result files from this script; prints the paired test and exits")
    args = p.parse_args()

    if args.mcnemar:
        first, second = (json.loads(path.read_text()) for path in args.mcnemar)
        table = mcnemar(first["per_item"], second["per_item"])
        print(f"{args.mcnemar[0].stem}: {first['preference_accuracy']:.4f}   "
              f"{args.mcnemar[1].stem}: {second['preference_accuracy']:.4f}")
        print(f"both right {table['both_right']}, both wrong {table['both_wrong']}, "
              f"only {args.mcnemar[0].stem} {table['only_first']}, "
              f"only {args.mcnemar[1].stem} {table['only_second']}")
        print(f"exact McNemar over {table['discordant']} discordant pairs: p = {table['p_value']}")
        return

    if not args.checkpoint or not args.tokenizer:
        raise SystemExit("--checkpoint and --tokenizer are required unless --mcnemar is used")

    pairs = error_pairs(args.cache, args.pairs, drop_degenerate=not args.keep_degenerate)
    model, tokenizer = load(args.checkpoint, args.tokenizer, args.device)
    print(f"{args.checkpoint.name}: {sum(x.numel() for x in model.parameters()) / 1e6:.2f}M params, "
          f"{len(pairs)} pairs\n")

    per_item, ties = [], 0
    for erroneous, correct in pairs:
        good = per_char_loss(model, tokenizer, correct, args.device)
        bad = per_char_loss(model, tokenizer, erroneous, args.device)
        ties += good == bad
        per_item.append(int(good < bad))

    result = {"checkpoint": args.checkpoint.name, "pairs": len(pairs), "ties": ties,
              "preference_accuracy": round(sum(per_item) / len(per_item), 4), "per_item": per_item}
    print(f"prefers the correct sentence in {result['preference_accuracy']:.1%} of pairs "
          f"({sum(per_item)}/{len(per_item)}, {ties} ties)")
    if ties and not args.keep_degenerate:
        print(f"  note: {ties} tie(s) are scored as misses — pairs that differ but score "
              f"identically, which is a real failure to discriminate")

    expected = Path(__file__).parent / "expected" / "judge.json"
    if expected.exists():
        reference = json.loads(expected.read_text())
        protocol = reference["protocol"]
        # The reference was re-measured at 2000 pairs; a run at any other count is a different
        # protocol, so print the published numbers but refuse to subtract across them.
        comparable = len(pairs) == protocol["pairs"]
        print(f"\npublished reference ({reference['measured_on']}), {protocol['pairs']} pairs:")
        for name, value in reference["preference_accuracy"].items():
            got = result["preference_accuracy"] if name in args.checkpoint.stem else None
            if got is None:
                delta = ""
            elif comparable:
                delta = f"  yours {got:.4f}  Δ {got - value:+.4f}"
            else:
                delta = f"  yours {got:.4f} over {len(pairs)} pairs, not comparable"
            print(f"  {name:34s} {value:.4f}{delta}")
        for opponent, table in reference["mcnemar"].items():
            print(f"  exact McNemar {opponent}: {table['discordant']} discordant pairs, "
                  f"p = {table['p_value']}")
        print(f"  tolerance: {reference['tolerance']}")
        if note := reference.get("why_500_real_pairs"):
            print(f"\n  {note['reason']}")
        if not comparable:
            print(f"\n  scored {len(pairs)} pairs against a {protocol['pairs']}-pair reference — "
                  f"rerun with --pairs {protocol['pairs']} to check the published number")

    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
