# bg-eval

A loader and evaluation harness for the [Glassbox](https://huggingface.co/glassbox) Bulgarian
models. One file to load the checkpoints, and the scripts that produced the numbers on their
cards, so you can re-run them instead of taking our word for it.

## How the 91M flagship stacks up

| | **ours** (`ckpt_gpt2bg`) | BgGPT-Gemma-2-2.6B-IT | BgGPT-Gemma-3-4B-IT |
|---|---|---|---|
| params | 91.26M | 2.61B | 4.30B |
| **Compression** — bpc, held-out Bulgarian, lower is better | 1.3041 | **1.1600** | 1.1683 |
| **Fluency discrimination** — paired preference, 2000 pairs, higher is better | **0.9305** | 0.9175 (p=0.0292) | 0.846 (p≈0) |

**Shlyokavitsa restoration** — exact match by phrase length, our specialist restorer
(`restore_big`, 4.73M, constrained decoding) against both BgGPT models 5-shot prompted, higher
is better:

| words | 3 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|
| **ours** | **0.965** | **0.920** | **0.820** | **0.790** | **0.720** | **0.370** |
| Gemma-2-2.6B | 0.095 | 0.045 | 0.015 | 0.015 | 0.005 | 0.000 |
| Gemma-3-4B | 0.090 | 0.060 | 0.025 | 0.015 | 0.005 | 0.000 |

**Bold marks the best score per row.** A 28x-larger, continued-pretrained model wins on raw
compression, as expected — but a from-scratch 91M matches or beats it at discriminating fluent
from broken Bulgarian, and a purpose-built 4.73M specialist with constrained decoding beats both
generalist models at restoration by more than an order of magnitude. Same reward, different
mechanics: scale buys compression, task-specific architecture buys the other two.

**What you can reproduce here, and what's cited.** The **ours** row for compression and fluency
is exactly `bpc.py` and `judge.py` below — run them yourself, no trust required. The restoration
row is the constrained decode, and `restore_eval.py` here decodes plain greedy: it reproduces the
unconstrained variant of that row (0.670 rather than 0.720 at 20 words), and
`expected/restore_eval.json` carries both so the difference is a number you can see rather than a
gap you discover. The constraint costs no parameters and no retraining; it is simply not part of
this loader. The
BgGPT columns are cited from `experiments/bulgarian/hf_baselines_FINDINGS.md`, measured on the
same protocol and the same device (CUDA, Modal A10G) but with `transformers`, which this repo
deliberately doesn't depend on — reproducing that side means the private training repo's
`hf_baselines.py`, not this one. The restoration comparison also mixes methods honestly: ours is
a fine-tuned specialist under constrained decoding, theirs is 5-shot prompting a generalist —
the gap is real, but it isn't a controlled ablation of the same recipe.

## Why this exists separately

The checkpoints are plain PyTorch state dicts. They carry their own config, which makes them
self-describing but not self-loading: you still need the class definitions, and the training
repo is private. `model.py` closes that gap in a single file.

It implements exactly one architecture, the one every published checkpoint uses:

```
rmsnorm · rope · swiglu · softmax attention · highway-scalar residuals · weight tying
```

and rejects a checkpoint asking for anything else rather than silently computing the wrong
thing. That is a deliberate trade: the training code needs ~1,150 lines because it carries every
ablation toggle, and none of that matters for running a released model.

**Verified bit-identical to the training implementation** — max absolute logit difference
`0.000e+00` on all nine published checkpoints (the 91M flagship, the corrector, the 14M, both
arms of the 29M optimizer pair, both restorers and both molecule models), argmax agreeing
everywhere. RoPE alone has two conventions that both "work"
and disagree against fixed weights, so "looks right" was not good enough. The check itself
lives in the training repo, because comparing against the private implementation is its whole
job; what it produces is the claim in this paragraph.

## Use

```python
import torch
from model import load

model, tok = load("ckpt_gpt2bg.pt", "tokenizer_gpt2.json")

ids = tok.encode("Столицата на България е").ids
print(tok.decode(model.generate(torch.tensor([ids]), 24, temperature=0.0)[0].tolist()))
# Столицата на България е градът, който е част от световното културно наследство на ЮНЕСКО.
```

Scoring, which is what these models are actually good at:

```python
import torch.nn.functional as F

def nll(text):
    x = torch.tensor([tok.encode(text).ids])
    return F.cross_entropy(model(x)[0, :-1], x[0, 1:]).item()

nll("детето играе в парка")   # 5.064
nll("парка в играе детето")   # 8.721
```

Checkpoints load under `weights_only=True`, so a `.pt` you downloaded cannot execute code on
the way in. Every published checkpoint loads that way; one that does not is refused with an
explanation rather than quietly unpickled.

## Install

```bash
uv sync                  # or: pip install -e .
uv run pytest -q         # 27 tests, no checkpoints and no network needed
```

`torch` alone runs `model.py`. The evaluation scripts add `tokenizers`, `pandas` and `pyarrow`;
all four are declared in `pyproject.toml`, so nothing here depends on guessing.

## Which checkpoints this loads

All of these live under the [Glassbox](https://huggingface.co/glassbox) org on Hugging Face.

| repo | works |
|---|---|
| `gpt-alpha-bg-91m` | yes |
| `gpt-alpha-bg-corrector` | yes |
| `gpt-alpha-bg-29m-muon-pair` | yes, both arms — the Muon model and its AdamW control |
| `gpt-alpha-bg-restorer` | yes, both checkpoints — though the WASM builds embed their weights and need nothing from here |
| `gpt-alpha-bg-14m-onnx` | not needed, it ships as ONNX and runs anywhere |
| `gpt-alpha-molgen-selfies-0.8m` | yes, both checkpoints — the vocabulary is SELFIES symbols and travels in the `.pt` rather than in a tokenizer file |

## Reproducing the numbers on the cards

Three scripts, each against public data only, each printing the published figure next to yours —
the same figures tabulated above.

```bash
uv run bpc.py          --checkpoints ckpt_gpt2bg.pt --tokenizer tokenizer_gpt2.json
uv run judge.py        --checkpoint  ckpt_gpt2bg.pt --tokenizer tokenizer_gpt2.json --out ours.json
uv run restore_eval.py --checkpoint  restore_big.pt
```

`expected/*.json` holds each published figure with its protocol, the device it was measured on
and how much a rerun is allowed to move. `judge.py --mcnemar a.json b.json` runs the paired test
between any two result files, so the p = 0.0292 above is something you check rather than accept.

### What counts as a pair, and how many

Some rows in the source datasets have `erroneous` character-for-character identical to
`correct`. Those are not pairs: there is nothing to discriminate, every model scores them the
same, and the scoring counts the resulting tie as a miss — so including them puts a silent
ceiling on the benchmark.

`judge.py` skips them and keeps reading, so **N means N usable pairs**. It does not filter a
fixed head-slice, because the identical rows cluster near the front: 65 of the first 500 against
517 in all 30,533 deduplicated rows. All models report **0 ties**, which is how you can check no
degenerate pair was scored. `--keep-degenerate` scores the raw rows if you want the difference.

**The default is 2000 pairs, and 500 was not enough.** At 500 the comparison against
Gemma-2-2.6B read p = 0.0961 and looked like a tie. The rows are not shuffled and the first 500
are easier than the corpus — the same model scores 0.942 on 500 and 0.9305 on 2000 — so the tie
was undermeasured rather than real. There are 30,016 usable pairs; use more of them than you
think you need.

## Files

| file | what it is |
|---|---|
| `model.py` | the loader. Needs only torch |
| `data.py` | fetches the public datasets the scripts score against |
| `bpc.py`, `judge.py`, `restore_eval.py` | the three evaluations |
| `expected/*.json` | the published numbers, their protocol, and their provenance |
| `test_bg_eval.py` | the loader's round-trip and refusal behaviour, and the scoring arithmetic. No checkpoints, no network |
| `pyproject.toml`, `uv.lock` | the dependency set, locked, so a rerun is not a resolver roll |

The evaluations need published checkpoints and the network, so CI cannot run them. It runs the
tests, which cover every claim this repo makes about itself rather than about a model.

MIT.
