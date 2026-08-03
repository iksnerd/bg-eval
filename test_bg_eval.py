"""Tests for the loader and the scoring, runnable by anyone with `pip install torch pytest`.

A repo whose whole claim is "check our numbers instead of trusting us" should not ask to be
trusted about its own scaffolding. Nothing here needs a published checkpoint or the network:
the model tests build a tiny random GPT and round-trip it through a checkpoint on disk, and the
scoring tests run on hand-written inputs whose answers can be worked out by hand.

    pytest -q

What each test is protecting is written above it, because a test whose purpose is unclear gets
deleted the first time it is inconvenient.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from data import error_pairs
from judge import mcnemar
from model import GPT, load, read_checkpoint
from restore_eval import char_error_rate, clean

TINY = dict(vocab_size=37, n_embd=16, n_head=2, n_layer=2, block_size=24)
# The config a checkpoint carries, in the shape train.py writes it.
TINY_CFG = {"n_embd": 16, "n_head": 2, "n_layer": 2, "block_size": 24,
            "norm": "rmsnorm", "pos": "rope", "ffn": "swiglu", "attn": "softmax"}


class NotAllowlisted:
    """Module scope so it can be pickled at all — a local class cannot be, and the test needs
    a checkpoint that really does carry an arbitrary object."""


def write_ckpt(path, model, cfg, **extra):
    torch.save({"model_state": model.state_dict(), "args": cfg, **extra}, path)
    return path


@pytest.fixture
def tiny(tmp_path):
    """A random 2-layer model and the checkpoint it was saved to."""
    torch.manual_seed(0)
    model = GPT(**TINY, residual="highway-scalar").eval()
    return model, write_ckpt(tmp_path / "tiny.pt", model, {**TINY_CFG, "residual": "highway-scalar"})


# --- the loader -------------------------------------------------------------------------


def test_round_trip_is_exact(tiny):
    """Save then load must reproduce the logits bit for bit.

    This is the property `verify.py` checks against the private training code, asserted here
    against the only reference a public clone has: the loader's own output before the trip.
    """
    model, path = tiny
    idx = torch.randint(0, TINY["vocab_size"], (2, 12))
    with torch.no_grad():
        before = model(idx)
    loaded, tok = load(path)
    with torch.no_grad():
        after = loaded(idx)
    assert tok is None
    assert torch.equal(before, after), f"max|Δ| = {(before - after).abs().max().item():.3e}"


def test_loads_in_eval_mode_without_grads(tiny):
    """These checkpoints are for scoring. A model that arrives training-mode scores differently."""
    _, path = tiny
    loaded, _ = load(path)
    assert not loaded.training
    assert all(not p.requires_grad for p in loaded.parameters())


def test_plain_residual_round_trips(tmp_path):
    """The restorers predate the highway gate. Their `plain` blocks must not gain one."""
    torch.manual_seed(1)
    model = GPT(**TINY, residual="plain").eval()
    path = write_ckpt(tmp_path / "plain.pt", model, {**TINY_CFG, "residual": "plain"})
    loaded, _ = load(path)
    assert not hasattr(loaded.blocks[0], "gate1")
    idx = torch.randint(0, TINY["vocab_size"], (1, 8))
    with torch.no_grad():
        assert torch.equal(model(idx), loaded(idx))


def test_residual_defaults_to_plain_when_unstated(tmp_path):
    """A checkpoint written before the toggle existed must rebuild the way it was trained."""
    torch.manual_seed(2)
    model = GPT(**TINY, residual="plain").eval()
    cfg = {k: v for k, v in TINY_CFG.items()}  # no "residual" key at all
    loaded, _ = load(write_ckpt(tmp_path / "old.pt", model, cfg))
    idx = torch.randint(0, TINY["vocab_size"], (1, 8))
    with torch.no_grad():
        assert torch.equal(model(idx), loaded(idx))


def test_tied_embedding_is_not_reported_as_missing(tiny):
    """Weight tying makes lm_head.weight a duplicate key. It must not read as a broken load."""
    _, path = tiny
    loaded, _ = load(path)
    assert loaded.lm_head.weight is loaded.token_embedding_table.weight


@pytest.mark.parametrize("key,value", [
    ("norm", "layernorm"), ("pos", "learned"), ("ffn", "moe"),
    ("attn", "delta"), ("residual", "highway-vector"),
])
def test_unsupported_architecture_is_refused(tmp_path, tiny, key, value):
    """Refusing beats loading something this file would compute incorrectly.

    Every value here is a real toggle in the training code, so these are the checkpoints that
    could plausibly be pointed at this loader, not invented ones.
    """
    model, _ = tiny
    path = write_ckpt(tmp_path / f"{key}.pt", model, {**TINY_CFG, key: value})
    with pytest.raises(ValueError, match="fixed set of configurations"):
        load(path)


def test_state_dict_mismatch_is_refused(tmp_path, tiny):
    """A half-initialised model that silently scores well is the worst outcome available."""
    model, _ = tiny
    state = model.state_dict()
    del state["blocks.0.ffwd.gate.weight"]
    torch.save({"model_state": state, "args": {**TINY_CFG, "residual": "highway-scalar"}},
               tmp_path / "broken.pt")
    with pytest.raises(ValueError, match="state dict mismatch"):
        load(tmp_path / "broken.pt")


def test_missing_config_is_refused(tmp_path, tiny):
    model, _ = tiny
    torch.save({"model_state": model.state_dict()}, tmp_path / "nocfg.pt")
    with pytest.raises(ValueError, match="no config found"):
        load(tmp_path / "nocfg.pt")


def test_config_key_is_accepted_as_well_as_args(tmp_path, tiny):
    """train.py writes "args"; the fine-tuning script writes "config". Both are published."""
    model, _ = tiny
    torch.save({"model_state": model.state_dict(),
                "config": {**TINY_CFG, "residual": "highway-scalar"}}, tmp_path / "sft.pt")
    loaded, _ = load(tmp_path / "sft.pt")
    idx = torch.randint(0, TINY["vocab_size"], (1, 8))
    with torch.no_grad():
        assert torch.equal(model(idx), loaded(idx))


# --- unpickling -------------------------------------------------------------------------


def test_pathlib_config_still_loads(tmp_path, tiny):
    """The real checkpoints store the tokenizer path as a Path, which is why it is allowlisted."""
    model, _ = tiny
    path = write_ckpt(tmp_path / "withpath.pt", model,
                      {**TINY_CFG, "residual": "highway-scalar", "tokenizer": tmp_path / "tok.json"})
    assert load(path)[0] is not None


def test_arbitrary_pickled_object_is_refused(tmp_path, tiny):
    """The point of weights_only: a downloaded .pt must not be able to execute code.

    `NotAllowlisted` is inert — the test is that unpickling an arbitrary class is refused, not
    that this particular one does harm. A real hostile checkpoint would carry `__reduce__`.
    """
    model, _ = tiny
    torch.save({"model_state": model.state_dict(), "args": TINY_CFG,
                "extra": NotAllowlisted()}, tmp_path / "evil.pt")
    with pytest.raises(ValueError, match="weights_only"):
        read_checkpoint(tmp_path / "evil.pt")


def test_the_escape_hatch_is_opt_in(tmp_path, tiny):
    """`trust_pickle=True` must still work — refusing everything is not the goal, refusing
    silently is what is being prevented."""
    model, _ = tiny
    torch.save({"model_state": model.state_dict(), "args": TINY_CFG,
                "extra": NotAllowlisted()}, tmp_path / "evil.pt")
    assert "extra" in read_checkpoint(tmp_path / "evil.pt", trust_pickle=True)


def test_a_bug_in_this_file_is_not_reported_as_a_bad_checkpoint(tmp_path, tiny, monkeypatch):
    """The refusal path must not swallow unrelated errors into "suspicious checkpoint".

    It did, once, during the change that added it: a NameError in this module surfaced as a
    verdict about the file being loaded. That is the most misleading thing an integrity check
    can do, so the narrow `except` is pinned here.
    """
    import model as model_module

    _, path = tiny

    def boom(*_args, **_kwargs):
        raise RuntimeError("unrelated bug")

    monkeypatch.setattr(model_module.torch, "load", boom)
    with pytest.raises(RuntimeError, match="unrelated bug"):
        read_checkpoint(path)


# --- scoring ----------------------------------------------------------------------------


def test_mcnemar_reproduces_every_published_table():
    """Rebuild each published contingency table from its cells and check the p-value lands.

    `expected/judge.json` records one table per opponent, so this walks all of them rather than
    pinning a single hard-coded comparison — adding a third baseline should extend the check for
    free instead of silently leaving it behind.
    """
    reference = json.loads((Path(__file__).parent / "expected" / "judge.json").read_text())
    assert reference["mcnemar"], "expected/judge.json records no contingency tables"
    for opponent, cells in reference["mcnemar"].items():
        ours = ([1] * cells["both_right"] + [0] * cells["both_wrong"]
                + [1] * cells["only_first"] + [0] * cells["only_second"])
        theirs = ([1] * cells["both_right"] + [0] * cells["both_wrong"]
                  + [0] * cells["only_first"] + [1] * cells["only_second"])
        table = mcnemar(ours, theirs)
        for cell in ("both_right", "both_wrong", "only_first", "only_second", "discordant"):
            assert table[cell] == cells[cell], f"{opponent}: {cell} disagrees with expected"
        # p ~ 0 underflows to exactly 0.0 in both the recorded value and the recomputation.
        assert abs(table["p_value"] - cells["p_value"]) < 5e-4, (
            f"{opponent}: recomputing McNemar from the recorded cells gives "
            f"{table['p_value']}, but expected/judge.json records {cells['p_value']}")


def test_mcnemar_ignores_agreements():
    """Concordant pairs carry no information about which model is better; adding them must not
    move the p-value. This is the property that makes the 65 dropped rows safe to drop."""
    ours, theirs = [1, 0, 1, 0], [0, 1, 1, 0]
    lean = mcnemar(ours, theirs)
    padded = mcnemar(ours + [1] * 50 + [0] * 50, theirs + [1] * 50 + [0] * 50)
    assert lean["p_value"] == padded["p_value"]
    assert lean["discordant"] == padded["discordant"] == 2


def test_mcnemar_is_symmetric_and_refuses_ragged_input():
    assert mcnemar([1, 0], [0, 1])["p_value"] == mcnemar([0, 1], [1, 0])["p_value"]
    with pytest.raises(ValueError, match="different pair counts"):
        mcnemar([1, 0], [1])


def test_identical_rows_are_skipped_and_reading_continues(tmp_path, monkeypatch):
    """`erroneous == correct` rows are not pairs, and asking for N must yield N real pairs.

    The distinction that matters: skip-and-continue, not filter-a-head-slice. Asking for 2 pairs
    from a frame whose second row is degenerate must reach past it to the third, rather than
    returning one pair and calling it two.
    """
    import pandas as pd

    import data as data_module

    frame = pd.DataFrame({"erroneous": ["греши", "същото", "друго", "четри"],
                          "correct": ["греша", "същото", "друг", "четири"]})
    monkeypatch.setattr(data_module, "_error_frame", lambda _cache: frame)

    pairs = error_pairs(tmp_path, 2)
    assert len(pairs) == 2
    assert pairs == [("греши", "греша"), ("друго", "друг")]
    assert all(bad != good for bad, good in pairs)

    # Unfiltered, the same request returns the degenerate row and only one real pair.
    assert error_pairs(tmp_path, 2, drop_degenerate=False) == [("греши", "греша"),
                                                              ("същото", "същото")]


def test_asking_for_more_pairs_than_exist_is_refused(tmp_path, monkeypatch):
    """Silently returning fewer pairs than requested is how a denominator goes wrong unnoticed."""
    import pandas as pd

    import data as data_module

    frame = pd.DataFrame({"erroneous": ["греши", "същото"], "correct": ["греша", "същото"]})
    monkeypatch.setattr(data_module, "_error_frame", lambda _cache: frame)
    with pytest.raises(ValueError, match="only 1 valid pairs available"):
        error_pairs(tmp_path, 2)


# --- restoration ------------------------------------------------------------------------


def test_char_error_rate_is_edit_distance_over_gold_length():
    assert char_error_rate("абв", "абв") == 0.0
    assert char_error_rate("абг", "абв") == pytest.approx(1 / 3)
    assert char_error_rate("", "абв") == pytest.approx(1.0)
    assert char_error_rate("абв", "") == 0.0


def test_clean_strips_to_what_the_model_has_seen():
    """The restorer's vocabulary is 26 Latin letters and a space. Anything else is not input."""
    assert clean("Az Shte IDA!") == "az shte ida"
    assert clean("  тест 123  ") == ""


def test_expected_files_are_readable_and_carry_their_provenance():
    """`expected/*.json` is the repo's contract. A figure without its protocol is a rumour."""
    from pathlib import Path

    for name in ("bpc", "judge", "restore_eval"):
        blob = json.loads((Path(__file__).parent / "expected" / f"{name}.json").read_text())
        for field in ("benchmark", "protocol", "measured_on", "tolerance", "source"):
            assert blob.get(field), f"{name}.json is missing {field}"


def _expected(name: str) -> dict:
    return json.loads((Path(__file__).parent / "expected" / f"{name}.json").read_text())


def _argparse_default(module: str, flag: str):
    """Each parser is built inside its `main()`, so read the default out of the source rather
    than importing and running it."""
    import ast

    tree = ast.parse((Path(__file__).parent / f"{module}.py").read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and getattr(node.args[0], "value", None) == flag):
            for keyword in node.keywords:
                if keyword.arg == "default":
                    return ast.literal_eval(keyword.value)
    raise AssertionError(f"{module}.py declares no default for {flag}")


def test_each_script_reads_keys_its_expected_file_actually_has():
    """The reference print-out is the one path these tests cannot reach — getting there needs a
    checkpoint, and needing none is the point of this suite. So it drifted unobserved: judge.json
    was re-measured to one contingency table per opponent while judge.py went on indexing a flat
    one, which is a KeyError on every real run. Assert the reads instead of the output.
    """
    judge = _expected("judge")
    assert isinstance(judge["protocol"]["pairs"], int)
    assert judge["preference_accuracy"], "no accuracies to compare against"
    for opponent, table in judge["mcnemar"].items():
        assert {"discordant", "p_value"} <= table.keys(), f"{opponent} is missing a cell"
    assert judge["why_500_real_pairs"]["reason"]

    restore = _expected("restore_eval")
    # restore_eval.py decodes unconstrained, so that is the row it may diff against.
    assert restore["exact_match_unconstrained"].keys() == restore["exact_match"].keys()

    assert _expected("bpc")["bpc"], "no bpc figures to compare against"


def test_defaults_reproduce_the_published_protocol():
    """A default that undershoots its reference means the documented command never reproduces the
    published number, and the delta it prints reads as a failed reproduction rather than as a
    different protocol. Both drifted this way when the reference was re-measured upward.
    """
    assert _argparse_default("judge", "--pairs") == _expected("judge")["protocol"]["pairs"]
    assert (_argparse_default("restore_eval", "--per-length")
            == _expected("restore_eval")["protocol"]["per_length"])
    assert _argparse_default("bpc", "--max-chars") == _expected("bpc")["protocol"]["max_chars"]
