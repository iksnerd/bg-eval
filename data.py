"""The public data these evaluations score against, fetched from the Hub.

Every dataset here is someone's public dataset, so a stranger runs the same evaluation on the
same rows in the same order as the published numbers. That last part is not decoration: the
judge benchmark takes the *first* 500 pairs and bits-per-character takes the *first* ~12k
characters, so a different row order is a different benchmark.

Public data is not clean data, and one flaw here is load-bearing: 65 of those first 500 pairs
have an `erroneous` column identical to `correct`. See `error_pairs`.

Parquet is fetched through the Hub's dataset API rather than through `datasets`, because it
keeps the dependency list at torch + tokenizers + pandas and because it is the same path the
published runs used.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

# Real Bulgarian error/correction pairs, external to every corpus these models trained on.
ERROR_DATASETS = ("thebogko/bulgarian-grammar-mistakes", "thebogko/bulgarian-spelling-mistakes")
RESTORATION_DATASET = "glassbox/shlyokavitsa-pairs"

TIMEOUT = 60  # seconds; a hung socket should fail with a message, not wait forever


def _fetch(url: str, timeout: int = TIMEOUT) -> bytes:
    """GET, turning the failures a first-time user actually hits into readable errors."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"{url} returned HTTP {exc.code}. If this is a 401 or 404 the dataset is private or "
            f"renamed; a public dataset is what these evaluations are supposed to score against."
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"could not reach {url}: {exc}. Check the network, then retry.") from exc


def _parquet(dataset: str, cache: Path, split: str = "train") -> list[Path]:
    """Download one dataset split's parquet shards once, into `cache`. Returns local paths.

    Every dataset here is a single shard today, but the Hub decides that, not us — reading only
    the first shard of a set that grew would silently score a prefix of the benchmark and report
    it as the whole thing.
    """
    cache.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset.replace('/', '__')}.{split}"
    local = sorted(cache.glob(f"{stem}.*.parquet"))
    if not local:
        api = f"https://huggingface.co/api/datasets/{dataset}/parquet/default/{split}"
        urls = json.loads(_fetch(api))
        if not urls:
            raise RuntimeError(f"{dataset} split {split!r} lists no parquet files")
        for i, url in enumerate(urls):
            shard = cache / f"{stem}.{i:03d}.parquet"
            shard.write_bytes(_fetch(url))
            local.append(shard)
        print(f"downloaded {dataset} [{split}] -> {len(local)} shard(s) in {cache}")
    return local


def _error_frame(cache: Path):
    import pandas as pd

    frames = [pd.read_parquet(shard)[["erroneous", "correct"]]
              for ds in ERROR_DATASETS for shard in _parquet(ds, cache)]
    return pd.concat(frames)


def held_out_text(cache: Path, max_chars: int) -> str:
    """The `correct` sentences as one string — real Bulgarian no model here trained on.

    Deduplicated with the first occurrence kept, grammar set before spelling set, exactly as
    the published bits-per-character run had them.
    """
    sentences = _error_frame(cache)["correct"].drop_duplicates().tolist()
    return "\n".join(sentences)[:max_chars]


def error_pairs(cache: Path, limit: int, drop_degenerate: bool = True) -> list[tuple[str, str]]:
    """The first `limit` valid (erroneous, correct) pairs, in dataset order.

    Some rows have `erroneous == correct`, character for character. They are not pairs: there is
    no error to prefer the correction of, every model scores them identically, and a preference
    test that counts the resulting tie as a miss would put a silent ceiling on itself. They are
    skipped, and reading continues down the file until `limit` real pairs are found — so `limit`
    means what it says rather than "however many of these rows happened to be usable".

    That distinction is not academic here. The identical rows cluster near the front: 65 of the
    first 500, against 517 in all 30,533 deduplicated rows. Taking a head-slice and filtering it
    afterwards lands in the densest patch and loses 13% of the sample.

    `drop_degenerate=False` takes the first `limit` rows unfiltered, which is what the harness
    did before this was noticed.
    """
    frame = _error_frame(cache).drop_duplicates().reset_index(drop=True)
    if not drop_degenerate:
        return list(frame.head(limit).itertuples(index=False, name=None))

    chosen = frame[frame.erroneous != frame.correct].head(limit)
    if len(chosen) < limit:
        raise ValueError(f"only {len(chosen)} valid pairs available, {limit} requested")
    scanned = int(chosen.index[-1]) + 1
    print(f"{limit} pairs from the first {scanned} rows ({scanned - limit} identical rows skipped)")
    return list(chosen.itertuples(index=False, name=None))


def restoration_pairs(cache: Path, per_length: int, local: Path | None = None) -> dict[int, list[dict]]:
    """Shlyokavitsa pairs grouped by phrase length: {n_words: [{latin, cyrillic, n_words}, ...]}.

    Reads a local JSONL if given, otherwise the Hub dataset. The first `per_length` rows at each
    length are taken in file order, which is the sampling the published table used.
    """
    if local is not None:
        rows = (json.loads(line) for line in local.open(encoding="utf-8"))
    else:
        import pandas as pd

        shards = _parquet(RESTORATION_DATASET, cache, "test")
        rows = pd.concat([pd.read_parquet(s) for s in shards]).to_dict("records")

    by_length: dict[int, list[dict]] = {}
    for row in rows:
        bucket = by_length.setdefault(int(row["n_words"]), [])
        if len(bucket) < per_length:
            bucket.append(row)
    return by_length
