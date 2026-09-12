"""Stage one: turn raw products into prompt/completion pairs."""

from __future__ import annotations

import json
from collections import Counter

from tqdm.auto import tqdm

from .config import DATA_DIR, Settings, ensure_dirs, hf_login
from .dataset import Splits, load_items, push_pairs
from .loader import load_tokenizer


def describe_tokens(splits: Splits, tokenizer, cutoff: int) -> None:
    """Sanity-check the cutoff before it silently mangles half the dataset."""
    items = splits.all()
    counts = [
        len(tokenizer.encode(item.summary or "", add_special_tokens=False))
        for item in tqdm(items, desc="measuring descriptions")
    ]
    clipped = sum(1 for c in counts if c > cutoff)
    print(
        f"\n  descriptions : n={len(counts):,}  "
        f"avg={sum(counts) / len(counts):.1f}  max={max(counts):,}"
    )
    print(f"  cutoff={cutoff} clips {clipped:,} items ({clipped / len(counts):.1%})")

    buckets = Counter(min(c // 25 * 25, 200) for c in counts)
    print("\n  token distribution")
    peak = max(buckets.values())
    for edge in sorted(buckets):
        bar = "#" * int(40 * buckets[edge] / peak)
        label = f"{edge}+" if edge == 200 else f"{edge:>3}-{edge + 24:<3}"
        print(f"    {label} | {bar} {buckets[edge]:,}")


def write_jsonl(name: str, items) -> None:
    path = DATA_DIR / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.as_pair()) + "\n")
    print(f"  wrote {path.relative_to(path.parents[1])}  ({len(items):,} rows)")


def read_jsonl(name: str) -> list[dict[str, str]]:
    path = DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `pricesense prepare` first")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare(settings: Settings, push: bool = False) -> Splits:
    ensure_dirs()
    hf_login()

    print(f"[prepare] source: {settings.data.source_repo}")
    splits = load_items(settings.data.source_repo)
    print(f"[prepare] {splits}")

    tokenizer = load_tokenizer(settings.data.base_model)
    cutoff = settings.data.summary_cutoff
    describe_tokens(splits, tokenizer, cutoff)

    print("\n[prepare] building prompts")
    for item in tqdm(splits.train + splits.val, desc="train+val"):
        item.build(tokenizer, cutoff, round_price=True)
    for item in tqdm(splits.test, desc="test"):
        item.build(tokenizer, cutoff, round_price=False)

    sample = splits.test[0]
    print("\n--- sample -------------------------------------------------")
    print(sample.prompt)
    print(f"[{sample.completion}]")
    print("------------------------------------------------------------\n")

    lengths = [item.token_length(tokenizer) for item in splits.all()]
    print(f"[prepare] full sequences: avg={sum(lengths) / len(lengths):.1f} max={max(lengths):,}")
    print(f"[prepare] suggested max_seq_len >= {max(lengths)}")

    write_jsonl("train", splits.train)
    write_jsonl("val", splits.val)
    write_jsonl("test", splits.test)

    if push:
        push_pairs(settings.pairs_repo, splits)

    return splits