"""Stage four: scoring any callable that maps a prompt to a price."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable

from sklearn.metrics import mean_squared_error, r2_score
from tqdm.auto import tqdm

from .config import RESULTS_DIR, ensure_dirs

GREEN, YELLOW, RED, RESET = "\033[92m", "\033[93m", "\033[91m", "\033[0m"
_INK = {"green": GREEN, "orange": YELLOW, "red": RED}

Predictor = Callable[[str], float]


def band(error: float, truth: float) -> str:
    """Absolute dollars matter on cheap items, percentages on expensive ones."""
    if error < 40 or (truth and error / truth < 0.20):
        return "green"
    if error < 80 or (truth and error / truth < 0.40):
        return "orange"
    return "red"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _title_of(prompt: str) -> str:
    """Pull the product title out of a prompt, for readable error reports."""
    for line in prompt.splitlines():
        if line.startswith("Title: "):
            title = line[len("Title: "):].strip()
            return title if len(title) <= 45 else title[:45] + "..."
    return prompt.strip().splitlines()[0][:45]


@dataclass
class Scorecard:
    name: str
    n: int
    mae: float
    rmse: float
    r2: float
    hit_rate: float
    ran_at: str
    worst: list[dict] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)

    def show(self) -> None:
        print(f"\n  {self.name}  (n={self.n})")
        print(f"    mean abs error : ${self.mae:,.2f}")
        print(f"    rmse           : ${self.rmse:,.2f}")
        print(f"    r-squared      : {self.r2:.1f}%")
        print(f"    within 20%/$40 : {self.hit_rate:.1f}%")

        if self.records:
            # MAE is blind to direction. A model that is $80 high half the time
            # and $80 low the other half scores the same as one that is
            # uniformly $80 low - but the two need completely different fixes.
            signed = [r["guess"] - r["truth"] for r in self.records]
            under = sum(1 for s in signed if s < 0)
            print(f"    mean signed err: ${sum(signed) / len(signed):,.2f}")
            print(f"    under-predicted: {100 * under / len(signed):.0f}% of items")

        if self.worst:
            print("\n    worst misses")
            for row in self.worst:
                print(
                    f"      {row['title']:<48} "
                    f"guess ${row['guess']:>9,.2f}  actual ${row['truth']:>9,.2f}"
                )
        print()

    def save(self) -> None:
        ensure_dirs()
        path = RESULTS_DIR / f"{slugify(self.name)}.json"
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        print(f"  saved -> results/{path.name}")


def score(
    predictor: Predictor,
    pairs: list[dict[str, str]],
    name: str,
    n: int = 200,
    verbose: bool = True,
    show_worst: int = 5,
) -> Scorecard:
    sample = pairs[:n]
    guesses, truths, errors, records, hits = [], [], [], [], 0

    for row in tqdm(sample, desc=name):
        truth = float(row["completion"])
        guess = float(predictor(row["prompt"]))
        error = abs(guess - truth)
        colour = band(error, truth)

        guesses.append(guess)
        truths.append(truth)
        errors.append(error)
        records.append(
            {
                "title": _title_of(row["prompt"]),
                "guess": guess,
                "truth": truth,
                "error": error,
            }
        )
        hits += colour == "green"

        if verbose:
            print(f"{_INK[colour]}${error:,.0f}{RESET}", end=" ", flush=True)

    if verbose:
        print()

    worst = sorted(records, key=lambda r: r["error"], reverse=True)[:show_worst]

    card = Scorecard(
        name=name,
        n=len(sample),
        mae=sum(errors) / len(errors),
        rmse=float(mean_squared_error(truths, guesses)) ** 0.5,
        r2=float(r2_score(truths, guesses)) * 100,
        hit_rate=100 * hits / len(sample),
        ran_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        worst=worst,
        records=records,
    )
    card.show()
    card.save()
    return card