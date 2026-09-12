"""The Item model and everything that turns raw products into training pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from datasets import Dataset, DatasetDict, load_dataset
from pydantic import BaseModel, ConfigDict

from .config import PRICE_PREFIX, QUESTION


class Item(BaseModel):
    """A single product: what it is, and what it actually sold for."""

    model_config = ConfigDict(extra="ignore")

    title: str
    category: str
    price: float
    summary: str | None = None
    prompt: str | None = None
    completion: str | None = None

    def __repr__(self) -> str:
        return f"<{self.title[:40]} = ${self.price:,.2f}>"

    def clip(self, tokenizer, max_tokens: int) -> str:
        """Truncate the description on a token boundary, not a character one."""
        tokens = tokenizer.encode(self.summary or "", add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return self.summary or ""
        return tokenizer.decode(tokens[:max_tokens]).rstrip()

    def build(self, tokenizer, max_tokens: int, *, round_price: bool) -> Self:
        """Attach the prompt and the completion.

        Train and validation completions are rounded to whole dollars so the model
        learns a clean output shape. Test keeps the true price - rounding the
        ground truth would flatter the scores.
        """
        body = self.clip(tokenizer, max_tokens)
        self.prompt = f"{QUESTION}\n\n{body}\n\n{PRICE_PREFIX}"
        self.completion = f"{round(self.price)}.00" if round_price else f"{self.price:.2f}"
        return self

    def token_length(self, tokenizer) -> int:
        text = (self.prompt or "") + (self.completion or "")
        return len(tokenizer.encode(text, add_special_tokens=False))

    def as_pair(self) -> dict[str, str]:
        return {"prompt": self.prompt or "", "completion": self.completion or ""}


@dataclass
class Splits:
    train: list[Item]
    val: list[Item]
    test: list[Item]

    def all(self) -> list[Item]:
        return self.train + self.val + self.test

    def __repr__(self) -> str:
        return (
            f"<Splits train={len(self.train):,} "
            f"val={len(self.val):,} test={len(self.test):,}>"
        )


def _pick(dataset_dict, *names: str):
    for name in names:
        if name in dataset_dict:
            return dataset_dict[name]
    raise KeyError(f"none of {names} found in {list(dataset_dict)}")


def load_items(repo: str) -> Splits:
    """Pull the raw product dataset off the Hub and hydrate it into Items."""
    raw = load_dataset(repo)
    return Splits(
        train=[Item.model_validate(row) for row in _pick(raw, "train")],
        val=[Item.model_validate(row) for row in _pick(raw, "validation", "val")],
        test=[Item.model_validate(row) for row in _pick(raw, "test")],
    )


def push_pairs(repo: str, splits: Splits) -> None:
    """Publish the prompt/completion pairs so the Colab trainer can fetch them."""
    DatasetDict(
        {
            "train": Dataset.from_list([i.as_pair() for i in splits.train]),
            "val": Dataset.from_list([i.as_pair() for i in splits.val]),
            "test": Dataset.from_list([i.as_pair() for i in splits.test]),
        }
    ).push_to_hub(repo)
    print(f"[dataset] pushed pairs to {repo}")