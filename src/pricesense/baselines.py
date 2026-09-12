"""Cheap reference points. If the fine-tune cannot beat these, something is wrong."""

from __future__ import annotations

import numpy as np

from .config import PRICE_PREFIX, QUESTION


def _body(prompt: str) -> str:
    """Strip the scaffolding back down to the raw description."""
    text = prompt.replace(QUESTION, "").replace(PRICE_PREFIX, "")
    return text.strip()


class ConstantPricer:
    """Always guesses the training mean. The floor any model must clear."""

    name = "Constant"

    def __init__(self, value: float):
        self.value = value

    @classmethod
    def fit(cls, pairs: list[dict[str, str]]) -> ConstantPricer:
        prices = [float(p["completion"]) for p in pairs]
        return cls(float(np.mean(prices)))

    def __call__(self, prompt: str) -> float:
        return self.value


class WordCountPricer:
    """Linear fit on description length. Longer blurb, pricier product - roughly."""

    name = "Word Count LR"

    def __init__(self, slope: float, intercept: float):
        self.slope = slope
        self.intercept = intercept

    @classmethod
    def fit(cls, pairs: list[dict[str, str]]) -> WordCountPricer:
        lengths = np.array([len(_body(p["prompt"]).split()) for p in pairs], dtype=float)
        prices = np.array([float(p["completion"]) for p in pairs], dtype=float)
        slope, intercept = np.polyfit(lengths, prices, 1)
        return cls(float(slope), float(intercept))

    def __call__(self, prompt: str) -> float:
        return max(0.0, self.slope * len(_body(prompt).split()) + self.intercept)