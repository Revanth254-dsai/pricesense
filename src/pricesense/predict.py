"""Stage three: turning a prompt into a number."""

from __future__ import annotations

import re

import torch

from .loader import load_model, load_tokenizer

_NUMBER = re.compile(r"[-+]?\d*\.\d+|\d+")


def parse_price(text: str) -> float:
    """Pull the first number out of whatever the model produced. 0 if none."""
    cleaned = text.replace("$", "").replace(",", "")
    match = _NUMBER.search(cleaned)
    return float(match.group()) if match else 0.0


class PriceSense:
    """A callable price estimator: prompt in, dollars out."""

    def __init__(
        self,
        base_model: str,
        adapter_repo: str | None = None,
        *,
        weighted: bool = False,
        top_k: int = 3,
        max_new_tokens: int = 6,
    ):
        self.tokenizer = load_tokenizer(base_model, padding_side="left")
        self.model = load_model(base_model, adapter_repo)
        self.weighted = weighted
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.name = adapter_repo or f"{base_model.split('/')[-1]} (base)"

    def __call__(self, prompt: str) -> float:
        if self.weighted:
            estimate = self._weighted_estimate(prompt)
            if estimate is not None:
                return estimate
        return self._greedy_estimate(prompt)

    @torch.no_grad()
    def _greedy_estimate(self, prompt: str) -> float:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        tail = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return parse_price(tail)

    @torch.no_grad()
    def _weighted_estimate(self, prompt: str) -> float | None:
        """Blend the top-k candidate first tokens by probability.

        Greedy decoding throws away the model's uncertainty - if it is torn
        between 90 and 100 it commits to one. This reads the distribution
        instead. Returns None when no candidate token is numeric.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        logits = self.model(**inputs).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        top_probs, top_ids = probs.topk(self.top_k)

        values, weights = [], []
        for prob, token_id in zip(top_probs.tolist(), top_ids.tolist()):
            piece = self.tokenizer.decode([token_id]).strip()
            if piece and piece.replace(".", "", 1).isdigit():
                values.append(float(piece))
                weights.append(prob)

        if not values or sum(weights) == 0:
            return None
        total = sum(weights)
        return sum(v * w for v, w in zip(values, weights)) / total