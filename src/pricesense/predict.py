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
    """A callable price estimator: prompt in, dollars out.

    Decoding is greedy. A probability-weighted variant was tried - blending the
    top-k candidates for the first token after the price prefix - and removed:
    it assumes the tokenizer groups digits, and Qwen emits them one at a time,
    so the blend averaged leading digits and returned single-dollar estimates
    for hundred-dollar items.
    """

    def __init__(
        self,
        base_model: str,
        adapter_repo: str | None = None,
        *,
        max_new_tokens: int = 6,
    ):
        self.tokenizer = load_tokenizer(base_model, padding_side="left")
        self.model = load_model(base_model, adapter_repo)
        self.max_new_tokens = max_new_tokens
        self.name = adapter_repo or f"{base_model.split('/')[-1]} (base)"

    @torch.no_grad()
    def __call__(self, prompt: str) -> float:
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