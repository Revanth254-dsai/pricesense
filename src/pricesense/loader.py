"""Model and tokenizer loading, with a graceful CPU path for laptop inference."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def has_cuda() -> bool:
    return torch.cuda.is_available()


def compute_dtype() -> torch.dtype:
    """bfloat16 where the GPU supports it, float16 otherwise.

    Ampere and later (A100, L4) handle bf16 natively. Turing - which is what a
    free Colab T4 is - does not, and asking for it there costs you silently.
    """
    if has_cuda() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def quant_config() -> BitsAndBytesConfig:
    """NF4 double-quantised 4-bit. Roughly a 4x memory cut over 16-bit."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype(),
    )


def load_tokenizer(base_model: str, padding_side: str = "right"):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def load_model(base_model: str, adapter_repo: str | None = None, *, quantize: bool = True):
    """Load the base model, optionally with a LoRA adapter stacked on top.

    Quantisation needs CUDA. Without a GPU we fall back to fp32 on CPU, which is
    only realistic for the smallest models - and even then it is slow.
    """
    if quantize and has_cuda():
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant_config(),
            device_map="auto",
        )
    else:
        if quantize:
            print("[loader] no CUDA detected - loading fp32 on CPU (slow)")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

    if adapter_repo:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_repo)
        print(f"[loader] adapter attached: {adapter_repo}")

    tokenizer = load_tokenizer(base_model)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model


def memory_footprint(model) -> str:
    return f"{model.get_memory_footprint() / 1e6:,.1f} MB"