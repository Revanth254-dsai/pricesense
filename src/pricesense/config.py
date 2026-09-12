"""Central configuration. Every tunable number in the project lives here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# --- paths -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = ROOT / "checkpoints"


def ensure_dirs() -> None:
    for directory in (DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# --- the prompt contract -----------------------------------------------------
# The model never writes prose. It sees a question, a description, and a dangling
# dollar sign - so the only sensible continuation is digits. PRICE_PREFIX doubles
# as the marker the training collator uses to mask the loss on everything before
# the answer.

QUESTION = "What does this cost to the nearest dollar?"
PRICE_PREFIX = "Price is $"


# --- helpers -----------------------------------------------------------------


def normalise_user(raw: str) -> str:
    """Accept a bare username or a pasted profile URL and return the username.

    The Hub wants 'namespace/repo'. Pasting a full profile URL silently builds
    a three-segment id that only fails at push time, minutes into a run.
    """
    cleaned = raw.strip().strip("/")
    for prefix in ("https://", "http://", "www.", "huggingface.co/"):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):]
    return cleaned.split("/")[0]


# --- settings ----------------------------------------------------------------


@dataclass(frozen=True)
class DataSettings:
    source_repo: str = "ed-donner/items_lite"
    pairs_repo: str = ""          # blank -> derived from HF_USER
    base_model: str = "Qwen/Qwen2.5-0.5B"
    summary_cutoff: int = 110     # tokens kept from each description


@dataclass(frozen=True)
class AdapterSettings:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    targets: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
    )


@dataclass(frozen=True)
class TrainSettings:
    epochs: int = 1
    batch_size: int = 4
    grad_accum: int = 1
    learning_rate: float = 5e-5
    scheduler: str = "cosine"
    warmup_ratio: float = 0.01
    optimizer: str = "paged_adamw_32bit"
    max_seq_len: int = 144        # measured: longest sequence is 129 tokens
    save_every: int = 500
    log_every: int = 50
    seed: int = 42
    train_cap: int = 0            # 0 means use the whole training split


@dataclass(frozen=True)
class Settings:
    hf_user: str
    run_name: str
    data: DataSettings = field(default_factory=DataSettings)
    adapter: AdapterSettings = field(default_factory=AdapterSettings)
    train: TrainSettings = field(default_factory=TrainSettings)

    @property
    def pairs_repo(self) -> str:
        """Where the prompt/completion dataset lives on the Hub."""
        return self.data.pairs_repo or f"{self.hf_user}/pricesense-pairs"

    @property
    def adapter_repo(self) -> str:
        """Where the trained LoRA adapter gets pushed."""
        return f"{self.hf_user}/{self.run_name}"

    @classmethod
    def load(cls) -> Settings:
        return cls(
            hf_user=normalise_user(os.environ.get("HF_USER", "local-user")),
            run_name=os.environ.get("RUN_NAME", "pricesense-llama-1b"),
        )


def hf_login() -> None:
    """Authenticate with the Hub if a token is present. Quiet no-op otherwise."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[config] HF_TOKEN not set - Hub pushes and gated models unavailable")
        return
    from huggingface_hub import login

    login(token, add_to_git_credential=False)