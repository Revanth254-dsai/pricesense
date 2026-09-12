"""PriceSense - predicting product prices from descriptions with a fine-tuned LLM."""

__version__ = "0.1.0"

from .config import PRICE_PREFIX, QUESTION, Settings
from .dataset import Item, Splits

__all__ = ["Settings", "Item", "Splits", "QUESTION", "PRICE_PREFIX"]