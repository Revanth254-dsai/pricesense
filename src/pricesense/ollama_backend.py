"""A locally-served LLM as a reference point, via Ollama's HTTP API.

This is the 'competent generalist' bar. It has never seen the training data and
knows nothing about this task beyond what the prompt tells it - which is exactly
what makes it the right thing to measure a fine-tune against.

Stdlib only. Ollama is an optional dependency of the project, not a required one.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .predict import parse_price

DEFAULT_HOST = "http://localhost:11434"


class OllamaUnavailable(RuntimeError):
    pass


def _post(host: str, path: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        f"{host}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OllamaUnavailable(
            f"Cannot reach Ollama at {host} - is the service running? ({exc})"
        ) from exc


def available_models(host: str = DEFAULT_HOST, timeout: int = 10) -> list[str]:
    request = urllib.request.Request(f"{host}/api/tags")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            tags = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OllamaUnavailable(f"Cannot reach Ollama at {host} ({exc})") from exc
    return [entry["name"] for entry in tags.get("models", [])]


class OllamaPricer:
    """Prompt in, dollars out - same contract as PriceSense."""

    def __init__(
        self,
        model: str = "llama3.2",
        host: str | None = None,
        timeout: int = 120,
        max_tokens: int = 6,
    ):
        self.model = model
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.name = f"{model} (Ollama, zero-shot)"

    def warmup(self) -> None:
        """Fail loudly now rather than 40 items into a scoring run."""
        models = available_models(self.host)
        stems = {name.split(":")[0] for name in models}
        if self.model not in models and self.model.split(":")[0] not in stems:
            raise OllamaUnavailable(
                f"Model '{self.model}' not found. Available: {', '.join(models) or 'none'}\n"
                f"Pull it with:  ollama pull {self.model}"
            )

    def __call__(self, prompt: str) -> float:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            # raw=True skips the chat template. Our prompt is a completion that
            # dead-ends on 'Price is $' - wrapping it in a chat turn would
            # invite the model to answer with a sentence instead of a number.
            "raw": True,
            "options": {
                "temperature": 0,
                "num_predict": self.max_tokens,
                "stop": ["\n"],
            },
        }
        result = _post(self.host, "/api/generate", payload, self.timeout)
        return parse_price(result.get("response", ""))