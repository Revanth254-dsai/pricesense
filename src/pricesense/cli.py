"""Command line entry point. Every stage of the pipeline hangs off here."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from .config import PRICE_PREFIX, QUESTION, Settings


def _settings(args) -> Settings:
    base = Settings.load()

    data = base.data
    if getattr(args, "base_model", None):
        data = replace(data, base_model=args.base_model)
    if getattr(args, "source", None):
        data = replace(data, source_repo=args.source)

    train = base.train
    if getattr(args, "cap", None):
        train = replace(train, train_cap=args.cap)

    return replace(base, data=data, train=train)


def cmd_prepare(args) -> None:
    from .prepare import prepare

    prepare(_settings(args), push=args.push)


def cmd_baseline(args) -> None:
    from .baselines import ConstantPricer, WordCountPricer
    from .evaluate import score
    from .prepare import read_jsonl

    train_pairs = read_jsonl("train")
    test_pairs = read_jsonl("test")

    for factory in (ConstantPricer, WordCountPricer):
        model = factory.fit(train_pairs)
        score(model, test_pairs, model.name, n=args.n, verbose=False)


def cmd_ollama(args) -> None:
    from .evaluate import score
    from .ollama_backend import OllamaPricer, OllamaUnavailable
    from .prepare import read_jsonl

    model = OllamaPricer(args.model, host=args.host)
    try:
        model.warmup()
    except OllamaUnavailable as exc:
        raise SystemExit(f"\n{exc}\n")

    score(model, read_jsonl("test"), args.label or model.name, n=args.n)


def cmd_train(args) -> None:
    from .train import train

    train(_settings(args), pairs_repo=args.pairs, use_wandb=args.wandb)


def cmd_eval(args) -> None:
    from .evaluate import score
    from .predict import PriceSense
    from .prepare import read_jsonl

    settings = _settings(args)
    adapter = None if args.base else (args.adapter or settings.adapter_repo)

    model = PriceSense(settings.data.base_model, adapter, weighted=args.weighted)
    score(model, read_jsonl("test"), args.label or model.name, n=args.n)


def cmd_report(args) -> None:
    from .report import build_leaderboard

    build_leaderboard(open_browser=not args.no_open)


def cmd_price(args) -> None:
    prompt = f"{QUESTION}\n\n{args.description}\n\n{PRICE_PREFIX}"

    if args.ollama:
        from .ollama_backend import OllamaPricer, OllamaUnavailable

        model = OllamaPricer(args.ollama)
        try:
            model.warmup()
        except OllamaUnavailable as exc:
            raise SystemExit(f"\n{exc}\n")
    else:
        from .predict import PriceSense

        settings = _settings(args)
        adapter = None if args.base else (args.adapter or settings.adapter_repo)
        model = PriceSense(settings.data.base_model, adapter, weighted=args.weighted)

    print(f"\n  {model.name} estimates ${model(prompt):,.2f}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pricesense",
        description="Predict product prices with a QLoRA fine-tuned LLM",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="build prompt/completion pairs")
    p.add_argument("--source", help="raw dataset repo on the Hub")
    p.add_argument("--base-model", dest="base_model")
    p.add_argument("--push", action="store_true", help="publish pairs to the Hub")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("baseline", help="score the non-LLM reference models")
    p.add_argument("-n", type=int, default=200)
    p.set_defaults(func=cmd_baseline, base_model=None, source=None)

    p = sub.add_parser("ollama", help="score a locally served model via Ollama")
    p.add_argument("--model", default="llama3.2")
    p.add_argument("--host", help="defaults to http://localhost:11434")
    p.add_argument("--label", help="name for the leaderboard")
    p.add_argument("-n", type=int, default=200)
    p.set_defaults(func=cmd_ollama, base_model=None, source=None)

    p = sub.add_parser("train", help="QLoRA fine-tune (CUDA required)")
    p.add_argument("--pairs", help="pairs dataset repo")
    p.add_argument("--base-model", dest="base_model")
    p.add_argument("--cap", type=int, help="limit training examples")
    p.add_argument("--wandb", action="store_true")
    p.set_defaults(func=cmd_train, source=None)

    p = sub.add_parser("eval", help="score a model on the test split")
    p.add_argument("--adapter", help="adapter repo (defaults to your run)")
    p.add_argument("--base", action="store_true", help="score the untuned base model")
    p.add_argument("--base-model", dest="base_model")
    p.add_argument("--label", help="name for the leaderboard")
    p.add_argument("--weighted", action="store_true", help="probability-weighted decoding")
    p.add_argument("-n", type=int, default=200)
    p.set_defaults(func=cmd_eval, source=None)

    p = sub.add_parser("report", help="build the leaderboard")
    p.add_argument("--no-open", action="store_true")
    p.set_defaults(func=cmd_report, base_model=None, source=None)

    p = sub.add_parser("price", help="estimate one product")
    p.add_argument("description")
    p.add_argument("--adapter")
    p.add_argument("--base", action="store_true")
    p.add_argument("--base-model", dest="base_model")
    p.add_argument("--ollama", metavar="MODEL", help="use an Ollama model instead")
    p.add_argument("--weighted", action="store_true")
    p.set_defaults(func=cmd_price, source=None)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())