"""Stage five: assemble every saved scorecard into one leaderboard."""

from __future__ import annotations

import json

import plotly.graph_objects as go

from .config import RESULTS_DIR

GREY = "#9aa0a6"      # non-learned baselines
DARK_RED = "#8b2f2f"  # untuned base model
SLATE = "#6b73a8"     # off-the-shelf LLMs, zero-shot
RED = "#d94f4f"       # our fine-tune


def _colour(name: str) -> str:
    lowered = name.lower()
    if "constant" in lowered or "word count" in lowered:
        return GREY
    if "zero-shot" in lowered or "ollama" in lowered:
        return SLATE
    if "base" in lowered:
        return DARK_RED
    return RED


def build_leaderboard(open_browser: bool = True) -> None:
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        raise SystemExit("No scorecards in results/ - run `pricesense eval` first")

    cards = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    cards.sort(key=lambda c: c["mae"], reverse=True)

    names = [c["name"] for c in cards]
    values = [c["mae"] for c in cards]

    figure = go.Figure(
        go.Bar(
            x=names,
            y=values,
            marker_color=[_colour(n) for n in names],
            text=[f"${v:,.2f}" for v in values],
            textposition="outside",
            hovertemplate="%{x}<br>MAE $%{y:,.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="PriceSense - mean absolute error by model (lower is better)",
        yaxis_title="Mean absolute error ($)",
        xaxis_tickangle=-35,
        template="plotly_white",
        width=950,
        height=620,
        margin=dict(t=80, b=170),
    )

    out = RESULTS_DIR / "leaderboard.html"
    figure.write_html(out, auto_open=open_browser)
    print(f"[report] {out}")

    print(f"\n  {'model':<38}{'MAE':>12}{'r2':>10}{'hit%':>8}")
    print("  " + "-" * 66)
    for card in sorted(cards, key=lambda c: c["mae"]):
        print(
            f"  {card['name'][:37]:<38}${card['mae']:>10,.2f}"
            f"{card['r2']:>9.1f}%{card['hit_rate']:>7.1f}%"
        )
    print()