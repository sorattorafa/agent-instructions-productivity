#!/usr/bin/env python3
"""
Figures of the paper. Every value drawn comes from the pipeline outputs.

  fig_dataset.pdf  (a) share of occurrences that are copies, skills vs context
               files (data/reuse/summary.csv); (b) share of each artifact's
               documents per topic, one dumbbell per topic, sorted by the
               difference (data/topics/topics.csv, labels from synthesis.py)
  fig_productivity.pdf  (a) distinct files that name a framework or one of its
               metrics, per 10,000, overall and per framework
               (data/productivity/productivity.csv); (b) how the
               files that mention measurement split: framework named only,
               framework and metric, metric only

Colors: skills blue, context files orange, the same in both figures (pair
checked for colour-vision deficiency); every series is also labelled in text.
Output: paper/fig/. Usage: python scripts/figures.py   (run from the repository root)
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from synthesis import LABELS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "paper", "fig")
SKILLS, CONTEXT = "#2a78d6", "#eb6834"
INK, MUTED, GRID, TRACK = "#0b0b0b", "#52514e", "#dddcd6", "#ecebe6"
WIDTH = 3.45  # IEEE column, inches
SEED = 7

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.linewidth": 0.5, "pdf.fonttype": 42, "text.color": INK,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
})


def bare(ax) -> None:
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def panel(ax, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", fontsize=7.5, fontweight="bold", pad=4)


def fig_dataset() -> None:
    r = pd.read_csv("data/reuse/summary.csv").set_index("artifact")
    t = pd.read_csv("data/topics/topics.csv")
    t["label"] = [LABELS[i].split(" (")[0] for i in t.topic]
    t["diff"] = t.pct_skills - t.pct_context
    t = t.sort_values("diff").reset_index(drop=True)

    fig, (a, b) = plt.subplots(2, 1, figsize=(WIDTH, 3.25), gridspec_kw={"height_ratios": [1, 3.6], "hspace": 0.42})
    # (a) copies
    rows = [("Skills", r.loc["SKILL.md"].pct_redundant, SKILLS),
            ("Context files", r.loc["(context, all)"].pct_redundant, CONTEXT)]
    for y, (name, v, c) in zip((1, 0), rows):
        a.barh(y, 100, height=0.5, color=TRACK, zorder=1)
        a.barh(y, v, height=0.5, color=c, zorder=2)
        a.text(v + 1.5, y, f"{v:.1f}%", va="center", ha="left", fontsize=8, fontweight="bold")
    a.set_yticks((1, 0), [x[0] for x in rows])
    a.set_xlim(0, 100); a.set_ylim(-0.5, 1.5); a.set_xticks([])
    bare(a)
    panel(a, "a", "Occurrences that are copies of another file")

    # (b) topics
    y = np.arange(len(t))
    b.hlines(y, t.pct_skills, t.pct_context, color="#bdbcb5", lw=1.6, zorder=1)
    b.scatter(t.pct_context, y, s=30, color=CONTEXT, zorder=3, edgecolor="white", linewidth=0.6)
    b.scatter(t.pct_skills, y, s=30, color=SKILLS, zorder=3, edgecolor="white", linewidth=0.6)
    b.set_yticks(y, t.label)
    b.set_xlim(0, 26); b.set_ylim(-0.6, len(t) - 0.4)
    b.set_xticks(range(0, 26, 5), [f"{v}%" for v in range(0, 26, 5)])
    b.grid(axis="x", color=GRID, lw=0.5); b.set_axisbelow(True)
    bare(b)
    top = t.iloc[-1]
    b.annotate("Skills", (top.pct_skills, y[-1]), xytext=(0, 7), textcoords="offset points",
               ha="center", color=INK, fontsize=7, fontweight="bold")
    b.annotate("Context files", (top.pct_context, y[-1]), xytext=(0, 7), textcoords="offset points",
               ha="center", color=INK, fontsize=7, fontweight="bold")
    b.set_ylim(-0.6, len(t) + 0.15)
    panel(b, "b", "Share of each artifact's files per topic")
    fig.savefig(os.path.join(OUT, "fig_dataset.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_productivity() -> None:
    q = pd.read_csv("data/productivity/productivity.csv")
    arts = (("skills", "Skills", SKILLS), ("context", "Context files", CONTEXT))
    g = {c: q[q.corpus == c].set_index("group") for c, _, _ in arts}

    fig, (a, b) = plt.subplots(2, 1, figsize=(WIDTH, 2.75), gridspec_kw={"height_ratios": [2.1, 1], "hspace": 0.62})
    # (a) rate per 10,000 distinct files, overall and per framework
    groups = [("any framework/any", "Any of them"), ("DORA/any", "DORA"),
              ("DevEx/any", "DevEx"), ("SPACE/any", "SPACE")]
    h = 0.34
    for k, (key, _) in enumerate(groups):
        y0 = len(groups) - 1 - k
        for off, (c, name, color) in zip((h / 2 + 0.02, -h / 2 - 0.02), arts):
            row = g[c].loc[key]
            v = 100 * row.pct
            a.barh(y0 + off, v, height=h, color=color, zorder=2)
            a.text(v + 0.12, y0 + off, f"{v:.1f}  ({int(row.texts):,} files)", va="center", ha="left", fontsize=6.3,
                   color=INK)
    a.set_yticks(range(len(groups)), [lab for _, lab in groups][::-1])
    a.set_xlim(0, 11.3); a.set_ylim(-0.6, len(groups) - 0.4)
    a.set_xticks(range(0, 11, 2))
    a.set_xlabel("per 10,000 distinct files", fontsize=6.5, labelpad=2)
    a.grid(axis="x", color=GRID, lw=0.5); a.set_axisbelow(True)
    bare(a)
    panel(a, "a", "Files that name a framework or one of its metrics")
    a.legend(handles=[matplotlib.patches.Patch(color=color, label=name) for _, name, color in arts],
             loc="lower right", frameon=False, fontsize=6.8, handlelength=1, handletextpad=0.4)

    # (b) split of those files
    shades = ["#0b0b0b", "#8a8983", "#c9c8c1"]
    names = ["Only cite a framework", "Framework and metric", "Metric only"]
    for y, (c, name, _) in zip((1, 0), arts):
        parts = [int(g[c].loc[k].texts) for k in ("named only", "named and metric", "metric only")]
        tot, left = sum(parts), 0.0
        for v, col, ink in zip(parts, shades, ("white", "white", INK)):
            w = 100 * v / tot
            b.barh(y, w - 0.6, left=left + 0.3, height=0.62, color=col)
            b.text(left + w / 2, y, f"{w:.0f}%", ha="center", va="center", color=ink, fontsize=7, fontweight="bold")
            left += w
    b.set_yticks((1, 0), [name for _, name, _ in arts])
    b.set_xlim(0, 100); b.set_ylim(-0.55, 1.55); b.set_xticks([])
    bare(b)
    panel(b, "b", "Files that name a metric vs. files that only cite a framework")
    b.legend(handles=[matplotlib.patches.Patch(color=col, label=nm) for col, nm in zip(shades, names)], ncol=3,
             loc="upper center", bbox_to_anchor=(0.42, -0.02), frameon=False, fontsize=6.3,
             handlelength=1, handletextpad=0.4, columnspacing=1.0)
    fig.savefig(os.path.join(OUT, "fig_productivity.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_dataset()
    fig_productivity()
    print(f"figures in {OUT}")
