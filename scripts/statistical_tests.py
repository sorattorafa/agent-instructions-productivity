#!/usr/bin/env python3
"""
Statistical tests for the results: chi-square with Cramer's V to compare
distributions, ratio of proportions with a confidence interval, Wilson
intervals for proportions.

What is a census and what is a sample: the reuse rates and the productivity
counts are counts over the WHOLE collected corpus (there is no sampling error
to estimate; the uncertainty lies in the coverage of the collection and in
what the patterns match). The only sample is the one of the topic model.

Productivity analysis
  skills vs context   ratio of the "any framework" rates (95% CI, log) and
             2x2 chi-square with Cramer's V, over distinct contents.
  file types          chi-square (type x mentions) with Cramer's V, and the
             ratio .claude/agents / AGENTS.md. Count per type =
             total x % from productivity_by_target.csv (% with 4 decimals:
             error < 1 text).
Dataset characterization, topics (fit sample, assignment.parquet: distinct contents)
  chi-square topic x artifact and topic x (AGENTS.md, CLAUDE.md), Cramer's V,
  adjusted standardized residuals of the cells cited in the text, and the
  largest Wilson CI half-width of the % per topic.
Output: data/statistics/summary.json
Usage: python scripts/statistical_tests.py   (run from the repository root)
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm

import lexicon

OUT = os.path.join("data", "statistics")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z = norm.ppf(0.975)


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    c = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return c - h, c + h


def ratio(k1: int, n1: int, k2: int, n2: int) -> dict:
    rr = (k1 / n1) / (k2 / n2)
    se = np.sqrt(1 / k1 - 1 / n1 + 1 / k2 - 1 / n2)
    return {"ratio": rr, "ci95": [rr * np.exp(-Z * se), rr * np.exp(Z * se)]}


def chi_square(tab: np.ndarray) -> dict:
    chi2, p, dof, expected = chi2_contingency(tab, correction=False)
    n = tab.sum()
    v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
    # adjusted standardized residual (Agresti)
    lin, col = tab.sum(axis=1, keepdims=True) / n, tab.sum(axis=0, keepdims=True) / n
    res = (tab - expected) / np.sqrt(expected * (1 - lin) * (1 - col))
    return {"chi2": float(chi2), "dof": int(dof), "p": float(p), "cramers_v": float(v), "n": int(n), "residuals": res}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    res: dict = {"productivity": {}, "topics": {}}

    # --- Productivity analysis: skills vs context and file types
    q = pd.read_csv("data/productivity/productivity.csv")
    any_row = q[q.group == "any framework/any"].set_index("corpus")
    ks, ns = int(any_row.loc["skills", "texts"]), int(any_row.loc["skills", "n_corpus"])
    kc, nc = int(any_row.loc["context", "texts"]), int(any_row.loc["context", "n_corpus"])
    res["productivity"]["skills_vs_context"] = {**ratio(ks, ns, kc, nc),
                                       **{x: y for x, y in chi_square(np.array([[ks, ns - ks], [kc, nc - kc]])).items()
                                          if x != "residuals"}}
    pa = pd.read_csv("data/productivity/productivity_by_target.csv")
    pa = pa[pa.total > 0]
    pa["k"] = (pa.total * pa["any framework/any"] / 100).round().astype(int)
    tab = np.column_stack([pa.k, pa.total - pa.k])
    qq = chi_square(tab)
    res["productivity"]["file_types"] = {x: y for x, y in qq.items() if x != "residuals"}
    res["productivity"]["file_types"]["types"] = len(pa)
    i = pa.set_index("target")
    res["productivity"]["claude_agents_vs_agents_md"] = ratio(int(i.loc[".claude/agents", "k"]), int(i.loc[".claude/agents", "total"]),
                                                    int(i.loc["AGENTS.md", "k"]), int(i.loc["AGENTS.md", "total"]))

    # --- Dataset characterization (topics): topics
    at = pd.read_parquet("data/topics/assignment.parquet")
    k = int(at.topic.max()) + 1
    tab = pd.crosstab(at.artifact, at.topic).reindex(columns=range(k), fill_value=0)
    qq = chi_square(tab.values)
    r = pd.DataFrame(qq.pop("residuals"), index=tab.index, columns=tab.columns)
    res["topics"]["topic_by_artifact"] = {**qq, "skill_residual_by_topic": r.loc["skill"].round(1).to_dict()}
    ct = at[at.target.isin(["AGENTS.md", "CLAUDE.md"])]
    tab = pd.crosstab(ct.target, ct.topic).reindex(columns=range(k), fill_value=0)
    qq = chi_square(tab.values)
    r = pd.DataFrame(qq.pop("residuals"), index=tab.index, columns=tab.columns)
    res["topics"]["topic_by_agents_claude"] = {**qq, "agents_residual_by_topic": r.loc["AGENTS.md"].round(1).to_dict()}
    half = 0.0
    for art, g in at.groupby("artifact"):
        for t in range(k):
            lo, hi = wilson(int((g.topic == t).sum()), len(g))
            half = max(half, (hi - lo) / 2)
    res["topics"]["max_ci95_half_width_pp"] = 100 * half

    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
