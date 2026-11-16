#!/usr/bin/env python3
"""
Agreement between the model raters that validated the productivity patterns.

Each rater received the same instructions (validation/llm/PROMPT.md) and the
same items, and labelled alone:
  claude   validation/llm/claude/{precision,recall}_*.json
  gemini   validation/llm/gemini/{precision,recall}.json   (llm_rater_gemini.py)
  openai   validation/llm/openai/{precision,recall}.json   (llm_rater_openai.py)
A rater enters a task only when it answered every item.

For each task (precision: 344 matches of the patterns; recall: 200 files
with no match):
  share of y per rater, with a Wilson 95% interval; by majority vote; and
  where all raters answer y
  Cohen's kappa and simple agreement for every pair of raters
  Fleiss' kappa over all raters
Kappa is undefined when every label is the same; it is then reported as null,
with the agreement.

Output: data/validation/llm_summary.json
Usage: python scripts/llm_agreement.py   (run from the repository root)
"""
from __future__ import annotations

import glob
import itertools
import json
import os

import pandas as pd

from statistical_tests import wilson

SRC = os.path.join("validation", "llm")
OUT = os.path.join("data", "validation")
RATERS = ("claude", "gemini", "openai")


def load(rater: str, task: str, n: int) -> pd.Series | None:
    files = sorted(glob.glob(os.path.join(SRC, rater, f"{task}*.json")))
    rows = [r for f in files for r in json.load(open(f))]
    s = pd.Series({int(r["id"]): str(r["answer"]).strip().lower()[:1] for r in rows}).sort_index()
    return s if len(s) == n and len(rows) == n and s.isin(["y", "n"]).all() else None


def share(s: pd.Series) -> dict:
    k, n = int((s == "y").sum()), len(s)
    return {"y": k, "n": n, "p": k / n, "ci95": list(wilson(k, n))}


def cohen(a: pd.Series, b: pd.Series) -> dict:
    po = float((a == b).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in ("y", "n"))
    return {"agreement": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else None,
            "both_y": int(((a == "y") & (b == "y")).sum()), "both_n": int(((a == "n") & (b == "n")).sum()),
            "disagree": int((a != b).sum())}


def fleiss(d: pd.DataFrame) -> float | None:
    m = d.shape[1]
    ny = (d == "y").sum(axis=1)
    p_item = (ny * (ny - 1) + (m - ny) * (m - ny - 1)) / (m * (m - 1))
    py = ny.sum() / (len(d) * m)
    pe = py ** 2 + (1 - py) ** 2
    return float((p_item.mean() - pe) / (1 - pe)) if pe < 1 else None


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    res: dict = {"models": {}}
    for r in RATERS:
        run = os.path.join(SRC, r, "run.json")
        if os.path.exists(run):
            res["models"][r] = json.load(open(run))["model"]
    for task in ("precision", "recall"):
        n = sum(1 for _ in open(os.path.join(SRC, f"{task}_items.jsonl")))
        got = {r: s for r in RATERS if (s := load(r, task, n)) is not None}
        d = pd.DataFrame(got)
        out: dict = {"items": n, "raters": list(got), "share_y": {r: share(d[r]) for r in d}}
        if len(got) >= 2:
            votes = (d == "y").sum(axis=1)
            out["majority_y"] = share(pd.Series(["y" if v * 2 > len(got) else "n" for v in votes]))
            out["all_y"] = share(pd.Series(["y" if v == len(got) else "n" for v in votes]))
            out["all_agree"] = float(((votes == 0) | (votes == len(got))).mean())
            out["pairs"] = {f"{a}_vs_{b}": cohen(d[a], d[b]) for a, b in itertools.combinations(got, 2)}
            out["fleiss_kappa"] = fleiss(d)
            out["items_with_disagreement"] = [int(i) for i in d.index[(votes > 0) & (votes < len(got))]]
        res[task] = out
    json.dump(res, open(os.path.join(OUT, "llm_summary.json"), "w"), indent=1)
    print(json.dumps({k: ({x: y for x, y in v.items() if x != "items_with_disagreement"} if isinstance(v, dict) else v)
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
