#!/usr/bin/env python3
"""
Items for the model raters (validation/llm/PROMPT.md).

  precision  every match of the patterns that the paper counts, in both
             datasets (a census, not a sample). Ids 1-344 are the matches of
             the first draw (validation_sheets.py); the others follow.
  recall     1,000 files with no match at all, 500 per dataset, drawn at
             random from the SHA-prefix sample of each one. Ids 1-200 are the
             files of the first draw; the others follow.
E-mail addresses and strings with the format of access tokens are redacted
before the text leaves the machine.

Outputs in validation/llm/: precision_items.jsonl, recall_items.jsonl and the
keys precision_items.csv, recall_items.csv (id, corpus, kind/item, sha).
Usage: python scripts/validation_items.py   (run from the repository root)
"""
from __future__ import annotations

import json
import os
import re

import duckdb
import pandas as pd

import validation_sheets as V
from productivity import OCCURRENCES
from vocabulary import q

OUT = os.path.join("validation", "llm")
RECALL_PER_CORPUS = 500
SEED = 2028
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")


SECRET = re.compile(r"\bghp_[A-Za-z0-9]{36}\b|\bgithub_pat_[A-Za-z0-9_]{60,}\b|\bsk-[A-Za-z0-9_-]{32,}\b|\bAKIA[0-9A-Z]{16}\b")


def mask(t: str) -> str:
    return EMAIL.sub("<redacted>", SECRET.sub("<redacted>", str(t)))


def main() -> None:
    # precision: all matches
    first = pd.read_csv(os.path.join("validation", "precision_sample.csv"))
    m = V.matches().drop_duplicates(["corpus", "item", "sha"])
    m = m.merge(first[["id", "corpus", "item", "sha"]], on=["corpus", "item", "sha"], how="left")
    rest = m[m.id.isna()].sort_values(["corpus", "item", "sha"]).copy()
    rest["id"] = range(len(first) + 1, len(first) + 1 + len(rest))
    p = pd.concat([m[m.id.notna()], rest]).astype({"id": int}).sort_values("id")
    p["meaning"] = p["item"].map(V.MEANING)
    assert p.id.is_unique and len(p) == len(m)
    with open(os.path.join(OUT, "precision_items.jsonl"), "w") as f:
        for r in p.itertuples():
            f.write(json.dumps({"id": r.id, "kind": r.kind, "item": r.item, "meaning": r.meaning,
                                "snippet": mask(r.snippet)}, ensure_ascii=False) + "\n")
    p[["id", "corpus", "kind", "item", "sha"]].to_csv(os.path.join(OUT, "precision_items.csv"), index=False)

    # recall: the first draw plus new files, up to RECALL_PER_CORPUS per dataset
    first = pd.read_csv(os.path.join("validation", "recall_sample.csv"))
    rows, nxt = [], int(first.id.max()) + 1
    for corpus in ("skills", "context"):
        con = duckdb.connect()
        con.sql("SET threads = 2"); con.sql("SET memory_limit = '1GB'")
        hit = set(pd.read_parquet(f"data/productivity/shas_any_{corpus}.parquet").sha)
        ids = con.sql(f"""SELECT DISTINCT sha FROM read_parquet({q(f'data/{corpus}/sample/*.parquet')})
                          WHERE sha IN (SELECT DISTINCT sha FROM read_parquet({q(OCCURRENCES[corpus])}))""").df()
        old = first[first.corpus == corpus]
        pool = ids[~ids.sha.isin(hit) & ~ids.sha.isin(set(old.sha))].sort_values("sha")
        new = pool.sample(n=RECALL_PER_CORPUS - len(old), random_state=SEED)
        new = new.assign(corpus=corpus, id=range(nxt, nxt + len(new)))
        nxt += len(new)
        pick = pd.concat([old[["id", "corpus", "sha"]], new[["id", "corpus", "sha"]]])
        con.register("pick", pick)
        t = con.sql(f"""SELECT sha, any_value(text) AS text FROM read_parquet({q(f'data/{corpus}/sample/*.parquet')})
                        WHERE sha IN (SELECT sha FROM pick) GROUP BY sha""").df()
        rows.append(pick.merge(t, on="sha"))
    u = pd.concat(rows).sort_values("id")
    assert u.id.is_unique and len(u) == 2 * RECALL_PER_CORPUS
    with open(os.path.join(OUT, "recall_items.jsonl"), "w") as f:
        for r in u.itertuples():
            t = str(r.text)
            t = t[:V.MAX_CHARS] + ("\n[... cut at 6,000 characters]" if len(t) > V.MAX_CHARS else "")
            f.write(json.dumps({"id": int(r.id), "text": mask(t)}, ensure_ascii=False) + "\n")
    u[["id", "corpus", "sha"]].to_csv(os.path.join(OUT, "recall_items.csv"), index=False)
    print(f"precision: {len(p)} matches; recall: {len(u)} files")


if __name__ == "__main__":
    main()
