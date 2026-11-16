#!/usr/bin/env python3
"""
Control count for the productivity analysis: how many distinct texts contain
much looser expressions than the patterns of the paper.

The patterns of the paper are strict (lexicon.py, corpus_pass.py). To bound
what they can miss, this pass counts plain words and acronyms with no
qualifier, over every distinct text with a valid occurrence. A text counted
here need not be about productivity measurement (the words have other
senses); the point is the ceiling: the strict counts cannot be missing more
than these.

Code blocks are removed first, as in the other passes.
Outputs in data/productivity/: loose/<corpus>/*.parquet (sha and one flag per
expression), loose_patterns.csv (texts and % of the corpus per expression
and for any of them).
Usage: python scripts/loose_patterns.py skills|context   (resumes if interrupted)
       python scripts/loose_patterns.py summary
"""
from __future__ import annotations

import json
import os
import sys
import time

import duckdb
import pandas as pd
from huggingface_hub import HfApi

from productivity import OCCURRENCES
from vocabulary import CORPORA, q

OUT = os.path.join("data", "productivity")
# RE2; acronyms are case-sensitive, words are not
LOOSE = {
    "dora": r"\bDORA\b",
    "mttr": r"\bMTT[RD]\b",
    "kpi": r"\bKPIs?\b",
    "deployment_frequency": r"(?i)\bdeploy(?:ment)? frequency\b",
    "lead_time": r"(?i)\blead[ -]time\b",
    "cycle_time": r"(?i)\bcycle[ -]time\b",
    "change_failure": r"(?i)\bchange[ -]fail(?:ure)?\b",
    "developer_experience": r"(?i)\bdeveloper experience\b|\bDevEx\b",
    "developer_productivity": r"(?i)\b(?:developer|engineering|team) productivity\b",
    "velocity": r"(?i)\b(?:team|sprint|delivery|development) velocity\b",
}


def scan(name: str) -> None:
    cfg = CORPORA[name]
    out = os.path.join(OUT, "loose", name)
    os.makedirs(out, exist_ok=True)
    done_path = os.path.join(out, "done.json")
    done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
    files = sorted(f for f in HfApi().list_repo_files(cfg["repo"], repo_type="dataset")
                   if f.startswith(cfg["prefix"]) and f.endswith(".parquet"))
    todo = [f for f in files if f not in done]
    print(f"{name}: {len(files)} files, {len(todo)} to scan", flush=True)
    con = duckdb.connect()
    con.sql("CREATE SECRET hf (TYPE huggingface, PROVIDER credential_chain)")
    con.sql("SET threads = 4"); con.sql("SET memory_limit = '2GB'")
    flags = ", ".join(f"regexp_matches(t, {q(p)}) AS {k}" for k, p in LOOSE.items())
    batch = cfg["batch"]
    for k in range(0, len(todo), batch):
        group = todo[k:k + batch]
        file_list = "[" + ",".join(q(f"hf://datasets/{cfg['repo']}/{f}") for f in group) + "]"
        dest = os.path.join(out, f"{len(done):06d}.parquet")
        t0 = time.time()
        for attempt in range(5):
            try:
                con.sql(f"""COPY (
                    WITH a AS (SELECT {cfg['sha']} AS sha,
                                      regexp_replace({cfg['text']}, '(?s)```.*?```', ' ', 'g') AS t
                               FROM read_parquet({file_list}) WHERE {cfg['filter']}),
                         f AS (SELECT sha, {flags} FROM a)
                    SELECT * FROM f WHERE {" OR ".join(LOOSE)}) TO {q(dest)} (FORMAT parquet, COMPRESSION zstd)""")
                break
            except duckdb.Error as e:
                print(f"  error (attempt {attempt + 1}): {str(e)[:300]}", flush=True)
                time.sleep(30 * (attempt + 1))
        else:
            sys.exit("batch failed 5 times; run again to resume")
        done.update(group)
        json.dump(sorted(done), open(done_path, "w"))
        print(f"{time.strftime('%H:%M:%S')} {len(done)}/{len(files)}, {time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)


def summary() -> None:
    rows = []
    for name in CORPORA:
        con = duckdb.connect()
        con.sql(f"CREATE TABLE oc AS SELECT DISTINCT sha FROM read_parquet({q(OCCURRENCES[name])})")
        n_corpus = con.sql("SELECT count(*) FROM oc").fetchall()[0][0]
        src = q(os.path.join(OUT, "loose", name, "*.parquet"))
        con.sql(f"""CREATE TABLE m AS SELECT sha, {', '.join(f'bool_or({k}) AS {k}' for k in LOOSE)}
                    FROM read_parquet({src}) WHERE sha IN (SELECT sha FROM oc) GROUP BY sha""")
        strict = con.sql(f"""SELECT count(*) FROM m WHERE sha IN (SELECT sha FROM
                             read_parquet({q(os.path.join(OUT, f'shas_any_{name}.parquet'))}))""").fetchall()[0][0]
        for k in list(LOOSE) + ["any"]:
            cond = k if k != "any" else " OR ".join(LOOSE)
            t = con.sql(f"SELECT count(*) FROM m WHERE {cond}").fetchall()[0][0]
            rows.append({"corpus": name, "expression": k, "texts": t, "pct": round(100 * t / n_corpus, 4),
                         "n_corpus": n_corpus})
        rows.append({"corpus": name, "expression": "strict texts also matched here", "texts": strict,
                     "pct": None, "n_corpus": n_corpus})
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "loose_patterns.csv"), index=False)
    print(d.to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1:2] and sys.argv[1] in CORPORA:
        scan(sys.argv[1])
    elif sys.argv[1:] == ["summary"]:
        summary()
    else:
        sys.exit(__doc__)
