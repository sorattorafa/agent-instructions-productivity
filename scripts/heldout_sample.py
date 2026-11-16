#!/usr/bin/env python3
"""
HELD-OUT sample for the topic model (Dataset characterization (topics)): texts the model has not seen,
about 4 times the fit sample.

Same criterion as corpus_pass.py (blob SHA prefix, a uniform random draw that
needs no seed), with prefixes DISJOINT from the fit ones:
  fit        skills 00-07 (8/256)    context 00-04 (5/256)    corpus_pass.py
  held-out   skills 08-27 (32/256)   context 05-18 (20/256)   this script
Downloads only sha and text (up to 30,000 characters); does not redo
mentions/markers.

Output: data/<corpus>/heldout/*.parquet, data/<corpus>/done_heldout.json
(checkpoint; running again resumes).
Usage: python scripts/heldout_sample.py skills|context   (run from the repository root)
"""
from __future__ import annotations

import json
import os
import sys
import time

import duckdb
from huggingface_hub import HfApi

from corpus_pass import PREFIXES as FIT
from vocabulary import CORPORA, q

PREFIXES = {
    "skills": [f"{i:02x}" for i in range(8, 40)],
    "context": [f"{i:02x}" for i in range(5, 25)],
}
for _n in PREFIXES:
    assert not set(PREFIXES[_n]) & set(FIT[_n]), "held-out prefixes must be disjoint from the fit ones"


def main(name: str) -> None:
    cfg = CORPORA[name]
    out = os.path.join("data", name, "heldout")
    os.makedirs(out, exist_ok=True)
    done_path = os.path.join("data", name, "done_heldout.json")
    done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
    files = sorted(f for f in HfApi().list_repo_files(cfg["repo"], repo_type="dataset")
                      if f.startswith(cfg["prefix"]) and f.endswith(".parquet"))
    todo = [f for f in files if f not in done]
    print(f"{name}: {len(files)} files, {len(todo)} to download", flush=True)

    con = duckdb.connect()
    con.sql("CREATE SECRET hf (TYPE huggingface, PROVIDER credential_chain)")
    con.sql("SET threads = 4"); con.sql("SET memory_limit = '2GB'")
    pref = "[" + ",".join(q(p) for p in PREFIXES[name]) + "]"
    batch = cfg["batch"]
    for k in range(0, len(todo), batch):
        group = todo[k:k + batch]
        file_list = "[" + ",".join(q(f"hf://datasets/{cfg['repo']}/{f}") for f in group) + "]"
        dest = os.path.join(out, f"{len(done):06d}.parquet")
        t0 = time.time()
        for attempt in range(5):
            try:
                con.sql(f"""COPY (SELECT {cfg['sha']} AS sha, left({cfg['text']}, 30000) AS text
                                 FROM read_parquet({file_list})
                                 WHERE {cfg['filter']} AND list_contains({pref}, left({cfg['sha']}, 2)))
                           TO {q(dest)} (FORMAT parquet, COMPRESSION zstd)""")
                n = con.sql(f"SELECT count(*) FROM read_parquet({q(dest)})").fetchall()[0][0]
                break
            except duckdb.Error as e:
                print(f"  error (attempt {attempt + 1}): {str(e)[:300]}", flush=True)
                time.sleep(30 * (attempt + 1))
        else:
            sys.exit("batch failed 5 times; run again to resume")
        done.update(group)
        json.dump(sorted(done), open(done_path, "w"))
        print(f"{time.strftime('%H:%M:%S')} {len(done)}/{len(files)}: {n:,} texts, "
              f"{time.time() - t0:.0f}s", flush=True)
    print(f"DONE {time.strftime('%H:%M')}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in PREFIXES:
        sys.exit(__doc__)
    main(sys.argv[1])
