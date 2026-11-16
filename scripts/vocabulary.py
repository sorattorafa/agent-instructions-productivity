#!/usr/bin/env python3
"""
Applies the metric-vocabulary lexicon (lexicon.py, version 2) to the FULL
corpus, by remote query with DuckDB -- no local sample.

Corpora:
  skills    GitSkills (`mvaccargiu/gitskills`), table artifacts (31 shards),
            only the representatives (dedup_primary = 1): 1.88 M distinct
            contents
  context   the context-file dataset (`CONTEXT_DATASET_REPO`),
            textos/*.parquet (productivity analysis)

The dataset ids come from the environment variables `GITSKILLS_REPO` and
`CONTEXT_DATASET_REPO`.

The text is the whole content without fenced code blocks (```...```): a
metric name inside code is an identifier, not an instruction.

Outputs in data/vocabulary/<corpus>/:
  partials/*.parquet   one row per text with some item: sha, flags and a
                       snippet of up to 300 characters around each match
  totals.json          size of the corpus scanned (denominator)
  done.json            checkpoint; running again resumes

Usage: python scripts/vocabulary.py skills|context
"""
from __future__ import annotations

import json
import os
import sys
import time

import duckdb
from huggingface_hub import HfApi

import lexicon

SKILLS_REPO = os.environ.get("GITSKILLS_REPO", "mvaccargiu/gitskills")
CONTEXT_REPO = os.environ.get("CONTEXT_DATASET_REPO", "ANONYMIZED/agent-context-files")
# "textos/" and "texto" are field names of the dataset as collected
CORPORA = {
    "skills": {"repo": SKILLS_REPO, "prefix": "data/artifacts/",
               "sha": "file_sha", "text": "content", "filter": "dedup_primary = 1",
               "batch": 1},
    "context": {"repo": CONTEXT_REPO, "prefix": "textos/",
                "sha": "sha", "text": "texto", "filter": "texto IS NOT NULL",
                "batch": 40},
}


def q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def main(name: str) -> None:
    cfg = CORPORA[name]
    out = os.path.join("data/vocabulary", name)
    partials_dir = os.path.join(out, "partials")
    os.makedirs(partials_dir, exist_ok=True)
    done_path, totals_path = os.path.join(out, "done.json"), os.path.join(out, "totals.json")
    done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
    totals = json.load(open(totals_path)) if os.path.exists(totals_path) else {}

    files = sorted(f for f in HfApi().list_repo_files(cfg["repo"], repo_type="dataset")
                      if f.startswith(cfg["prefix"]) and f.endswith(".parquet"))
    todo = [f for f in files if f not in done]
    print(f"{name}: {len(files)} files, {len(todo)} to scan", flush=True)

    con = duckdb.connect()
    con.sql("CREATE SECRET hf (TYPE huggingface, PROVIDER credential_chain)")
    con.sql("SET threads = 8")

    keys = [i["key"] for i in lexicon.ITEMS]
    flags = ",\n".join(f"regexp_matches(t, {q(lexicon.regex(i))}) AS {i['key']}"
                       for i in lexicon.ITEMS)
    snippets = ",\n".join(
        f"CASE WHEN {i['key']} THEN regexp_extract(t, "
        f"{q('(?is).{0,150}\\b(?:' + i['pattern'] + ')\\b.{0,150}')}) END AS snippet_{i['key']}"
        for i in lexicon.ITEMS)

    batch = cfg["batch"]
    for k in range(0, len(todo), batch):
        group = todo[k:k + batch]
        file_list = "[" + ",".join(q(f"hf://datasets/{cfg['repo']}/{f}") for f in group) + "]"
        dest = os.path.join(partials_dir, f"{len(done):06d}.parquet")
        t0 = time.time()
        for attempt in range(5):
            try:
                con.sql(f"""CREATE OR REPLACE TEMP TABLE m AS
                  WITH a AS (SELECT {cfg['sha']} AS sha,
                                    regexp_replace({cfg['text']}, '(?s)```.*?```', ' ', 'g') AS t
                             FROM read_parquet({file_list}) WHERE {cfg['filter']}),
                       f AS (SELECT sha, t, {flags} FROM a)
                  SELECT sha, {", ".join(keys)}, {snippets} FROM f
                  WHERE {" OR ".join(keys)}""")
                n = con.sql(f"SELECT count(*) FROM read_parquet({file_list}) "
                            f"WHERE {cfg['filter']}").fetchall()[0][0]
                con.sql(f"COPY m TO {q(dest)} (FORMAT parquet, COMPRESSION zstd)")
                break
            except duckdb.Error as e:
                print(f"  error (attempt {attempt + 1}): {str(e)[:200]}", flush=True)
                time.sleep(30 * (attempt + 1))
        else:
            sys.exit("batch failed 5 times; run again to resume")
        found = con.sql("SELECT count(*) FROM m").fetchall()[0][0]
        for f in group:
            done.add(f)
        totals["texts_scanned"] = totals.get("texts_scanned", 0) + n
        json.dump(totals, open(totals_path, "w"))
        json.dump(sorted(done), open(done_path, "w"))
        print(f"{time.strftime('%H:%M:%S')} {len(done)}/{len(files)}: "
              f"{n:,} texts, {found:,} with any item, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in CORPORA:
        sys.exit(__doc__)
    if lexicon.verify(verbose=False):
        sys.exit("lexicon does not match the sources; run lexicon.py")
    main(sys.argv[1])
