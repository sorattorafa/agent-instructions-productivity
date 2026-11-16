#!/usr/bin/env python3
"""
One remote pass per corpus (DuckDB over Hugging Face) that gathers everything
the paper needs from the full text:

  1. EXPLICIT mention of the frameworks (productivity analysis): DORA, SPACE, DevEx, DX Core 4
     and the phrase "developer/engineering productivity". Strict patterns:
     the acronym alone does not count, it has to come attached to
     metric/framework/research.
  2. markers of tool-generated text (dataset characterization): header of Claude Code's /init,
     block of the claude-mem plugin, pointer file (the whole text is just the
     name of another context file).
  3. reproducible sample for the topic model (dataset characterization): every text whose blob SHA
     starts with one of the prefixes in PREFIXES. The SHA is a hash, so the
     prefix is a uniform random draw that anyone can redo without a seed.

As in vocabulary.py: fenced code blocks (```...```) are removed before
matching the patterns; the sample keeps the original text.

Outputs in data/<corpus>/:
  mentions/*.parquet  sha, mention flags and a snippet of up to 300 characters
  markers/*.parquet   sha, marker flags, size
  sample/*.parquet    sha, text (up to 30,000 characters)
  totals.json, done.json (checkpoint; running again resumes)

Usage: python scripts/corpus_pass.py skills|context
"""
from __future__ import annotations

import json
import os
import sys
import time

import duckdb
from huggingface_hub import HfApi

from vocabulary import CORPORA, q

# skills: 8/256 of 1.88 M ~ 59 thousand; context: 5/256 of 2.95 M ~ 58 thousand
PREFIXES = {
    "skills": [f"{i:02x}" for i in range(8)],
    "context": [f"{i:02x}" for i in range(5)],
}

# RE2. (?i:...) limits case-insensitivity to the group; SPACE in upper case
# on purpose (avoids "space" as in whitespace/key/disk).
MENTIONS = {
    "dora": r"(?i:\bdora\b[ -]?(?:metrics?|four keys|key metrics|research|report|capabilities"
            r"|benchmarks?|performance levels?|framework|survey)\b)"
            r"|(?i:\b(?:state of devops report|accelerate state of devops|four key metrics)\b)",
    "space": r"\bSPACE[ -](?:framework|metrics|model|dimensions)\b"
             r"|(?i:\bspace (?:framework|metrics) (?:for|of) (?:developer|engineering) productivity\b)"
             r"|(?i:satisfaction and well-?being, performance, activity)",
    "devex": r"(?i:\bdevex\b[ -]?(?:framework|metrics?|survey|dimensions)\b)"
             r"|(?i:feedback loops,? cognitive load,? and flow state)",
    "dx_core4": r"(?i:\bdx core ?4\b)",
    "dev_productivity": r"(?i:\b(?:developer|engineering) productivity\b)",
}

MARKERS = {
    # header that Claude Code's /init writes to CLAUDE.md
    "init_claude": "(?i)this file provides guidance to claude code",
    "claude_mem": "<claude-mem-context>",
}
# pointer: the whole text (ignoring leading and trailing spaces and line breaks) is just the
# name of another context file, with or without the import @
POINTER = r"^\s*@?\.?/?(AGENTS|CLAUDE|GEMINI)\.md\s*$"


def main(name: str) -> None:
    cfg = CORPORA[name]
    out = os.path.join("data", name)
    for d in ("mentions", "markers", "sample"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    done_path, totals_path = os.path.join(out, "done.json"), os.path.join(out, "totals.json")
    done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
    totals = json.load(open(totals_path)) if os.path.exists(totals_path) else {}

    files = sorted(f for f in HfApi().list_repo_files(cfg["repo"], repo_type="dataset")
                      if f.startswith(cfg["prefix"]) and f.endswith(".parquet"))
    todo = [f for f in files if f not in done]
    print(f"{name}: {len(files)} files, {len(todo)} to scan", flush=True)

    con = duckdb.connect()
    con.sql("CREATE SECRET hf (TYPE huggingface, PROVIDER credential_chain)")
    con.sql("SET threads = 4"); con.sql("SET memory_limit = '2GB'")

    ks = list(MENTIONS)
    flags = ", ".join(f"regexp_matches(t, {q(p)}) AS {k}" for k, p in MENTIONS.items())
    snippets = ", ".join(
        f"CASE WHEN {k} THEN regexp_extract(t, {q('(?s).{0,150}(?:' + p + ').{0,150}')}) END AS snippet_{k}"
        for k, p in MENTIONS.items())
    marker_sql = ", ".join(f"regexp_matches(text, {q(p)}) AS {k}" for k, p in MARKERS.items())
    pref = "[" + ",".join(q(p) for p in PREFIXES[name]) + "]"

    batch = cfg["batch"]
    for k in range(0, len(todo), batch):
        group = todo[k:k + batch]
        file_list = "[" + ",".join(q(f"hf://datasets/{cfg['repo']}/{f}") for f in group) + "]"
        suffix = f"{len(done):06d}.parquet"
        t0 = time.time()
        for attempt in range(5):
            try:
                con.sql(f"""CREATE OR REPLACE TEMP TABLE base AS
                  SELECT {cfg['sha']} AS sha, {cfg['text']} AS text
                  FROM read_parquet({file_list}) WHERE {cfg['filter']}""")
                con.sql(f"""CREATE OR REPLACE TEMP TABLE m AS
                  WITH a AS (SELECT sha, regexp_replace(text, '(?s)```.*?```', ' ', 'g') AS t FROM base),
                       f AS (SELECT sha, t, {flags} FROM a)
                  SELECT sha, {", ".join(ks)}, {snippets} FROM f WHERE {" OR ".join(ks)}""")
                con.sql(f"COPY m TO {q(os.path.join(out, 'mentions', suffix))} (FORMAT parquet, COMPRESSION zstd)")
                con.sql(f"""COPY (SELECT sha, length(text) AS n_chars, {marker_sql},
                                  regexp_matches(text, {q(POINTER)}) AS pointer
                           FROM base) TO {q(os.path.join(out, 'markers', suffix))}
                           (FORMAT parquet, COMPRESSION zstd)""")
                con.sql(f"""COPY (SELECT sha, left(text, 30000) AS text FROM base
                                 WHERE list_contains({pref}, left(sha, 2)))
                           TO {q(os.path.join(out, 'sample', suffix))} (FORMAT parquet, COMPRESSION zstd)""")
                n = con.sql("SELECT count(*) FROM base").fetchall()[0][0]
                found = con.sql("SELECT count(*) FROM m").fetchall()[0][0]
                break
            except duckdb.Error as e:
                print(f"  error (attempt {attempt + 1}): {str(e)[:300]}", flush=True)
                time.sleep(30 * (attempt + 1))
        else:
            sys.exit("batch failed 5 times; run again to resume")
        for f in group:
            done.add(f)
        totals["texts_scanned"] = totals.get("texts_scanned", 0) + n
        json.dump(totals, open(totals_path, "w"))
        json.dump(sorted(done), open(done_path, "w"))
        print(f"{time.strftime('%H:%M:%S')} {len(done)}/{len(files)}: "
              f"{n:,} texts, {found:,} with a mention, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in CORPORA:
        sys.exit(__doc__)
    main(sys.argv[1])
