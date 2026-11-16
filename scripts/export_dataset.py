#!/usr/bin/env python3
"""
Builds the public release of the context-file dataset from the dataset as
collected (CONTEXT_DATASET_REPO, field names as collected).

What the release contains (folder release/, not tracked in git):
  occurrences.parquet   one row per file kept by name_filter.py:
                        target (file type), repo, path, sha (blob SHA)
  texts/part-*.parquet  one row per distinct content with a kept occurrence:
                        sha, byte_size, is_truncated, text, redactions
  size_ranges.parquet   state of every size range searched by the collector
  masking.json          how many texts and strings were masked

Masking: e-mail addresses and strings with the format of access tokens
(GitHub, OpenAI-style and AWS key formats) are replaced by <redacted>, and
the column redactions counts the replacements in each text. The two kinds
share one placeholder and one count on purpose: the release does not say
which texts held a token. The blob SHA is the one of the original file, so a
text with redactions no longer hashes to it.

Usage: python scripts/export_dataset.py   (run from the repository root;
resumes if interrupted)
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import time

import duckdb
from huggingface_hub import HfApi

from vocabulary import CORPORA, q

OUT = "release"
EMAIL = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
SECRET = r"\bghp_[A-Za-z0-9]{36}\b|\bgithub_pat_[A-Za-z0-9_]{60,}\b|\bsk-[A-Za-z0-9_-]{32,}\b|\bAKIA[0-9A-Z]{16}\b"
BATCH = 10
MERGE = 5
PLACEHOLDER = "<redacted>"


def main() -> None:
    cfg = CORPORA["context"]
    os.makedirs(os.path.join(OUT, "texts"), exist_ok=True)
    done_path = os.path.join(OUT, "done.json")
    done = json.load(open(done_path)) if os.path.exists(done_path) else {}

    con = duckdb.connect()
    con.sql("CREATE SECRET hf (TYPE huggingface, PROVIDER credential_chain)")
    con.sql("SET threads = 2"); con.sql("SET memory_limit = '4GB'")
    con.sql("SET preserve_insertion_order = false")
    con.sql("""COPY (SELECT target, repo, path, sha FROM read_parquet('data/context_occurrences.parquet')
                     ORDER BY target, repo, path)
               TO 'release/occurrences.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    shutil.copy("data/size_ranges.parquet", os.path.join(OUT, "size_ranges.parquet"))
    con.sql("CREATE TABLE kept AS SELECT DISTINCT sha FROM read_parquet('data/context_occurrences.parquet')")

    files = sorted(f for f in HfApi().list_repo_files(cfg["repo"], repo_type="dataset")
                   if f.startswith(cfg["prefix"]) and f.endswith(".parquet"))
    for k in range(0, len(files), BATCH):
        name = f"texts-{k // BATCH:03d}.parquet"
        if name in done:
            continue
        file_list = "[" + ",".join(q(f"hf://datasets/{cfg['repo']}/{f}") for f in files[k:k + BATCH]) + "]"
        dest = os.path.join(OUT, "texts", name)
        t0 = time.time()
        for attempt in range(5):
            try:
                # field names of the dataset as collected: texto
                con.sql(f"""COPY (
                    WITH a AS (SELECT sha, byte_size, is_truncated, texto AS t
                               FROM read_parquet({file_list})
                               WHERE texto IS NOT NULL AND sha IN (SELECT sha FROM kept))
                    SELECT sha, byte_size, is_truncated,
                           regexp_replace(regexp_replace(t, {q(SECRET)}, {q(PLACEHOLDER)}, 'g'),
                                          {q(EMAIL)}, {q(PLACEHOLDER)}, 'g') AS text,
                           len(regexp_extract_all(t, {q(EMAIL)})) AS emails,
                           len(regexp_extract_all(t, {q(SECRET)})) AS secrets
                    FROM a) TO {q(dest)} (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 20000)""")
                break
            except duckdb.Error as e:
                print(f"  error (attempt {attempt + 1}): {str(e)[:300]}", flush=True)
                time.sleep(30 * (attempt + 1))
        else:
            sys.exit("batch failed 5 times; run again to resume")
        n, te, ne, ts, ns = con.sql(f"""SELECT count(*), sum((emails > 0)::int), sum(emails),
                                               sum((secrets > 0)::int), sum(secrets)
                                        FROM read_parquet({q(dest)})""").fetchall()[0]
        done[name] = {"texts": n, "texts_with_email": int(te or 0), "emails": int(ne or 0),
                      "texts_with_secret": int(ts or 0), "secrets": int(ns or 0)}
        json.dump(done, open(done_path, "w"), indent=0)
        print(f"{time.strftime('%H:%M:%S')} {name}: {n:,} texts, {int(te or 0):,} with e-mail, "
              f"{int(ts or 0):,} with token, {time.time() - t0:.0f}s", flush=True)

    # Consolidation: the parts are merged five by five, and the two counts become
    # one column. The released files do not say which texts held a token, so the
    # dataset cannot be used to locate them in the source repositories; the
    # totals stay in masking.json.
    parts = sorted(glob.glob(os.path.join(OUT, "texts", "texts-*.parquet")))
    for k in range(0, len(parts), MERGE):
        group = parts[k:k + MERGE]
        dest = os.path.join(OUT, "texts", f"part-{k // MERGE:02d}.parquet")
        srcs = "[" + ",".join(q(f) for f in group) + "]"
        con.sql(f"""COPY (SELECT sha, byte_size, is_truncated, text, emails + secrets AS redactions
                         FROM read_parquet({srcs}))
                   TO {q(dest)} (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 20000)""")
        for f in group:
            os.remove(f)
        print(f"{time.strftime('%H:%M:%S')} {os.path.basename(dest)}", flush=True)
    total = con.sql("SELECT count(DISTINCT sha) FROM read_parquet('release/texts/part-*.parquet')").fetchall()[0][0]
    summary = {"distinct_texts": total, "placeholder": PLACEHOLDER,
               **{k: sum(d[k] for d in done.values()) for k in ("texts_with_email", "emails", "texts_with_secret", "secrets")}}
    json.dump(summary, open(os.path.join(OUT, "masking.json"), "w"), indent=1)
    os.makedirs(os.path.join("data", "dataset"), exist_ok=True)
    json.dump(summary, open(os.path.join("data", "dataset", "masking.json"), "w"), indent=1)  # read by synthesis.py
    print(json.dumps(summary, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
