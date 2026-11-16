#!/usr/bin/env python3
"""
Dataset characterization: length of the files, in characters, per file type.

Reads the length of every distinct content recorded by corpus_pass.py
(data/<corpus>/markers) and keeps the contents with a valid occurrence.
A content that occurs under more than one context file type counts in each.
Output: data/reuse/file_lengths.csv (artifact, distinct contents, median and
quartiles of the length).
Usage: python scripts/file_lengths.py   (run from the repository root)
"""
from __future__ import annotations

import duckdb
import pandas as pd

STATS = ("count(*) AS contents, median(n_chars) AS median_chars, "
         "quantile_cont(n_chars, 0.25) AS q1_chars, quantile_cont(n_chars, 0.75) AS q3_chars")


def main() -> None:
    con = duckdb.connect()
    con.sql("SET threads = 2"); con.sql("SET memory_limit = '1GB'")
    con.sql("""CREATE TABLE sk AS SELECT DISTINCT sha, n_chars FROM read_parquet('data/skills/markers/*.parquet')
               WHERE sha IN (SELECT DISTINCT sha FROM read_parquet('data/reuse/skills_occurrences.parquet'))""")
    con.sql("""CREATE TABLE ct AS SELECT DISTINCT o.target, o.sha, m.n_chars
               FROM (SELECT DISTINCT target, sha FROM read_parquet('data/context_occurrences.parquet')) o
               JOIN (SELECT DISTINCT sha, n_chars FROM read_parquet('data/context/markers/*.parquet')) m USING (sha)""")
    parts = [con.sql(f"SELECT 'SKILL.md' AS artifact, {STATS} FROM sk").df(),
             con.sql(f"SELECT '(context, all)' AS artifact, {STATS} FROM (SELECT DISTINCT sha, n_chars FROM ct)").df(),
             con.sql(f"SELECT target AS artifact, {STATS} FROM ct GROUP BY target ORDER BY contents DESC").df()]
    d = pd.concat(parts, ignore_index=True)
    d.to_csv("data/reuse/file_lengths.csv", index=False)
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()
