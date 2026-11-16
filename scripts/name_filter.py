#!/usr/bin/env python3
"""
Filters the context occurrences by the exact file name.

The `filename:` qualifier of GitHub code search matches suffixes and
prefixes: the target AGENTS.md brought in agents.mdx, subagents.md,
AGENTS.md.template etc. (~10% of the AGENTS.md occurrences, ~4% of CLAUDE.md,
~24% of GEMINI.md). Rule:
  target that is a file name -> the file name (without the folder) must be
                                equal to the target, case-insensitively;
  target that is a folder    -> the path must contain the folder (already
                                true of all occurrences; checked on
                                13 Nov 2026);
  any target                 -> dropped if the path has a `skills/` folder:
                                the file is part of a vendored skill
                                (e.g. skills/react-best-practices/AGENTS.md),
                                not repository context (decision of
                                13 Nov 2026).
Input:  data/collection/occurrences.parquet
Output: data/context_occurrences.parquet and data/name_filter.csv
Usage: python scripts/name_filter.py  (run from the repository root)
"""
import os

import duckdb

SRC = "data/collection/occurrences.parquet"
DST = "data/context_occurrences.parquet"
FOLDERS = (".claude/agents", ".claude/commands", ".cursor/rules", ".claude/rules",
          ".github/instructions", "memory-bank")

COND = f"""CASE WHEN target IN ({",".join(repr(p) for p in FOLDERS)})
              THEN contains(lower('/' || path), lower('/' || target || '/'))
              ELSE lower(regexp_extract(path, '[^/]+$')) = lower(regexp_extract(target, '[^/]+$')) END
           AND NOT regexp_matches(lower(path), '(^|/)skills/')"""


def main() -> None:
    os.makedirs("data", exist_ok=True)
    con = duckdb.connect()
    con.sql("SET memory_limit = '1500MB'")
    con.sql(f"COPY (SELECT * FROM read_parquet('{SRC}') WHERE {COND}) TO '{DST}' (FORMAT parquet, COMPRESSION zstd)")
    r = con.sql(f"""SELECT a.target, a.n AS before, coalesce(b.n, 0) AS after, a.n - coalesce(b.n, 0) AS removed
                    FROM (SELECT target, count(*) n FROM read_parquet('{SRC}') GROUP BY 1) a
                    LEFT JOIN (SELECT target, count(*) n FROM read_parquet('{DST}') GROUP BY 1) b USING (target)
                    ORDER BY before DESC""").df()
    r.to_csv("data/name_filter.csv", index=False)
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
