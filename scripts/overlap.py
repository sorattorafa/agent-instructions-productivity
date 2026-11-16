#!/usr/bin/env python3
"""
Repositories that appear in both datasets: GitSkills (skills) and the
context-file dataset.

The two datasets were collected at different times (GitSkills in July 2026,
context files in November 2026), so a repository that added one of the
artifacts in between is counted as it stood at each collection.

Outputs in data/overlap/:
  overlap.csv          repositories with skills, with context files, with
                       both; shares in each direction
  overlap_by_target.csv  per context file type: repositories with that type
                       and how many of them also have skills
  overlap_naming.csv   repositories with both artifacts, by where a framework
                       or metric is named (skills, context files, both, none)
Usage: python scripts/overlap.py   (run from the repository root)
"""
from __future__ import annotations

import os

import duckdb
import pandas as pd

OUT = os.path.join("data", "overlap")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    con = duckdb.connect()
    con.sql("SET threads = 2"); con.sql("SET memory_limit = '1500MB'")
    con.sql("SET preserve_insertion_order = false")
    con.sql("""CREATE TABLE sk AS SELECT DISTINCT lower(repo) AS repo, sha
               FROM read_parquet('data/reuse/skills_occurrences.parquet')""")
    con.sql("""CREATE TABLE ct AS SELECT DISTINCT lower(repo) AS repo, target, sha
               FROM read_parquet('data/context_occurrences.parquet')""")
    con.sql("CREATE TABLE rs AS SELECT DISTINCT repo FROM sk")
    con.sql("CREATE TABLE rc AS SELECT DISTINCT repo FROM ct")
    con.sql("CREATE TABLE rb AS SELECT repo FROM rs INTERSECT SELECT repo FROM rc")
    one = lambda sql: con.sql(sql).fetchall()[0][0]
    n_s, n_c, n_b = one("SELECT count(*) FROM rs"), one("SELECT count(*) FROM rc"), one("SELECT count(*) FROM rb")
    rows = [{"measure": "repositories with skills", "repos": n_s},
            {"measure": "repositories with context files", "repos": n_c},
            {"measure": "repositories with both", "repos": n_b},
            {"measure": "repositories with either", "repos": n_s + n_c - n_b},
            {"measure": "pct of skill repositories that also have a context file", "repos": round(100 * n_b / n_s, 2)},
            {"measure": "pct of context-file repositories that also have a skill", "repos": round(100 * n_b / n_c, 2)}]
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "overlap.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    t = con.sql("""SELECT target, count(DISTINCT repo) AS repos,
                          count(DISTINCT CASE WHEN repo IN (SELECT repo FROM rs) THEN repo END) AS repos_with_skills
                   FROM ct GROUP BY target ORDER BY repos DESC""").df()
    t["pct_with_skills"] = (100 * t.repos_with_skills / t.repos).round(2)
    t.to_csv(os.path.join(OUT, "overlap_by_target.csv"), index=False)
    print(t.to_string(index=False))

    # where a framework or metric is named, in repositories that have both artifacts
    con.sql("""CREATE TABLE ns AS SELECT DISTINCT repo FROM sk
               WHERE sha IN (SELECT sha FROM read_parquet('data/productivity/shas_any_skills.parquet'))""")
    con.sql("""CREATE TABLE nc AS SELECT DISTINCT repo FROM ct
               WHERE sha IN (SELECT sha FROM read_parquet('data/productivity/shas_any_context.parquet'))""")
    n = con.sql("""SELECT count(*) AS repos_both,
                          sum((repo IN (SELECT repo FROM ns) AND repo NOT IN (SELECT repo FROM nc))::int) AS skills_only,
                          sum((repo NOT IN (SELECT repo FROM ns) AND repo IN (SELECT repo FROM nc))::int) AS context_only,
                          sum((repo IN (SELECT repo FROM ns) AND repo IN (SELECT repo FROM nc))::int) AS in_both,
                          sum((repo NOT IN (SELECT repo FROM ns) AND repo NOT IN (SELECT repo FROM nc))::int) AS nowhere
                   FROM rb""").df()
    n["repos_naming_in_skills_all"] = one("SELECT count(*) FROM ns")
    n["repos_naming_in_context_all"] = one("SELECT count(*) FROM nc")
    n.to_csv(os.path.join(OUT, "overlap_naming.csv"), index=False)
    print(n.T.to_string(header=False))


if __name__ == "__main__":
    main()
