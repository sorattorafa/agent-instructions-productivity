#!/usr/bin/env python3
"""
Dataset characterization (replication): how much of the content is a copy, and which contents are the most
copied.

Measures, for the skills (GitSkills) and for each type of context file (our
own collection), over ALL occurrences:
  occurrences, distinct contents, repositories;
  pct_redundant  (occurrences - distinct) / occurrences: the definition used
               in the GitSkills paper (50.5% for the skills); serves as a
               check that the computation is right;
  pct_copies   fraction of the occurrences whose content appears in >= 2
               occurrences (includes the first instance of each content);
  pct_unique   fraction of the distinct contents that appear only once;
  top10        fraction of the occurrences covered by the 10 most copied
               contents.

It also lists the most copied contents (by number of distinct repositories),
with the opening of the text, so the origin can be classified by hand
(tool-generated, scaffold template, pointer, community template...).

Inputs: remote skills (GitSkills, `mvaccargiu/gitskills`, artifacts); local
context (data/context_occurrences.parquet, deduplicated by
framework_search.py and filtered by name_filter.py) and remote texts (the
context-file dataset, `CONTEXT_DATASET_REPO`, textos/).

Outputs in data/reuse/: summary.csv, top_contents.csv, skills_occurrences.parquet
Usage: python scripts/reuse.py   (run from the repository root)
"""
from __future__ import annotations

import os

import duckdb
import pandas as pd

from vocabulary import CONTEXT_REPO, SKILLS_REPO, q

OUT = os.path.join("data", "reuse")
SK = f"hf://datasets/{SKILLS_REPO}/data/artifacts/*.parquet"
CT = f"hf://datasets/{CONTEXT_REPO}/textos/*.parquet"
TOP_SKILLS = 30
TOP_CONTEXT_OVERALL = 30
TOP_PER_TARGET = 5


def summary(con, table: str, group: str) -> pd.DataFrame:
    return con.sql(f"""
      WITH c AS (SELECT {group} AS artifact, sha, count(*) AS n,
                        count(DISTINCT repo) AS nr FROM {table} GROUP BY ALL),
           r AS (SELECT artifact, sha, n, row_number() OVER (PARTITION BY artifact ORDER BY nr DESC, n DESC) AS pos FROM c),
           rp AS (SELECT {group} AS artifact, count(DISTINCT repo) AS repos FROM {table} GROUP BY ALL)
      SELECT artifact,
             sum(n) AS occurrences,
             count(*) AS distinct,
             any_value(rp.repos) AS repos,
             round(100.0 * (sum(n) - count(*)) / sum(n), 1) AS pct_redundant,
             round(100.0 * sum(CASE WHEN n >= 2 THEN n ELSE 0 END) / sum(n), 1) AS pct_copies,
             round(100.0 * sum(CASE WHEN n = 1 THEN 1 ELSE 0 END) / count(*), 1) AS pct_unique,
             round(100.0 * sum(CASE WHEN pos <= 10 THEN n ELSE 0 END) / sum(n), 2) AS top10
      FROM r JOIN rp USING (artifact) GROUP BY artifact ORDER BY occurrences DESC""").df()


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    con = duckdb.connect()
    con.sql("SET memory_limit = '2GB'"); con.sql("SET threads = 4")
    con.sql("CREATE SECRET hf (TYPE huggingface, PROVIDER credential_chain)")

    oc_sk = os.path.join(OUT, "skills_occurrences.parquet")
    if not os.path.exists(oc_sk):
        con.sql(f"""COPY (SELECT repo_full_name AS repo, path AS path, file_sha AS sha
                         FROM read_parquet({q(SK)})) TO {q(oc_sk)} (FORMAT parquet, COMPRESSION zstd)""")
    con.sql(f"CREATE VIEW sk AS SELECT 'SKILL.md' AS target, * FROM read_parquet({q(oc_sk)})")
    con.sql("CREATE VIEW ct AS SELECT * FROM read_parquet('data/context_occurrences.parquet')")

    parts = [summary(con, "sk", "target"),
              summary(con, "ct", "'(context, all)'"),
              summary(con, "ct", "target")]
    res = pd.concat(parts, ignore_index=True)
    res.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print(res.to_string(index=False))

    # most copied contents
    top_sk = con.sql(f"""SELECT 'SKILL.md' AS artifact, sha, count(DISTINCT repo) AS repos, count(*) AS occurrences
                         FROM sk GROUP BY sha ORDER BY repos DESC, occurrences DESC LIMIT {TOP_SKILLS}""").df()
    top_ct = con.sql(f"""SELECT '(context, all)' AS artifact, sha, count(DISTINCT repo) AS repos, count(*) AS occurrences
                         FROM ct GROUP BY sha ORDER BY repos DESC, occurrences DESC LIMIT {TOP_CONTEXT_OVERALL}""").df()
    top_target = con.sql(f"""SELECT target AS artifact, sha, repos, occurrences FROM (
                             SELECT target, sha, count(DISTINCT repo) AS repos, count(*) AS occurrences,
                                    row_number() OVER (PARTITION BY target ORDER BY count(DISTINCT repo) DESC, count(*) DESC) AS pos
                             FROM ct GROUP BY target, sha) WHERE pos <= {TOP_PER_TARGET}""").df()
    top = pd.concat([top_sk, top_ct, top_target], ignore_index=True)

    # texts: skills from the representative; context from the remote texts
    # ("textos/" and "texto" are field names of the dataset as collected)
    list_sk = "[" + ",".join(q(s) for s in top_sk.sha) + "]"
    tx_sk = con.sql(f"""SELECT file_sha AS sha, name AS skill_name, left(content, 400) AS opening, length(content) AS n_chars
                        FROM read_parquet({q(SK)}) WHERE dedup_primary = 1 AND list_contains({list_sk}, file_sha)""").df()
    shas_ct = sorted(set(top_ct.sha) | set(top_target.sha))
    list_ct = "[" + ",".join(q(s) for s in shas_ct) + "]"
    tx_ct = con.sql(f"""SELECT sha, NULL AS skill_name, left(texto, 400) AS opening, length(texto) AS n_chars
                        FROM read_parquet({q(CT)}) WHERE list_contains({list_ct}, sha)""").df()
    tx = pd.concat([tx_sk, tx_ct], ignore_index=True).drop_duplicates("sha")
    top = top.merge(tx, on="sha", how="left")
    top["opening"] = top["opening"].fillna("").str.replace(r"\s+", " ", regex=True)
    top.to_csv(os.path.join(OUT, "top_contents.csv"), index=False)
    print(f"\n{len(top)} rows in {OUT}/top_contents.csv")


if __name__ == "__main__":
    main()
