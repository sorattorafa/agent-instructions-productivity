#!/usr/bin/env python3
"""
Summarizes the output of vocabulary.py: how many texts (and repositories)
name each metric, each dimension and each framework.

A text counts for a dimension if it matches any item that the framework
itself links to it (lexicon.item_dimensions). Coverage, not partition: a
text can count for several dimensions.

Outputs in data/vocabulary/<corpus>/:
  item_summary.csv        per item: texts, % of the corpus, repositories
  dimension_summary.csv   per dimension and per framework, same columns
  snippet_sample.csv      up to 25 randomly drawn snippets per item (fixed
                          seed), to check the precision of each pattern
  summary_by_target.csv, summary_by_target_pct.csv
                          (context only) the same counts per file type

Usage: python scripts/vocabulary_summary.py skills|context
"""
from __future__ import annotations

import json
import os
import sys

import duckdb
import pandas as pd

import lexicon
from vocabulary import CORPORA, q

SAMPLE_PER_ITEM = 25
ITEMS: list[dict] = []


def any_of(ks: list[str]) -> str:
    """OR of the columns; a dimension with no item becomes a false condition."""
    return " OR ".join(ks) if ks else "false"


def main(name: str) -> None:
    cfg = CORPORA[name]
    out = os.path.join("data/vocabulary", name)
    n_corpus = json.load(open(os.path.join(out, "totals.json")))["texts_scanned"]
    con = duckdb.connect()
    con.sql("CREATE SECRET hf (TYPE huggingface, PROVIDER credential_chain)")
    con.sql(f"CREATE TABLE m AS SELECT DISTINCT * FROM read_parquet({q(out + '/partials/*.parquet')})")
    global ITEMS
    ITEMS = lexicon.items(name)
    keys = [i["key"] for i in ITEMS]

    # repositories: all occurrences of the texts found, including copies
    if name == "skills":
        source = f"hf://datasets/{cfg['repo']}/{cfg['prefix']}*.parquet"
        con.sql(f"""CREATE TABLE r AS SELECT DISTINCT file_sha AS sha, repo_full_name AS repo
                    FROM read_parquet({q(source)}) WHERE file_sha IN (SELECT sha FROM m)""")
    else:
        # local, deduplicated copy written by framework_search.py
        source = "data/collection/occurrences.parquet"
        con.sql(f"""CREATE TABLE r AS SELECT DISTINCT sha, repo
                    FROM read_parquet({q(source)}) WHERE sha IN (SELECT sha FROM m)""")

    def count_where(cond: str) -> tuple[int, int]:
        t = con.sql(f"SELECT count(*) FROM m WHERE {cond}").fetchall()[0][0]
        rp = con.sql(f"SELECT count(DISTINCT repo) FROM r WHERE sha IN "
                     f"(SELECT sha FROM m WHERE {cond})").fetchall()[0][0]
        return t, rp

    rows = []
    for it in ITEMS:
        t, rp = count_where(it["key"])
        rows.append({"item": it["key"],
                       "dimensions": "; ".join(f"{f}/{d}" for f, d in lexicon.item_dimensions(it)),
                       "texts": t, "pct": 100 * t / n_corpus, "repos": rp})
    pd.DataFrame(rows).to_csv(os.path.join(out, "item_summary.csv"), index=False)

    dims = []
    for fw, dim in lexicon.DIMENSIONS:
        ks = [i["key"] for i in ITEMS if (fw, dim) in lexicon.item_dimensions(i)]
        t, rp = count_where(any_of(ks))
        dims.append({"framework": fw, "dimension": dim, "items": len(ks),
                     "texts": t, "pct": 100 * t / n_corpus, "repos": rp})
    for fw in lexicon.FRAMEWORKS:
        ks = [i["key"] for i in ITEMS
              if any(f == fw for f, _ in lexicon.item_dimensions(i))]
        t, rp = count_where(any_of(ks))
        dims.append({"framework": fw, "dimension": "(any)", "items": len(ks),
                     "texts": t, "pct": 100 * t / n_corpus, "repos": rp})
    t, rp = count_where(any_of(keys))
    dims.append({"framework": "(any)", "dimension": "(any)", "items": len(keys),
                 "texts": t, "pct": 100 * t / n_corpus, "repos": rp})
    d = pd.DataFrame(dims)
    d.to_csv(os.path.join(out, "dimension_summary.csv"), index=False)

    con.sql("SELECT setseed(0.42)")
    parts = [f"""(SELECT '{k}' AS item, sha, snippet_{k} AS snippet FROM m WHERE {k}
                   ORDER BY random() LIMIT {SAMPLE_PER_ITEM})""" for k in keys]
    con.sql(" UNION ALL ".join(parts)).df().to_csv(
        os.path.join(out, "snippet_sample.csv"), index=False)

    if name == "context":
        by_target(con, out)

    print(f"corpus {name}: {n_corpus:,} texts")
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print(d.to_string(index=False))


def by_target(con, out: str) -> None:
    """Productivity analysis: the same count split by file type (CLAUDE.md, AGENTS.md,
    .cursor/rules...). Denominator = distinct texts of the target, taken from
    the deduplicated occurrences that framework_search.py already wrote."""
    oc = "data/collection/occurrences.parquet"
    con.sql(f"CREATE TABLE oa AS SELECT DISTINCT target, sha FROM read_parquet({q(oc)})")
    groups = [(f"{fw}/{dim}", [i["key"] for i in ITEMS
                               if (fw, dim) in lexicon.item_dimensions(i)])
              for fw, dim in lexicon.DIMENSIONS]
    groups += [(f"{fw}/(any)", [i["key"] for i in ITEMS
                                     if any(f == fw for f, _ in lexicon.item_dimensions(i))])
               for fw in lexicon.FRAMEWORKS]
    groups.append(("(any)", [i["key"] for i in ITEMS]))
    cols = ", ".join(f"count(DISTINCT CASE WHEN {any_of(ks)} THEN m.sha END) AS \"{g}\""
                     for g, ks in groups)
    t = con.sql(f"""SELECT oa.target, count(DISTINCT oa.sha) AS texts, {cols}
                    FROM oa LEFT JOIN m USING (sha) GROUP BY oa.target ORDER BY texts DESC""").df()
    t.to_csv(os.path.join(out, "summary_by_target.csv"), index=False)
    pct = t.copy()
    for g, _ in groups:
        pct[g] = (100 * t[g] / t["texts"]).round(3)
    pct.to_csv(os.path.join(out, "summary_by_target_pct.csv"), index=False)
    print(pct[["target", "texts"] + [g for g, _ in groups[-4:]]].to_string(index=False))
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in CORPORA:
        sys.exit(__doc__)
    main(sys.argv[1])
