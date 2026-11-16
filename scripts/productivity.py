#!/usr/bin/env python3
"""
Productivity analysis: do skills and context files mention productivity measurement?

Combines two measures:
  mention  the text names the framework (corpus_pass.py: dora, space, devex,
           dx_core4) or the phrase developer/engineering productivity;
  metric   the text names a metric of the framework (vocabulary.py +
           lexicon.py, only the items kept for each corpus:
           lexicon.items).

Per corpus: distinct texts, % of the corpus and repositories for each framework
(mention, metric, either of the two), each dimension (metric only) and, for
context files, each file type. It also draws snippets of the explicit mentions
at random, to read what each pattern matches (25 per pattern in skills, 10 in
context files).

Inputs (run from the repository root): data/<c>/mentions, data/<c>/totals.json,
data/vocabulary/<c>/partials, data/reuse/skills_occurrences.parquet,
data/context_occurrences.parquet.
Outputs in data/productivity/.
Usage: python scripts/productivity.py
"""
from __future__ import annotations

import json
import os

import duckdb
import pandas as pd

import lexicon
from corpus_pass import MENTIONS
from vocabulary import q

OUT = os.path.join("data", "productivity")
SAMPLE_SIZE = {"skills": 25, "context": 10}
OCCURRENCES = {"skills": "data/reuse/skills_occurrences.parquet",
         "context": "data/context_occurrences.parquet"}
# framework of each mention pattern; dx_core4 is a framework of its own (DX)
# that combines DORA, SPACE and DevEx, and is kept separate so that it does not
# inflate any of the three
FRAMEWORK_OF = {"dora": "DORA", "space": "SPACE", "devex": "DevEx"}
STRICT = {
    "devex": r"(?i:(?:^|[^-\w])devex[ -]?(?:framework|metrics?|survey)\b)"
             r"|(?i:feedback loops,? cognitive load,? and flow state)",
    "space": r"\bSPACE[ -](?:framework|metrics|dimensions)\b"
             r"|(?i:\bspace (?:framework|metrics) (?:for|of) (?:developer|engineering) productivity\b)"
             r"|(?i:satisfaction and well-?being, performance, activity)",
}

# A value or target stated right after the metric name (same list item or
# table row, up to 80 characters): a comparison with a number, a percentage, a
# duration, or a word such as target, baseline, benchmark, elite. Applied to
# the stored snippet of the first occurrence, so it undercounts.
VALUE = (r"(?:[<>≤≥]=?\s*\d|\d+(?:\.\d+)?\s*%"
         r"|\b\d+(?:\.\d+)?\s*(?:x\b|hours?\b|hrs?\b|h\b|days?\b|d\b|minutes?\b|mins?\b|weeks?\b|per (?:day|week|month)\b)"
         r"|\b(?:target|baseline|benchmark|threshold|goal|elite|slo)s?\b)")

MENTION_EXCLUDED = {
    "context": {"devex": "in context files the matches are mostly a package or project name, API or PR metrics"},
}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rows, target_rows, samples = [], [], []
    for corpus in ("skills", "context"):
        con = duckdb.connect()
        con.sql(f"CREATE TABLE men AS SELECT DISTINCT * FROM read_parquet({q(f'data/{corpus}/mentions/*.parquet')})")
        items = lexicon.items(corpus)
        ks = [i["key"] for i in items]
        con.sql(f"""CREATE TABLE met AS SELECT DISTINCT sha, {', '.join(ks)}
                    FROM read_parquet({q(f'data/vocabulary/{corpus}/partials/*.parquet')})
                    WHERE {' OR '.join(ks)}""")
        con.sql(f"CREATE TABLE oc AS SELECT DISTINCT target, sha, repo FROM read_parquet({q(OCCURRENCES[corpus])})"
                if corpus == "context" else
                f"CREATE TABLE oc AS SELECT DISTINCT 'SKILL.md' AS target, sha, repo FROM read_parquet({q(OCCURRENCES[corpus])})")
        # only texts with at least one valid occurrence count (name_filter.py);
        # the denominator is the number of distinct texts of those occurrences
        con.sql("DELETE FROM men WHERE sha NOT IN (SELECT sha FROM oc)")
        # reading of the matches (13 Nov 2026): the devex and space patterns were too wide.
        # Reapplies narrower patterns to the stored snippet (300 characters
        # around the first occurrence); if the first one fails, the text is
        # dropped (conservative).
        #   devex: drops "devex dimensions" (generic use) and the gstack command
        #          plan-devex-framework; space: drops "SPACE model" (NAPO).
        for p_, rx in STRICT.items():
            con.sql(f"UPDATE men SET {p_} = regexp_matches(snippet_{p_}, {q(rx)}) WHERE {p_}")
        # second reading of the matches (15 Nov 2026):
        # a mention pattern that does not pass in a corpus stops counting in it
        for p_ in MENTION_EXCLUDED.get(corpus, {}):
            con.sql(f"UPDATE men SET {p_} = false")
        con.sql("DELETE FROM met WHERE sha NOT IN (SELECT sha FROM oc)")
        with_value = " OR ".join(
            f"({i['key']} AND regexp_matches(snippet_{i['key']}, {q('(?is)\\b(?:' + i['pattern'] + ')\\b.{0,80}?' + VALUE)}))"
            for i in items)
        con.sql(f"""CREATE TABLE metv AS SELECT DISTINCT sha
                    FROM read_parquet({q(f'data/vocabulary/{corpus}/partials/*.parquet')})
                    WHERE ({with_value}) AND sha IN (SELECT sha FROM oc)""")
        n_corpus = con.sql("SELECT count(DISTINCT sha) FROM oc").fetchall()[0][0]

        def make_set(name: str, sql: str) -> None:
            con.sql(f"CREATE OR REPLACE TABLE s_{name} AS {sql}")

        groups = []  # (label, framework, dimension, measure, sql that yields shas)
        for p in MENTIONS:
            groups.append((p, FRAMEWORK_OF.get(p, p), "", "mention", f"SELECT sha FROM men WHERE {p}"))
        for fw in lexicon.FRAMEWORKS:
            km = [i["key"] for i in items if any(f == fw for f, _ in lexicon.item_dimensions(i))]
            sql_met = f"SELECT sha FROM met WHERE {' OR '.join(km)}" if km else "SELECT NULL AS sha WHERE false"
            p = {v: k for k, v in FRAMEWORK_OF.items()}[fw]
            groups.append((f"{fw}/metric", fw, "(any)", "metric", sql_met))
            groups.append((f"{fw}/any", fw, "(any)", "mention or metric",
                           f"SELECT sha FROM men WHERE {p} UNION {sql_met}"))
            for f2, dim in lexicon.DIMENSIONS:
                if f2 != fw:
                    continue
                kd = [i["key"] for i in items if (fw, dim) in lexicon.item_dimensions(i)]
                sql = f"SELECT sha FROM met WHERE {' OR '.join(kd)}" if kd else "SELECT NULL AS sha WHERE false"
                groups.append((f"{fw}/{dim}", fw, dim, "metric" if kd else "not measured", sql))
        three = " UNION ".join(f"SELECT sha FROM men WHERE {p}" for p in FRAMEWORK_OF)
        groups.append(("any framework/any", "(any)", "(any)", "mention or metric",
                       f"{three} UNION SELECT sha FROM met"))
        # how the texts that count split: name a framework only, name a metric
        # only, or both (the three add up to "any framework/any")
        groups.append(("named only", "(any)", "(any)", "mention only",
                       f"({three}) EXCEPT SELECT sha FROM met"))
        groups.append(("metric only", "(any)", "(any)", "metric only",
                       f"SELECT sha FROM met EXCEPT ({three})"))
        groups.append(("named and metric", "(any)", "(any)", "mention and metric",
                       f"({three}) INTERSECT SELECT sha FROM met"))
        # metric named with a value or target next to it, among all texts that name a metric
        groups.append(("any metric", "(any)", "(any)", "metric", "SELECT sha FROM met"))
        groups.append(("metric with value", "(any)", "(any)", "metric with value", "SELECT sha FROM metv"))

        for rot, fw, dim, measure, sql in groups:
            make_set("x", sql)
            t = con.sql("SELECT count(DISTINCT sha) FROM s_x WHERE sha IS NOT NULL").fetchall()[0][0]
            r = con.sql("SELECT count(DISTINCT repo) FROM oc WHERE sha IN (SELECT sha FROM s_x)").fetchall()[0][0]
            # file occurrences of these contents (copies included), for the group
            # of all texts that count: how much of the mentions is itself copied
            occ = None
            if rot == "any framework/any":
                occ = con.sql(f"""SELECT count(*) FROM read_parquet({q(OCCURRENCES[corpus])})
                                  WHERE sha IN (SELECT sha FROM s_x)""").fetchall()[0][0]
            rows.append({"corpus": corpus, "group": rot, "framework": fw, "dimension": dim, "measure": measure,
                           "texts": t, "pct": round(100 * t / n_corpus, 4), "repos": r, "n_corpus": n_corpus,
                           "occurrences": occ})
            if rot == "any framework/any":
                # list of the texts that count, for languages.py
                con.sql(f"""COPY (SELECT DISTINCT sha FROM s_x WHERE sha IS NOT NULL ORDER BY sha)
                           TO {q(os.path.join(OUT, f'shas_any_{corpus}.parquet'))} (FORMAT parquet)""")
            if corpus == "context" and dim in ("", "(any)"):
                per_target = con.sql("""SELECT target, count(DISTINCT sha) AS texts FROM oc
                                 WHERE sha IN (SELECT sha FROM s_x) GROUP BY target""").df()
                per_target["group"] = rot
                target_rows.append(per_target)

        con.sql("SELECT setseed(0.42)")
        for p in MENTIONS:
            a = con.sql(f"""SELECT '{corpus}' AS corpus, '{p}' AS pattern, sha, snippet_{p} AS snippet
                            FROM men WHERE {p} ORDER BY random() LIMIT {SAMPLE_SIZE[corpus]}""").df()
            samples.append(a)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "productivity.csv"), index=False)
    print(res.to_string(index=False))
    if target_rows:
        tot = duckdb.sql("""SELECT target, count(DISTINCT sha) AS total
                            FROM read_parquet('data/context_occurrences.parquet') GROUP BY target""").df()
        pa = pd.concat(target_rows).merge(tot, on="target")
        pa["pct"] = (100 * pa["texts"] / pa["total"]).round(4)
        pa.pivot(index=["target", "total"], columns="group", values="pct").fillna(0).reset_index() \
          .sort_values("total", ascending=False).to_csv(os.path.join(OUT, "productivity_by_target.csv"), index=False)
    sample_df = pd.concat(samples, ignore_index=True)
    sample_df["snippet"] = sample_df["snippet"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    # snippets of the named mentions, drawn at random, to read what each pattern matches
    sample_df.to_csv(os.path.join(OUT, "mention_sample.csv"), index=False)
    print(f"\nmention sample: {len(sample_df)} snippets")

if __name__ == "__main__":
    main()
