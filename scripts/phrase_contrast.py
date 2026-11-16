#!/usr/bin/env python3
"""
What each artifact talks about, by direct count: the two- and three-word
phrases most characteristic of skills and of context files.

Uses the documents of the topic model (assignment.parquet: same cleaning,
language filter, near-duplicate removal, cap per owner and balance), and
counts each phrase once per document.

Selection rule, the same for both artifacts:
  p = share of the artifact's documents that contain the phrase, q = the
  share in the other artifact; score = p * log(p / q), with p >= MIN_P;
  phrases that start or end with a generic word are left out, and so are
  phrases that are part of the header written by Claude Code's /init command
  (tool text, reported apart) and phrases that contain or are contained in
  one already chosen. The TOP highest scores are kept.

The phrases are chosen on that sample, where memory allows the full list of
n-grams and the cleaning keeps one account or one template from deciding the
ranking. They are then counted over EVERY distinct content of both datasets
(count stage): a phrase matches when its words appear in sequence, separated
only by characters that are not letters, case-insensitive, after fenced code
blocks are removed. Those counts are exact.

Outputs in data/topics/: phrase_contrast.csv (artifact, rank, phrase, share in
skills and in context files, in the sample), phrase_contrast.json (number of
documents and share of context documents with the /init header) and
phrase_counts.csv (the same phrases with texts and share over all distinct
contents).
Usage: python scripts/phrase_contrast.py          selection on the sample
       python scripts/phrase_contrast.py count    counts over both datasets
(run from the repository root; the count resumes if interrupted)
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import phrases as P

OUT = os.path.join("data", "topics")
MIN_P = 0.02
TOP = 8
INIT_HEADER = ("this file provides guidance to claude code claude ai code when working with code in this "
               "repository project overview")


def main() -> None:
    d = P.load()
    vec = CountVectorizer(ngram_range=(2, 3), min_df=P.MIN_DOCS, binary=True,
                          token_pattern=r"(?u)\b[a-z][a-z-]+\b", dtype=np.int32)
    X = vec.fit_transform(d["clean"]).tocsc()
    terms = vec.get_feature_names_out()
    ok = np.array([t.split()[0] not in P.GENERIC_WORDS and t.split()[-1] not in P.GENERIC_WORDS
                   and t not in INIT_HEADER for t in terms])
    X, terms = X[:, ok], terms[ok]
    sk = (d.artifact == "skill").to_numpy()
    share = {"skill": np.asarray(X[sk].sum(axis=0)).ravel() / sk.sum(),
             "context": np.asarray(X[~sk].sum(axis=0)).ravel() / (~sk).sum()}
    rows = []
    for art, other in (("skill", "context"), ("context", "skill")):
        p, q = share[art], share[other]
        score = np.where(p >= MIN_P, p * np.log((p + 1e-9) / (q + 1e-9)), -np.inf)
        chosen: list[str] = []
        for j in np.argsort(-score):
            if len(chosen) == TOP or not np.isfinite(score[j]):
                break
            if any(terms[j] in c or c in terms[j] for c in chosen):
                continue
            chosen.append(terms[j])
            rows.append({"artifact": art, "rank": len(chosen), "phrase": terms[j],
                         "pct_skills": round(100 * share["skill"][j], 1),
                         "pct_context": round(100 * share["context"][j], 1)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "phrase_contrast.csv"), index=False)
    init = d.loc[~sk, "clean"].str.contains("this file provides guidance to claude code", regex=False).mean()
    json.dump({"documents_per_artifact": int(sk.sum()), "pct_context_with_init_header": round(100 * float(init), 1)},
              open(os.path.join(OUT, "phrase_contrast.json"), "w"), indent=1)
    print(out.to_string(index=False))
    print(f"/init header in {100 * init:.1f}% of the context documents")


def count() -> None:
    """Counts the chosen phrases over every distinct content of both datasets."""
    import time

    import duckdb
    from huggingface_hub import HfApi

    from productivity import OCCURRENCES
    from vocabulary import CORPORA, q

    ph = pd.read_csv(os.path.join(OUT, "phrase_contrast.csv"))
    cols = {f"p{i}": "(?i)\\b" + "[^a-zA-Z]+".join(p.split()) + "\\b" for i, p in enumerate(ph.phrase)}
    flags = ", ".join(f"regexp_matches(t, {q(rx)}) AS {c}" for c, rx in cols.items())
    sums = ", ".join(f"sum({c}::int) AS {c}" for c in cols)
    con = duckdb.connect()
    con.sql("CREATE SECRET hf (TYPE huggingface, PROVIDER credential_chain)")
    con.sql("SET threads = 2"); con.sql("SET memory_limit = '4GB'")
    con.sql("SET preserve_insertion_order = false")
    part = os.path.join(OUT, "phrase_counts_parts.json")
    done = json.load(open(part)) if os.path.exists(part) else {}
    totals = {}
    for name in ("skills", "context"):
        cfg = CORPORA[name]
        if name == "context" and os.path.isdir(os.path.join("release", "texts")):
            # the released texts are the distinct contents with a valid occurrence, already on disk
            groups = [[f] for f in sorted(__import__("glob").glob("release/texts/part-*.parquet"))]
            sel = "SELECT sha, regexp_replace(text, '(?s)```.*?```', ' ', 'g') AS t FROM read_parquet({files})"
        else:
            files = sorted(f for f in HfApi().list_repo_files(cfg["repo"], repo_type="dataset")
                           if f.startswith(cfg["prefix"]) and f.endswith(".parquet"))
            groups = [[f"hf://datasets/{cfg['repo']}/{f}" for f in files[k:k + cfg["batch"]]]
                      for k in range(0, len(files), cfg["batch"])]
            sel = (f"SELECT {cfg['sha']} AS sha, regexp_replace({cfg['text']}, '(?s)```.*?```', ' ', 'g') AS t "
                   f"FROM read_parquet({{files}}) WHERE {cfg['filter']}")
        con.sql(f"CREATE OR REPLACE TABLE oc AS SELECT DISTINCT sha FROM read_parquet({q(OCCURRENCES[name])})")
        for g in groups:
            key = f"{name}:{g[0]}"
            if key in done:
                continue
            t0 = time.time()
            src = "[" + ",".join(q(f) for f in g) + "]"
            r = con.sql(f"""WITH a AS ({sel.format(files=src)}), f AS (SELECT {flags} FROM a
                            WHERE sha IN (SELECT sha FROM oc))
                            SELECT count(*) AS n, {sums} FROM f""").df().iloc[0]
            done[key] = {k: int(v or 0) for k, v in r.items()}
            json.dump(done, open(part, "w"))
            print(f"{time.strftime('%H:%M:%S')} {name} {len([k for k in done if k.startswith(name)])}/{len(groups)}, "
                  f"{time.time() - t0:.0f}s", flush=True)
        totals[name] = pd.DataFrame([v for k, v in done.items() if k.startswith(name + ":")]).sum()
    rows = []
    for i, r in ph.iterrows():
        rows.append({"artifact": r.artifact, "rank": r["rank"], "phrase": r.phrase,
                     "skills": int(totals["skills"][f"p{i}"]), "pct_skills": round(100 * totals["skills"][f"p{i}"] / totals["skills"]["n"], 2),
                     "context": int(totals["context"][f"p{i}"]), "pct_context": round(100 * totals["context"][f"p{i}"] / totals["context"]["n"], 2),
                     "sample_pct_skills": r.pct_skills, "sample_pct_context": r.pct_context})
    out = pd.DataFrame(rows)
    out.attrs = {}
    out.to_csv(os.path.join(OUT, "phrase_counts.csv"), index=False)
    json.dump({"skills": int(totals["skills"]["n"]), "context": int(totals["context"]["n"])},
              open(os.path.join(OUT, "phrase_counts_totals.json"), "w"))
    print(out.to_string(index=False))
    print("DONE", flush=True)


if __name__ == "__main__":
    import sys
    count() if sys.argv[1:] == ["count"] else main()
