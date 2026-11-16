#!/usr/bin/env python3
"""
Validation of the topic model (Dataset characterization (topics)) and of the typical phrases on texts the
model has not seen (heldout_sample.py: disjoint SHA prefixes, ~4x larger).

1. Redoes the fit of topics.py on the same documents (assignment.parquet),
   with the same parameters, and checks that it reproduces the saved
   assignment.
2. Applies to the held-out sample the same filters (length, language, near
   duplicates, cap per owner) and also removes any text whose beginning
   matches a fit document. No balancing: the % of each artifact is already
   normalized by the artifact itself.
3. Projects each held-out text onto the fitted topics (NMF.transform) and
   compares: % of skills and of context files per topic, % per context file
   type, typical phrases per topic (same rule as phrases.py, with the phrase
   vocabulary of the fit) and the coverage test of the 100 most frequent
   phrases.

Outputs in data/topics/heldout/:
  topic_comparison.csv        per topic: % fit vs % held-out, per artifact
  topics_by_target.csv        per context file type (held-out)
  phrases_by_topic.csv        top 30 per topic in the held-out sample + whether it was in the fit top 30
  coverage_top100.csv         coverage test on the held-out sample
  summary.json                sizes, filters, reproduction, agreements
Does not change any number in the paper.
Usage: python scripts/heldout_validation.py   (run from the repository root)
"""
from __future__ import annotations

import glob
import json
import os

import duckdb
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer

import phrases as E
import topics as T

OUT = os.path.join("data", "topics", "heldout")
ART = {"skills": "skill", "context": "context"}


def typical_phrases(cont: np.ndarray, n_topic: np.ndarray, terms: np.ndarray, top: int) -> pd.DataFrame:
    k, n = len(n_topic), n_topic.sum()
    total = cont.sum(axis=0)
    rows = []
    for i in range(k):
        p_in = cont[i] / max(1, n_topic[i])
        p_out = (total - cont[i]) / max(1, n - n_topic[i])
        score = np.where(p_in >= E.MIN_PIN, p_in * np.log((p_in + 1e-9) / (p_out + 1e-9)), -np.inf)
        for r, j in enumerate(np.argsort(-score)[:top]):
            rows.append({"topic": i, "position": r + 1, "phrase": terms[j],
                           "pct_in_topic": round(100 * p_in[j], 2), "pct_outside": round(100 * p_out[j], 2)})
    return pd.DataFrame(rows)


def filters(v: pd.DataFrame) -> pd.DataFrame:
    v["ntok"], v["en"] = zip(*[T.english_stats(t) for t in v["clean"]])
    v["nascii"] = [sum(ord(c) > 127 for c in t) / max(1, len(t)) for t in v["clean"]]
    v["prefix"] = v["clean"].str[:T.PREFIX_LEN]
    return v


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    res = {}

    # 1. fit, redone
    d = E.load()
    stop_list = sorted(set(ENGLISH_STOP_WORDS) | T.NOISE)
    vec = TfidfVectorizer(stop_words=stop_list, min_df=30, max_df=0.30, max_features=50_000,
                          ngram_range=(1, 2), sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-z][a-z-]{2,}\b")
    X = vec.fit_transform(d["clean"])
    k = int(d["topic"].max()) + 1
    nmf = NMF(n_components=k, init="nndsvda", random_state=T.SEED, max_iter=400)
    W = nmf.fit_transform(X)
    res["fit_reproduction"] = round(float((W.argmax(axis=1) == d["topic"].to_numpy()).mean()), 4)
    res["transform_fit_agreement"] = round(float((nmf.transform(X).argmax(axis=1) == W.argmax(axis=1)).mean()), 4)
    print(f"[val] reproduction {res['fit_reproduction']}, transform x fit {res['transform_fit_agreement']}", flush=True)

    cv = CountVectorizer(ngram_range=(2, 3), min_df=E.MIN_DOCS, binary=True,
                         token_pattern=r"(?u)\b[a-z][a-z-]+\b", dtype=np.int32)
    cv.fit(d["clean"])
    vocab = [t for t in cv.get_feature_names_out() if not all(w in E.GENERIC_WORDS for w in t.split())]
    cv = CountVectorizer(ngram_range=(2, 3), vocabulary=vocab, binary=True,
                         token_pattern=r"(?u)\b[a-z][a-z-]+\b", dtype=np.int32)
    terms = np.array(vocab)
    fit_prefixes = set(d["clean"].str[:T.PREFIX_LEN])
    fit_pct = pd.read_csv(os.path.join("data", "topics", "topics.csv")).set_index("topic")
    fit_top = pd.read_csv(os.path.join("data", "topics", "phrases_by_topic.csv"))
    del d, X, W

    # 2. held-out, step 1: filters on metadata (without keeping the text)
    meta = []
    for corpus, art in ART.items():
        for f in sorted(glob.glob(os.path.join("data", corpus, "heldout", "*.parquet"))):
            v = pd.read_parquet(f)
            v["clean"] = [T.clean_text(t or "") for t in v["text"]]
            v = filters(v)
            v["artifact"], v["file"] = art, f
            meta.append(v[["sha", "artifact", "file", "ntok", "en", "nascii", "prefix"]])
    m = pd.concat(meta, ignore_index=True).drop_duplicates(["artifact", "sha"])
    res["raw"] = m.groupby("artifact").size().to_dict()

    con = duckdb.connect()
    con.sql("SET threads = 4"); con.sql("SET memory_limit = '2GB'")
    con.register("vs", m[m.artifact == "skill"][["sha"]])
    con.register("vc", m[m.artifact == "context"][["sha"]])
    owner_sk = con.sql("""SELECT sha, min(split_part(repo, '/', 1)) AS owner
                         FROM read_parquet('data/reuse/skills_occurrences.parquet')
                         WHERE sha IN (SELECT sha FROM vs) GROUP BY sha""").df()
    # most common target per text; ties go to the first in alphabetical order (as in topics.py)
    ct = con.sql("""WITH o AS (SELECT sha, target, repo FROM read_parquet('data/context_occurrences.parquet')
                               WHERE sha IN (SELECT sha FROM vc)),
                         t AS (SELECT sha, target FROM (
                                 SELECT sha, target, row_number() OVER (PARTITION BY sha ORDER BY count(*) DESC, target) AS rn
                                 FROM o GROUP BY sha, target) WHERE rn = 1),
                         w AS (SELECT sha, min(split_part(repo, '/', 1)) AS owner FROM o GROUP BY sha)
                    SELECT sha, target, owner FROM t JOIN w USING (sha)""").df()
    sk = m[m.artifact == "skill"].merge(owner_sk, on="sha", how="left")
    sk["target"] = "SKILL.md"
    cm = m[m.artifact == "context"].merge(ct, on="sha", how="inner")
    res["context_without_valid_occurrence"] = int((m.artifact == "context").sum() - len(cm))
    m = pd.concat([sk, cm], ignore_index=True)
    m["owner"] = m["owner"].fillna("?")

    short = m["ntok"] < T.MIN_TOKENS
    other_lang = ~short & ((m["en"] < T.MIN_EN) | (m["nascii"] >= T.MAX_NON_ASCII))
    res["short"] = m[short].groupby("artifact").size().to_dict()
    res["non_english"] = m[other_lang].groupby("artifact").size().to_dict()
    m = m[~short & ~other_lang].reset_index(drop=True)
    dup = m.duplicated(["artifact", "prefix"]) | m["prefix"].isin(fit_prefixes)
    res["near_duplicates_or_in_fit"] = m[dup].groupby("artifact").size().to_dict()
    m = m[~dup].reset_index(drop=True)
    m = m.sample(frac=1, random_state=T.SEED)
    over_cap = m.groupby(["artifact", "owner"]).cumcount() >= T.MAX_PER_OWNER
    res["over_owner_cap"] = m[over_cap].groupby("artifact").size().to_dict()
    m = m[~over_cap]
    res["heldout_final"] = m.groupby("artifact").size().to_dict()
    print(json.dumps(res, indent=1), flush=True)

    # 3. step 2: projection and phrase counting, file by file
    cont = {a: np.zeros((k, len(terms))) for a in ("all", "skill", "context")}
    assigned = []
    for f, g in m.groupby("file"):
        v = pd.read_parquet(f)
        v = v[v.sha.isin(set(g.sha))].drop_duplicates("sha")
        art = g["artifact"].iloc[0]
        clean = [T.clean_text(t or "") for t in v["text"]]
        topic = nmf.transform(vec.transform(clean)).argmax(axis=1)
        C = cv.transform(clean).tocsc()
        for i in range(k):
            s = np.asarray(C[topic == i].sum(axis=0)).ravel()
            cont["all"][i] += s
            cont[art][i] += s
        assigned.append(pd.DataFrame({"sha": v.sha.values, "artifact": art, "topic": topic,
                                   "target": g.set_index("sha").loc[v.sha.values, "target"].values}))
        print(f"[val] {f}: {len(v)}", flush=True)
    a = pd.concat(assigned, ignore_index=True)
    a.to_parquet(os.path.join(OUT, "assignment.parquet"))

    # comparison of the %
    comp = []
    for i in range(k):
        row_d = {"topic": i}
        for art, col in (("skill", "pct_skills"), ("context", "pct_context")):
            g = a[a.artifact == art]
            row_d[f"{col}_fit"] = fit_pct.loc[i, col]
            row_d[f"{col}_heldout"] = round(100 * (g.topic == i).mean(), 2)
        comp.append(row_d)
    comp = pd.DataFrame(comp)
    comp.to_csv(os.path.join(OUT, "topic_comparison.csv"), index=False)
    res["max_diff_pp"] = round(float(max((comp.pct_skills_heldout - comp.pct_skills_fit).abs().max(),
                                        (comp.pct_context_heldout - comp.pct_context_fit).abs().max())), 2)
    print(comp.to_string(index=False), flush=True)

    c = a[a.artifact == "context"]
    pa = (pd.crosstab(c["target"], c["topic"], normalize="index") * 100).round(1)
    pa.insert(0, "n", c.groupby("target").size())
    pa = pa.sort_values("n", ascending=False)
    pa.to_csv(os.path.join(OUT, "topics_by_target.csv"))
    print(pa.head(6).to_string(), flush=True)

    # typical phrases
    n_topic = np.bincount(a["topic"], minlength=k)
    pt = typical_phrases(cont["all"], n_topic, terms, E.TOP_PER_TOPIC)
    pt["in_fit_top30"] = [e in set(fit_top[fit_top.topic == t].phrase) for t, e in zip(pt.topic, pt.phrase)]
    pt.to_csv(os.path.join(OUT, "phrases_by_topic.csv"), index=False)
    res["shared_top30_by_topic"] = pt.groupby("topic")["in_fit_top30"].sum().astype(int).to_dict()
    for i in range(k):
        print(f"topic {i}: " + " | ".join(pt[pt.topic == i].phrase.head(15)), flush=True)

    # coverage
    total = cont["all"].sum(axis=0)
    lift = (cont["all"] / n_topic[:, None]) / (total / n_topic.sum())
    rows = []
    for art in ("skill", "context"):
        df_art = cont[art].sum(axis=0)
        n_art = (a.artifact == art).sum()
        for r, j in enumerate(np.argsort(-df_art)[:E.TOP_OVERALL]):
            b = int(np.nanargmax(lift[:, j]))
            rows.append({"artifact": art, "position": r + 1, "phrase": terms[j],
                           "pct_docs": round(100 * df_art[j] / n_art, 2), "top_topic": b,
                           "lift": round(float(lift[b, j]), 2),
                           "phrase_class": "topical" if lift[b, j] >= E.TOPIC_LIFT else "cross_cutting"})
    cb = pd.DataFrame(rows)
    cb.to_csv(os.path.join(OUT, "coverage_top100.csv"), index=False)
    res["coverage"] = {f"{x}_{y}": int(n) for (x, y), n in cb.groupby(["artifact", "phrase_class"]).size().items()}
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=str)
    print(json.dumps({x: res[x] for x in ("max_diff_pp", "shared_top30_by_topic", "coverage")}, indent=1))
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
