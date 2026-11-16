#!/usr/bin/env python3
"""
Support for labeling the Dataset characterization (topics) topics: 2- and 3-word phrases.

Uses exactly the documents of the topic model (assignment.parquet from
topics.py: same cleaning, language filter, near duplicates, cap per owner and
balancing), and counts each phrase once per document (document frequency), so
that a text that repeats the phrase does not weigh more.

Phrases: 2- and 3-word n-grams of the cleaned text, WITHOUT removing function
words first (to keep "do not modify", "pull request"). A phrase made up only
of function or generic words (the NOISE list in topics.py) is dropped, and so
is one that appears in fewer than MIN_DOCS documents.

1) Per topic (phrases_by_topic.csv): the TOP_PER_TOPIC most typical phrases of
   each topic. Typicality = p_in * log(p_in / p_out), with p_in the fraction
   of the topic's documents that contain the phrase and p_out that of the
   other topics (the contribution of each phrase to the KL divergence; it
   favors what is at once frequent in the topic and rare outside it).
   Requires p_in >= MIN_PIN.

2) Coverage (coverage_top100.csv): the TOP_OVERALL phrases with the highest
   document frequency, separately for skills and context files. For each one,
   the topic in which it is most over-represented (lift = fraction in the
   topic / overall fraction) and the fraction of the documents containing it
   that fall in that topic. phrase_class = "topical" if lift >= TOPIC_LIFT,
   otherwise "cross_cutting". A frequent cross-cutting phrase is either
   language common to all topics OR a subject that no topic captures; that
   reading is manual (manual_reading column, for the reviewer).

Outputs in data/topics/. Does not change any number in the paper.
Usage: python scripts/phrases.py   (run from the repository root)
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

import topics

OUT = os.path.join("data", "topics")
MIN_DOCS = 50
MIN_PIN = 0.02
TOP_PER_TOPIC = 30
TOP_OVERALL = 100
TOPIC_LIFT = 1.5
GENERIC_WORDS = set(ENGLISH_STOP_WORDS) | topics.NOISE


def load() -> pd.DataFrame:
    a = pd.read_parquet(os.path.join(OUT, "assignment.parquet"))
    sk = pd.read_parquet(glob.glob("data/skills/sample/*.parquet"), columns=["sha", "text"])
    sk["artifact"] = "skill"
    ct = pd.read_parquet(glob.glob("data/context/sample/*.parquet"), columns=["sha", "text"])
    ct["artifact"] = "context"
    t = pd.concat([sk, ct], ignore_index=True).drop_duplicates(["artifact", "sha"])
    d = a.merge(t, on=["artifact", "sha"], how="inner")
    assert len(d) == len(a), "model documents with no text in the sample"
    d["clean"] = [topics.clean_text(x or "") for x in d["text"]]
    return d.drop(columns="text")


def main() -> None:
    d = load()
    k = int(d["topic"].max()) + 1
    print(f"[expr] {len(d)} documents, k = {k}", flush=True)

    vec = CountVectorizer(ngram_range=(2, 3), min_df=MIN_DOCS, binary=True,
                          token_pattern=r"(?u)\b[a-z][a-z-]+\b", dtype=np.int32)
    X = vec.fit_transform(d["clean"]).tocsc()
    terms = vec.get_feature_names_out()
    useful = np.array([not all(w in GENERIC_WORDS for w in t.split()) for t in terms])
    X, terms = X[:, useful], terms[useful]
    print(f"[expr] {X.shape[1]} phrases", flush=True)

    topic = d["topic"].to_numpy()
    n_topic = np.bincount(topic, minlength=k)
    # documents of each topic that contain each phrase (k x terms)
    cont = np.vstack([np.asarray(X[topic == i].sum(axis=0)).ravel() for i in range(k)]).astype(float)
    total = cont.sum(axis=0)

    # 1) per topic
    rows = []
    for i in range(k):
        p_in = cont[i] / n_topic[i]
        p_out = (total - cont[i]) / (len(d) - n_topic[i])
        score = np.where(p_in >= MIN_PIN, p_in * np.log((p_in + 1e-9) / (p_out + 1e-9)), -np.inf)
        for r, j in enumerate(np.argsort(-score)[:TOP_PER_TOPIC]):
            rows.append({"topic": i, "position": r + 1, "phrase": terms[j],
                           "pct_in_topic": round(100 * p_in[j], 2), "pct_outside": round(100 * p_out[j], 2)})
    pt = pd.DataFrame(rows)
    pt.to_csv(os.path.join(OUT, "phrases_by_topic.csv"), index=False)
    for i in range(k):
        print(f"topic {i}: " + " | ".join(pt[pt.topic == i].phrase.head(15)), flush=True)

    # 2) coverage of the most frequent ones, per artifact
    topic_frac = cont / n_topic[:, None]
    overall_frac = total / len(d)
    lift = topic_frac / overall_frac
    rows = []
    for art in ("skill", "context"):
        sel = (d["artifact"] == art).to_numpy()
        df_art = np.asarray(X[sel].sum(axis=0)).ravel()
        for r, j in enumerate(np.argsort(-df_art)[:TOP_OVERALL]):
            b = int(lift[:, j].argmax())
            rows.append({"artifact": art, "position": r + 1, "phrase": terms[j],
                           "pct_docs": round(100 * df_art[j] / sel.sum(), 2),
                           "top_topic": b, "lift": round(float(lift[b, j]), 2),
                           "pct_in_topic": round(100 * cont[b, j] / total[j], 1),
                           "phrase_class": "topical" if lift[b, j] >= TOPIC_LIFT else "cross_cutting",
                           "manual_reading": ""})
    cb = pd.DataFrame(rows)
    cb.to_csv(os.path.join(OUT, "coverage_top100.csv"), index=False)
    print(cb.groupby(["artifact", "phrase_class"]).size().to_string(), flush=True)
    print(cb.groupby(["artifact", "top_topic"]).size().unstack(fill_value=0).to_string(), flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
