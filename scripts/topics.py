#!/usr/bin/env python3
"""
Dataset characterization (topics): what skills and context files instruct the agent about.

A single NMF over the two samples from corpus_pass.py (blob SHA prefix), so
that the topics are the same in both artifacts and the comparison is direct.
The two samples enter with the same size (the larger one is subsampled with a
fixed seed), so that neither artifact defines the topics on its own.

Cleaning (the same as in the earlier model): removes YAML front matter, code
blocks and inline code, URLs, link targets and Markdown markup.
Language filter: a text is kept only if it has >= MIN_TOKENS tokens, a
fraction of English function words >= MIN_EN and a fraction of non-ASCII
characters < MAX_NON_ASCII (a simple, documented heuristic; the discarded
fraction is written to diagnostics.json). MIN_EN used to be 0.20 and discarded
44% of the context files: instructions for agents are written as bullet
points, with few function words; after reading 30 texts per band
(13 Nov 2026), 0.10-0.20 is almost all English and < 0.08 is another
language.

Near duplicates: of the texts with the same normalized beginning (PREFIX_LEN
characters of the cleaned text), only one is kept. Without this, the version
variants of a single template (Next.js rules, Laravel Boost) form topics of
their own; counting those templates is the subject of Dataset characterization (replication), in reuse.py.

Choice of k: for each candidate, (i) the fraction of documents in the largest
group and (ii) the mean UMass coherence of the 10 terms of each component,
measured on the binary document matrix. The k with the highest coherence is
chosen among those that keep the largest group below MAX_GROUP. The curve is
written to k_selection.csv.

Cap per account: at most MAX_PER_OWNER texts per repository owner (random draw
with a seed). Without this, a single account with 1,748 mass-generated skills
(3 repos) formed a topic of its own (14 Nov 2026).

Stability: the chosen k is refitted on subsamples of SUBSAMPLE_FRACTION of the
documents, one per seed in SEEDS; each component of the main fit is paired
(Hungarian algorithm) with the most similar one of each refit, by cosine of
the term vectors. Written to stability.csv. (Refitting while changing only
random_state tests nothing: the nndsvda initialization is deterministic; the
first version gave cosine 1.0.)

Outputs in data/topics/:
  k_selection.csv, stability.csv, diagnostics.json
  topics.csv             per topic: terms, % of the skills, % of the context files
  topics_by_target.csv   per context file type: % in each topic
  examples.csv           the 10 highest-weight documents and 10 randomly drawn
                         ones per topic, to label the topic and check the label
Usage: python scripts/topics.py [k]   (run from the repository root)
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

OUT = os.path.join("data", "topics")
CANDIDATES = (8, 10, 12, 14, 16, 18, 20)
MAX_GROUP = 0.25
SEEDS = (1, 2, 3)
MIN_TOKENS = 40
MIN_EN = 0.08
MAX_NON_ASCII = 0.05
PREFIX_LEN = 300
MAX_PER_OWNER = 10
SUBSAMPLE_FRACTION = 0.8
SEED = 42

FRONT_MATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`]*`")
URL = re.compile(r"https?://\S+|www\.\S+")
LINK_MD = re.compile(r"\[([^\]]*)\]\([^)]*\)")
MARKUP = re.compile(r"[#*_>|~\[\]()<>{}`]")
WHITESPACE = re.compile(r"\s+")
WORD = re.compile(r"[a-z]+")
FUNCTION_WORDS = {"the", "and", "to", "of", "in", "for", "is", "that", "with", "you",
              "this", "are", "be", "on", "it", "as", "or", "if", "not", "from",
              "by", "at", "when", "should", "all", "any", "each", "will", "a", "an"}
# generic words of agent instructions: they appear in almost every text of
# both artifacts and do not separate subjects (same list as the earlier model,
# plus the names of the artifacts themselves)
NOISE = {
    "skill", "skills", "md", "use", "used", "using", "user", "users",
    "file", "files", "make", "sure", "need", "needs", "want", "like",
    "just", "don", "doesn", "didn", "won", "let", "way", "ways", "example",
    "examples", "note", "notes", "step", "steps", "following", "follow",
    "return", "returns", "true", "false", "null", "yes", "http", "https",
    "www", "com", "org", "io", "new", "old", "good", "bad", "best",
    "claude", "agents", "agent", "gemini", "copilot", "cursor", "windsurf", "cline",
}


def clean_text(t: str) -> str:
    t = FRONT_MATTER.sub(" ", t)
    t = CODE_BLOCK.sub(" ", t)
    t = INLINE_CODE.sub(" ", t)
    t = URL.sub(" ", t)
    t = LINK_MD.sub(r"\1", t)
    t = MARKUP.sub(" ", t)
    return WHITESPACE.sub(" ", t).lower().strip()


def english_stats(t: str) -> tuple[int, float]:
    p = WORD.findall(t)
    if not p:
        return 0, 0.0
    return len(p), sum(w in FUNCTION_WORDS for w in p) / len(p)


def load(con) -> pd.DataFrame:
    sk = pd.read_parquet(glob.glob("data/skills/sample/*.parquet"))
    sk["artifact"], sk["target"] = "skill", "SKILL.md"
    ct = pd.read_parquet(glob.glob("data/context/sample/*.parquet"))
    ct["artifact"] = "context"
    # file type: one sha may be in more than one target; the most common is kept
    con.register("ct_sha", ct[["sha"]])
    # ties go to the first target in alphabetical order (mode() breaks ties by row order)
    target = con.sql("""SELECT sha, target FROM (
                          SELECT sha, target, row_number() OVER (PARTITION BY sha ORDER BY count(*) DESC, target) AS rn
                          FROM read_parquet('data/context_occurrences.parquet')
                          WHERE sha IN (SELECT sha FROM ct_sha) GROUP BY sha, target) WHERE rn = 1""").df()
    # inner: a text with no valid occurrence leaves the sample (name_filter.py)
    ct = ct.merge(target, on="sha", how="inner")
    # repository owner (one account per text: the first in alphabetical order)
    con.register("sk_sha", sk[["sha"]])
    owner_sk = con.sql("""SELECT sha, min(split_part(repo, '/', 1)) AS owner
                         FROM read_parquet('data/reuse/skills_occurrences.parquet')
                         WHERE sha IN (SELECT sha FROM sk_sha) GROUP BY sha""").df()
    owner_ct = con.sql("""SELECT sha, min(split_part(repo, '/', 1)) AS owner
                         FROM read_parquet('data/context_occurrences.parquet')
                         WHERE sha IN (SELECT sha FROM ct_sha) GROUP BY sha""").df()
    sk = sk.merge(owner_sk, on="sha", how="left")
    ct = ct.merge(owner_ct, on="sha", how="left")
    return pd.concat([sk, ct], ignore_index=True)


def umass(Xb, top_idx: np.ndarray, df: np.ndarray) -> float:
    """Mean UMass coherence of the term pairs of one component."""
    cols = Xb[:, top_idx]
    co = (cols.T @ cols).toarray()
    s, n = 0.0, 0
    for i in range(1, len(top_idx)):
        for j in range(i):
            s += np.log((co[i, j] + 1) / df[top_idx[j]])
            n += 1
    return s / n


def fit_nmf(X, k: int, seed: int):
    m = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=400)
    return m, m.fit_transform(X)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    con = duckdb.connect()
    d = load(con)
    diag = {"raw_sample": d.groupby("artifact").size().to_dict()}
    d["clean"] = [clean_text(t or "") for t in d["text"]]
    tok = [english_stats(t) for t in d["clean"]]
    d["ntok"], d["en"] = [a for a, _ in tok], [b for _, b in tok]
    d["nascii"] = [sum(ord(c) > 127 for c in t) / max(1, len(t)) for t in d["clean"]]
    short = d["ntok"] < MIN_TOKENS
    other_lang = ~short & ((d["en"] < MIN_EN) | (d["nascii"] >= MAX_NON_ASCII))
    diag["short"] = d[short].groupby("artifact").size().to_dict()
    diag["non_english"] = d[other_lang].groupby("artifact").size().to_dict()
    d = d[~short & ~other_lang].reset_index(drop=True)
    d["prefix"] = d["clean"].str[:PREFIX_LEN]
    dup = d.duplicated(["artifact", "prefix"])
    diag["near_duplicates"] = d[dup].groupby("artifact").size().to_dict()
    d = d[~dup].reset_index(drop=True)
    d["owner"] = d["owner"].fillna("?")
    d = d.sample(frac=1, random_state=SEED)
    over_cap = d.groupby(["artifact", "owner"]).cumcount() >= MAX_PER_OWNER
    diag["over_owner_cap"] = d[over_cap].groupby("artifact").size().to_dict()
    d = d[~over_cap].sort_index().reset_index(drop=True)
    diag["after_filters"] = d.groupby("artifact").size().to_dict()
    n = int(d.groupby("artifact").size().min())
    d = (d.groupby("artifact", group_keys=False)
           .apply(lambda g: g.sample(n=n, random_state=SEED) if len(g) > n else g)
           .reset_index(drop=True))
    diag["balanced_per_artifact"] = n
    print(json.dumps(diag, indent=1), flush=True)

    stop_list = sorted(set(ENGLISH_STOP_WORDS) | NOISE)
    vec = TfidfVectorizer(stop_words=stop_list, min_df=30, max_df=0.30, max_features=50_000,
                          ngram_range=(1, 2), sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-z][a-z-]{2,}\b")
    X = vec.fit_transform(d["clean"])
    terms = vec.get_feature_names_out()
    Xb = (X > 0).astype(np.float64).tocsc()
    df = np.asarray(Xb.sum(axis=0)).ravel()
    diag["matrix"] = list(X.shape)
    print(f"[topics] matrix {X.shape}", flush=True)

    if len(sys.argv) > 1:
        k = int(sys.argv[1])
    else:
        rows = []
        for cand in CANDIDATES:
            m, W = fit_nmf(X, cand, SEED)
            dom = W.argmax(axis=1)
            largest = np.bincount(dom, minlength=cand).max() / len(dom)
            coherence = np.mean([umass(Xb, c.argsort()[:-11:-1], df) for c in m.components_])
            rows.append({"k": cand, "largest_group": round(float(largest), 4), "umass": round(float(coherence), 4)})
            print(rows[-1], flush=True)
        curve = pd.DataFrame(rows)
        curve.to_csv(os.path.join(OUT, "k_selection.csv"), index=False)
        okk = curve[curve["largest_group"] < MAX_GROUP]
        base = okk if not okk.empty else curve
        k = int(base.loc[base["umass"].idxmax(), "k"])
    diag["k"] = k
    print(f"[topics] k = {k}", flush=True)

    m, W = fit_nmf(X, k, SEED)
    H = m.components_
    Hn = H / np.linalg.norm(H, axis=1, keepdims=True)

    # stability
    stab = []
    rng_sub = np.random.default_rng(SEED)
    for s in SEEDS:
        idx = np.sort(rng_sub.choice(X.shape[0], int(SUBSAMPLE_FRACTION * X.shape[0]), replace=False))
        m2, _ = fit_nmf(X[idx], k, s)
        H2 = m2.components_ / np.linalg.norm(m2.components_, axis=1, keepdims=True)
        sim = Hn @ H2.T
        lin, col = linear_sum_assignment(-sim)
        for i, j in zip(lin, col):
            stab.append({"seed": s, "topic": int(i), "cosine": round(float(sim[i, j]), 4)})
    stab = pd.DataFrame(stab)
    stab.to_csv(os.path.join(OUT, "stability.csv"), index=False)
    diag["min_stability_cosine_by_topic"] = stab.groupby("topic")["cosine"].min().round(3).to_dict()

    dom = W.argmax(axis=1)
    d["topic"] = dom
    d["weight"] = W.max(axis=1)
    rows = []
    for i in range(k):
        g = d[d["topic"] == i]
        rows.append({
            "topic": i,
            "terms": ", ".join(terms[j] for j in H[i].argsort()[:-13:-1]),
            "pct_skills": round(100 * (g["artifact"] == "skill").sum() / n, 2),
            "pct_context": round(100 * (g["artifact"] == "context").sum() / n, 2),
            "umass": round(float(umass(Xb, H[i].argsort()[:-11:-1], df)), 3),
        })
    topics = pd.DataFrame(rows)
    topics.to_csv(os.path.join(OUT, "topics.csv"), index=False)
    print(topics.to_string(index=False), flush=True)

    ct = d[d["artifact"] == "context"]
    pa = (pd.crosstab(ct["target"], ct["topic"], normalize="index") * 100).round(1)
    pa.insert(0, "n", ct.groupby("target").size())
    pa.sort_values("n", ascending=False).to_csv(os.path.join(OUT, "topics_by_target.csv"))

    ex = []
    rng = np.random.default_rng(SEED)
    for i in range(k):
        g = d[d["topic"] == i]
        for kind, sel in (("top", g.nlargest(10, "weight")),
                          ("random", g.iloc[rng.choice(len(g), size=min(10, len(g)), replace=False)])):
            for _, r in sel.iterrows():
                ex.append({"topic": i, "kind": kind, "artifact": r["artifact"], "target": r["target"],
                           "sha": r["sha"], "snippet": r["clean"][:500]})
    pd.DataFrame(ex).to_csv(os.path.join(OUT, "examples.csv"), index=False)
    d[["sha", "artifact", "target", "topic", "weight"]].to_parquet(os.path.join(OUT, "assignment.parquet"))
    json.dump(diag, open(os.path.join(OUT, "diagnostics.json"), "w"), indent=1, default=str)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
