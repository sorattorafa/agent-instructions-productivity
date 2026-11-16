#!/usr/bin/env python3
"""
Language of the texts that count in the productivity analysis (any framework: named mention or
metric), compared with the whole corpus. A descriptive side note of
the paper.

  download  full text of the shas in data/productivity/shas_any_<c>.parquet
            (written by productivity.py), straight from Hugging Face
  detect    language of each text: code blocks, inline code, URLs and
            HTML tags are removed; if at least 10% of the letters are CJK, the
            language is given by the script (kana -> ja, hangul -> ko,
            otherwise zh), because Chinese prose full of English terms fools
            the classifier; otherwise py3langid (Lui & Baldwin) on the first
            5,000 characters.
            Reference: the topic sample (SHA prefix = uniform random draw),
            only texts with a valid occurrence.
Output: data/languages/{texts_<c>.parquet, languages.csv, summary.json}
Usage: python scripts/languages.py download skills|context ; python scripts/languages.py detect
(run from the repository root)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

import duckdb
import pandas as pd
import py3langid
from huggingface_hub import HfApi

from vocabulary import CORPORA, q

OUT = os.path.join("data", "languages")
OCCURRENCES = {"skills": "data/reuse/skills_occurrences.parquet",
         "context": "data/context_occurrences.parquet"}
TOP = 5


def download(name: str) -> None:
    cfg = CORPORA[name]
    os.makedirs(OUT, exist_ok=True)
    shas = os.path.join("data", "productivity", f"shas_any_{name}.parquet")
    files = sorted(f for f in HfApi().list_repo_files(cfg["repo"], repo_type="dataset")
                      if f.startswith(cfg["prefix"]) and f.endswith(".parquet"))
    con = duckdb.connect()
    con.sql("CREATE SECRET hf (TYPE huggingface, PROVIDER credential_chain)")
    con.sql("SET threads = 4"); con.sql("SET memory_limit = '2GB'")
    con.sql(f"CREATE TABLE wanted AS SELECT sha FROM read_parquet({q(shas)})")
    con.sql("CREATE TABLE found (sha VARCHAR, text VARCHAR)")
    batch = cfg["batch"]
    for k in range(0, len(files), batch):
        file_list = "[" + ",".join(q(f"hf://datasets/{cfg['repo']}/{f}") for f in files[k:k + batch]) + "]"
        t0 = time.time()
        for attempt in range(5):
            try:
                con.sql(f"""INSERT INTO found SELECT {cfg['sha']}, left({cfg['text']}, 30000)
                            FROM read_parquet({file_list})
                            WHERE {cfg['filter']} AND {cfg['sha']} IN (SELECT sha FROM wanted)""")
                break
            except duckdb.Error as e:
                print(f"  error (attempt {attempt + 1}): {str(e)[:300]}", flush=True)
                time.sleep(30 * (attempt + 1))
        else:
            sys.exit("batch failed 5 times")
        n = con.sql("SELECT count(DISTINCT sha) FROM found").fetchall()[0][0]
        print(f"{time.strftime('%H:%M:%S')} {min(k + batch, len(files))}/{len(files)}: "
              f"{n} texts, {time.time() - t0:.0f}s", flush=True)
    con.sql(f"""COPY (SELECT sha, any_value(text) AS text FROM found GROUP BY sha)
               TO {q(os.path.join(OUT, f'texts_{name}.parquet'))} (FORMAT parquet, COMPRESSION zstd)""")
    print(f"DONE {time.strftime('%H:%M')}", flush=True)


CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
LETTER = re.compile(r"[^\W\d_]")


def clean_text(t: str) -> str:
    t = re.sub(r"```.*?(```|$)", " ", str(t), flags=re.S)
    t = re.sub(r"`[^`\n]*`", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    return re.sub(r"<[^>\n]+>", " ", t)


def language(t: str) -> str:
    t = clean_text(t)
    letters = LETTER.findall(t)
    if not letters:
        return "?"
    cjk = CJK.findall(t)
    if len(cjk) >= 0.1 * len(letters):
        if re.search(r"[぀-ヿ]", t):
            return "ja"
        if re.search(r"[가-힯]", t):
            return "ko"
        return "zh"
    return py3langid.classify(t[:5000])[0]


def detect() -> None:
    rows, res = [], {}
    for name in ("skills", "context"):
        productivity = pd.read_parquet(os.path.join(OUT, f"texts_{name}.parquet"))
        expected = len(pd.read_parquet(os.path.join("data", "productivity", f"shas_any_{name}.parquet")))
        val = set(duckdb.sql(f"SELECT DISTINCT sha FROM read_parquet({q(OCCURRENCES[name])})").df().sha)
        ref = pd.read_parquet(os.path.join("data", name, "sample"))
        ref = ref[ref.sha.isin(val)].drop_duplicates("sha")
        res[name] = {"mention_texts": len(productivity), "mention_expected": expected, "reference_texts": len(ref)}
        for group, df in (("productivity", productivity), ("corpus", ref)):
            c = df.text.map(language).value_counts()
            for lang, k in c.items():
                rows.append({"corpus": name, "group": group, "language": lang, "texts": int(k),
                               "pct": round(100 * k / len(df), 2)})
            res[name][group] = {lang: round(100 * k / len(df), 2) for lang, k in c.head(TOP).items()}
            res[name][group]["non_english"] = round(100 * (1 - c.get("en", 0) / len(df)), 2)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "languages.csv"), index=False)
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["download"] and sys.argv[2:] and sys.argv[2] in CORPORA:
        download(sys.argv[2])
    elif sys.argv[1:] == ["detect"]:
        detect()
    else:
        sys.exit(__doc__)
