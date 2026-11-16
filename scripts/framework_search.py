"""
First-round search for explicit framework mentions: looks for mentions of
DORA, SPACE, DevEx, developer experience and productivity in the file paths
and in the text of the full collection (the context-file dataset,
`CONTEXT_DATASET_REPO`, on Hugging Face).

Unlike lexicon.py (which maps DIMENSIONS anchored in the multivocal
literature review, MLR), here the search is for the EXPLICIT mention of the
frameworks and of the generic terms.

The pipeline uses two of its outputs:
  data/collection/occurrences.parquet   deduplicated occurrences
  data/size_ranges.parquet              size ranges of the collection

Other outputs in data/collection/:
  text_flags.parquet   one row per sha with some pattern, flags + snippets
  path_flags.parquet   occurrences (repo, path) whose path matches
  summary.csv          counts per pattern x target (distinct repos and files)
  overall_summary.csv  the same counts summed over the targets

One remote pass over textos/*.parquet, in batches of files, with a
checkpoint: if it stops, running again resumes from the next batch.
Usage: python scripts/framework_search.py [summary]
"""
import os, sys, time, json
import duckdb
from huggingface_hub import HfApi

from vocabulary import CONTEXT_REPO

BASE = f"hf://datasets/{CONTEXT_REPO}"
OUT = "data/collection"
PARTIALS = os.path.join(OUT, "partials")
DONE_FILE = os.path.join(OUT, "done.json")
BATCH = 40

# RE2 patterns. (?i) = case-insensitive. The acronym ones stay case-sensitive to avoid
# "dora" as a proper name, "space" as a key, "dx" as lower-case DirectX.
TEXT_PATTERNS = {
    # DORA: the acronym with metric/research context, and the four (five) metrics
    "dora_acronym": r"\bDORA\b",
    "dora_context": r"(?i)\bdora\b[^\n]{0,40}\b(metric|framework|report|research|key|capabilit|survey|score)"
                     r"|(?i)\b(state of devops|four key metrics|accelerate metrics)",
    "dora_metrics": r"(?i)deploy(ment)?[ -]frequency|lead[ -]time for changes|change[ -]failure[ -]rate"
                     r"|time to restore( service)?|mean[ -]time[ -]to[ -](restore|recover)|\bMTTR\b"
                     r"|failed deployment recovery time",
    "dora_eu_regulation": r"(?i)digital operational resilience",
    # SPACE: only the upper-case acronym attached to framework/metrics/dimensions
    "space_framework": r"\bSPACE\b[ -](framework|metrics|model|dimensions)|(?i)space framework"
                       r"|(?i)satisfaction and well-?being",
    # DevEx (Noda et al. 2023) and DX (getdx, Core 4)
    "devex": r"(?i)\bdevex\b",
    "dx_framework": r"(?i)\bdx core ?4\b|getdx\.com|\bDX\b[ -](framework|metrics|survey|index)",
    # generic terms
    "developer_experience": r"(?i)developer[ -]experience",
    "developer_productivity": r"(?i)(developer|engineering|engineer|dev|team)[ -]productivity",
    "productivity": r"(?i)\bproductiv",
}
# in the path: delimiters around the acronyms (avoids fedora, pandora, ...)
D = r"(^|[/_. -])"
F = r"([/_. -]|$)"
PATH_PATTERNS = {
    "dora": rf"(?i){D}dora{F}|(?i)dora[-_](metric|report)",
    "space": r"(?i)space[-_ ]?(framework|metric)",
    "devex": r"(?i)devex",
    "dx": rf"(?i){D}dx{F}",
    "developer_experience": r"(?i)developer[-_ ]?experience|dev[-_]?experience",
    "productivity": r"(?i)productiv",
}
SNIPPET_KEYS = ("dora_context", "dora_metrics", "space_framework", "devex", "dx_framework",
          "developer_experience", "developer_productivity")


def q(s):
    return "'" + s.replace("'", "''") + "'"


def connect():
    con = duckdb.connect()
    con.sql("CREATE SECRET hf (TYPE huggingface, PROVIDER credential_chain)")
    con.sql("SET threads = 8")
    return con


def scan(con):
    os.makedirs(PARTIALS, exist_ok=True)
    files = sorted(f for f in HfApi().list_repo_files(CONTEXT_REPO, repo_type="dataset")
                      if f.startswith("textos/") and f.endswith(".parquet"))
    done = set(json.load(open(DONE_FILE))) if os.path.exists(DONE_FILE) else set()
    todo = [f for f in files if f not in done]
    print(f"{len(files)} text files, {len(todo)} to scan", flush=True)
    flags = ",\n".join(f"regexp_matches(text, {q(p)}) AS {k}" for k, p in TEXT_PATTERNS.items())
    snippets = ",\n".join(
        f"CASE WHEN {k} THEN regexp_extract(text, {q('(?s).{0,150}(' + TEXT_PATTERNS[k] + ').{0,150}')}) END AS snippet_{k}"
        for k in SNIPPET_KEYS)
    any_cond = " OR ".join(TEXT_PATTERNS)
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        file_list = "[" + ",".join(q(f"{BASE}/{f}") for f in batch) + "]"
        dest = os.path.join(PARTIALS, f"{i // BATCH:04d}_{len(done):06d}.parquet")
        t = time.time()
        for attempt in range(5):
            try:
                con.sql(f"""
                COPY (
                  WITH f AS (SELECT sha, repo, byte_size, text, {flags}
                             FROM (SELECT sha, repo, byte_size, texto AS text FROM read_parquet({file_list}))
                             WHERE text IS NOT NULL)
                  SELECT sha, repo, byte_size, {", ".join(TEXT_PATTERNS)}, {snippets}
                  FROM f WHERE {any_cond}
                ) TO {q(dest)} (FORMAT parquet, COMPRESSION zstd)""")
                break
            except duckdb.Error as e:
                print(f"  error (attempt {attempt + 1}): {str(e)[:200]}", flush=True)
                time.sleep(30 * (attempt + 1))
        else:
            sys.exit("batch failed 5 times; run again to resume")
        done.update(batch)
        json.dump(sorted(done), open(DONE_FILE, "w"))
        print(f"{time.strftime('%H:%M:%S')} batch {i // BATCH + 1}/{-(-len(todo) // BATCH)}: "
              f"{len(done)}/{len(files)} files, {time.time() - t:.0f}s", flush=True)


def paths(con):
    flags = ",\n".join(f"regexp_matches(path, {q(p)}) AS path_{k}" for k, p in PATH_PATTERNS.items())
    any_cond = " OR ".join(f"path_{k}" for k in PATH_PATTERNS)
    # alvo, caminho, nome, nome_exato, familia, obtidos and the folder names are field names
    # of the dataset as collected
    con.sql(f"""
    COPY (
      WITH o AS (SELECT DISTINCT alvo AS target, repo, caminho AS path, nome AS name, sha,
                                 nome_exato AS exact_name, familia AS family
                 FROM read_parquet('{BASE}/ocorrencias/*.parquet')),
           f AS (SELECT *, {flags} FROM o)
      SELECT * FROM f WHERE {any_cond}
    ) TO '{OUT}/path_flags.parquet' (FORMAT parquet)""")
    # deduplicated occurrences, local, for the summary
    con.sql(f"""COPY (SELECT DISTINCT alvo AS target, repo, caminho AS path, sha
                      FROM read_parquet('{BASE}/ocorrencias/*.parquet'))
                TO '{OUT}/occurrences.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    # size ranges of the collection (which byte sizes were truncated at the search cap)
    con.sql(f"""COPY (SELECT alvo AS target, a, b,
                             CASE status WHEN 'truncada' THEN 'truncated' WHEN 'feita' THEN 'done'
                                         WHEN 'erro' THEN 'error' ELSE status END AS status,
                             total, obtidos AS fetched
                      FROM read_parquet('{BASE}/estado/faixas.parquet'))
                TO 'data/size_ranges.parquet' (FORMAT parquet)""")


def summary(con):
    con.sql(f"CREATE OR REPLACE VIEW t AS SELECT * FROM read_parquet('{PARTIALS}/*.parquet')")
    con.sql(f"CREATE OR REPLACE VIEW o AS SELECT * FROM read_parquet('{OUT}/occurrences.parquet')")
    con.sql(f"CREATE OR REPLACE VIEW c AS SELECT * FROM read_parquet('{OUT}/path_flags.parquet')")
    con.sql(f"COPY (SELECT * FROM t) TO '{OUT}/text_flags.parquet' (FORMAT parquet, COMPRESSION zstd)")
    parts = []
    for k in TEXT_PATTERNS:
        parts.append(f"""SELECT 'text' AS scope, '{k}' AS pattern, o.target,
            count(DISTINCT o.repo) AS repos, count(*) AS files, count(DISTINCT o.sha) AS distinct_texts
            FROM o JOIN t USING (sha) WHERE t.{k} GROUP BY o.target""")
    for k in PATH_PATTERNS:
        parts.append(f"""SELECT 'path', '{k}', target, count(DISTINCT repo), count(*), count(DISTINCT sha)
            FROM c WHERE path_{k} GROUP BY target""")
    parts.append("""SELECT 'total', 'all', target, count(DISTINCT repo), count(*), count(DISTINCT sha)
            FROM o GROUP BY target""")
    r = con.sql(" UNION ALL ".join(parts)).df().sort_values(["scope", "pattern", "files"],
                                                             ascending=[True, True, False])
    r.to_csv(f"{OUT}/summary.csv", index=False)
    overall = r.groupby(["scope", "pattern"])[["files", "distinct_texts"]].sum()
    repos = []
    for k in TEXT_PATTERNS:
        repos.append(("text", k, con.sql(f"SELECT count(DISTINCT o.repo) FROM o JOIN t USING (sha) WHERE t.{k}").fetchall()[0][0]))
    for k in PATH_PATTERNS:
        repos.append(("path", k, con.sql(f"SELECT count(DISTINCT repo) FROM c WHERE path_{k}").fetchall()[0][0]))
    repos.append(("total", "all", con.sql("SELECT count(DISTINCT repo) FROM o").fetchall()[0][0]))
    overall["repos"] = [dict(((a, b), n) for a, b, n in repos)[i] for i in overall.index]
    print(overall.to_string())
    overall.to_csv(f"{OUT}/overall_summary.csv")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    con = connect()
    if sys.argv[1:] != ["summary"]:
        paths(con)
        print("paths ok", flush=True)
        scan(con)
    summary(con)
