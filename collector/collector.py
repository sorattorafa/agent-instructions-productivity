"""
Collector of the context-file dataset: agent instruction and context files on GitHub.

Repeats the GitSkills discovery procedure (code search by file name or path,
partitioned by file size until every range fits under the 1,000-result cap)
for AGENTS.md, CLAUDE.md and the equivalent files of other tools.

Two lines of execution:
  - discovery (main thread): code search, 10 requests/min;
  - download (secondary thread): text via GraphQL, 50 blobs per query,
    deduplicated by blob sha, so each content is downloaded only once.

The data does not stay on disk. At every flush, occurrences and texts become
Parquet, are uploaded to the private dataset on the Hugging Face Hub and the
local copy is deleted. Only the state (finished ranges, seen shas) is kept
locally, in collector/state/state.db, which makes it possible to resume after
a crash: just run it again.

A range is only marked as done after its occurrences have been uploaded. If
the process dies between the upload and the marking, the range is redone and
produces repeated rows; the analysis deduplicates by (repo, caminho, sha).

Usage:
  python collector/collector.py          # collect (resumes if already started)
  python collector/collector.py status   # summary of the local state
"""

# ---------------------------------------------------------------------------
# Glossary of the data schema.
#
# The published dataset was collected with Portuguese field names, so the
# strings below are part of the data schema and are kept exactly as they are
# (Parquet columns, Hub folder names, SQLite tables/columns, status values).
#
#   Fields
#     alvo                = target file type
#     caminho             = path
#     nome                = file name
#     nome_exato          = exact-name match
#     familia             = tool family
#     faixa_a / faixa_b   = size range bounds in bytes
#     coletado_em         = collected at
#     texto               = text
#     baixado_em          = downloaded at
#     faixas              = size ranges (SQLite table and Parquet state file)
#     obtidos             = fetched
#     atualizado          = updated
#     ocorrencias, textos, estado = occurrences, texts, state (Hub folders)
#
#   Status values
#     pendente = pending       processada = processed    feita = done
#     truncada = truncated     erro = error              baixado = downloaded
#     falhou   = failed        buffer = held in the in-memory buffer
#
#   Tool family values
#     neutro = neutral         desconhecido = unknown
#
#   Match types in the target list
#     "nome" = match by file name    "dir" = match by directory
# ---------------------------------------------------------------------------

import json
import math
import os
import queue
import shutil
import signal
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(ROOT, "state")
STATE_DB = os.path.join(STATE_DIR, "state.db")
HF_REPO = os.environ.get("CONTEXT_DATASET_REPO", "ANONYMIZED/agent-context-files")

MAX_SIZE = 393216         # code search only indexes files below 384 KB
CAP = 1000                # results reachable per query
LEAF_TARGET = 800         # when splitting, aim for ranges with ~800 results
SEARCH_PAUSE = 6.2        # 10 searches/min
GQL_BATCH = 50
FLUSH_OCCURRENCES = 20000
FLUSH_BYTES = 40_000_000
FLUSH_SECONDS = 600

# (target, base query, exact-match type)
TARGETS = [
    ("CLAUDE.local.md", "filename:CLAUDE.local.md", "nome"),
    ("GEMINI.md", "filename:GEMINI.md", "nome"),
    (".windsurfrules", "filename:.windsurfrules", "nome"),
    (".clinerules", "filename:.clinerules", "nome"),
    (".cursorrules", "filename:.cursorrules", "nome"),
    ("copilot-instructions.md", "filename:copilot-instructions.md path:.github", "nome"),
    (".github/instructions", "path:.github/instructions extension:md", "dir"),
    # the two main targets come before the directories (decision of 4 Nov 2026)
    ("CLAUDE.md", "filename:CLAUDE.md", "nome"),
    ("AGENTS.md", "filename:AGENTS.md", "nome"),
    (".claude/rules", "path:.claude/rules extension:md", "dir"),
    (".claude/agents", "path:.claude/agents extension:md", "dir"),
    (".claude/commands", "path:.claude/commands extension:md", "dir"),
    (".cursor/rules", "path:.cursor/rules", "dir"),
    ("memory-bank", "path:memory-bank extension:md", "dir"),
]

stop = threading.Event()
discovery_done = threading.Event()
lock = threading.Lock()


def log(msg):
    print(f"{datetime.now():%m-%d %H:%M:%S} {msg}", flush=True)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_token():
    tok = os.environ.get("GITHUB_TOKEN")
    if not tok:
        tok = open(os.path.expanduser("~/.config/gitskills/github_token")).read().strip()
    return tok


TOKEN = None


# ---------------------------------------------------------------- state

def connect():
    con = sqlite3.connect(STATE_DB, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS faixas (
            alvo TEXT, a INTEGER, b INTEGER, status TEXT, total INTEGER,
            obtidos INTEGER, atualizado TEXT, PRIMARY KEY (alvo, a, b));
        CREATE TABLE IF NOT EXISTS shas (
            sha TEXT PRIMARY KEY, repo TEXT, status TEXT);
        CREATE INDEX IF NOT EXISTS shas_status ON shas(status);
    """)
    return con


def initial_grid():
    # geometric ranges: narrow where there are many small files
    ranges, a = [(0, 99)], 100
    while a <= MAX_SIZE:
        w = max(100, int(a * 0.1))
        ranges.append((a, min(a + w - 1, MAX_SIZE)))
        a += w
    return ranges


def seed(con):
    # whatever was processed but never reached the Hub was lost in the crash: redo it
    con.execute("UPDATE faixas SET status='pendente' WHERE status='processada'")
    con.execute("UPDATE shas SET status='pendente' WHERE status='buffer'")
    for target, _, _ in TARGETS:
        if con.execute("SELECT 1 FROM faixas WHERE alvo=? LIMIT 1", (target,)).fetchone():
            continue
        con.executemany("INSERT INTO faixas VALUES (?,?,?,'pendente',NULL,NULL,?)",
                        [(target, a, b, now()) for a, b in initial_grid()])
    con.commit()


# ---------------------------------------------------------------- http

def request(url, data=None, attempts=6):
    headers = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "context-file-collector"}
    t, blocks = 0, 0
    while t < attempts:
        if stop.is_set():
            return None, None
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r), r.headers
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode(errors="replace")
            if e.code in (403, 429):
                ra = e.headers.get("retry-after")
                reset = e.headers.get("x-ratelimit-reset")
                if ra:
                    wait = int(ra) + 1
                elif reset and e.headers.get("x-ratelimit-remaining") == "0":
                    wait = max(5, int(reset) - int(time.time()) + 2)
                else:
                    wait = 60
                # a rate-limit block does not consume an attempt; if it
                # repeats, the wait grows (a secondary limit asks for backing
                # off, not for insisting)
                wait = max(wait, min(600, 15 * 2 ** blocks))
                blocks += 1
                if "/search/" in url:
                    # secondary search limit: slow down the pace of all searches
                    pause[0] = min(30.0, pause[0] * 1.5)
                log(f"rate limit {e.code}, waiting {wait}s, pause {pause[0]:.1f}s: {body[:100]}")
                stop.wait(wait)
                continue
            if e.code == 422:
                return {"erro422": body}, e.headers
            if e.code >= 500 or e.code == 408:
                t += 1
                stop.wait(10 * t)
                continue
            log(f"HTTP {e.code} {url[:120]} {body[:120]}")
            return None, None
        except Exception as e:
            t += 1
            log(f"network failure ({type(e).__name__}), attempt {t}")
            stop.wait(10 * t)
    return None, None


last_search = [0.0]
pause = [SEARCH_PAUSE]    # adaptive pace: up 50% on each block, down 3% on each success, capped at 30 s
window = {"remaining": None, "reset": 0}


def search(q, page=1, order=None):
    wait = pause[0] - (time.time() - last_search[0])
    # the fixed pause alone drifts out of step with the API window after a
    # network failure; the counter returned by the API itself rules
    if window["remaining"] == 0:
        wait = max(wait, window["reset"] - time.time() + 1)
    if wait > 0:
        stop.wait(wait)
    params = {"q": q, "per_page": 100, "page": page}
    if order:
        params.update(sort="indexed", order=order)
    for _ in range(3):
        last_search[0] = time.time()
        body, headers = request("https://api.github.com/search/code?" + urllib.parse.urlencode(params))
        if body is not None and "erro422" not in body:
            pause[0] = max(SEARCH_PAUSE, pause[0] * 0.97)
        if headers and headers.get("x-ratelimit-remaining") is not None:
            window["remaining"] = int(headers["x-ratelimit-remaining"])
            window["reset"] = int(headers.get("x-ratelimit-reset", 0))
        if body is None or "erro422" in body or not body.get("incomplete_results"):
            return body
        stop.wait(SEARCH_PAUSE)
    return body


# ---------------------------------------------------------------- discovery

def family(path):
    c = "/" + path.lower()
    name = c.rsplit("/", 1)[-1]
    if "/.claude/" in c or name.startswith("claude"):
        return "claude"
    if "/.cursor/" in c or name == ".cursorrules":
        return "cursor"
    if "/.github/" in c and ("copilot" in name or "/.github/instructions/" in c):
        return "copilot"
    if name == "gemini.md":
        return "gemini"
    if name == ".windsurfrules":
        return "windsurf"
    if name == ".clinerules" or "/.clinerules/" in c or "/memory-bank/" in c:
        return "cline"
    if name == "agents.md":
        return "neutro"
    return "desconhecido"


def is_exact(target, match_type, path, name):
    if match_type == "nome":
        ok = name.lower() == target.lower()
        if target == "copilot-instructions.md":
            ok = ok and path.lower().endswith(".github/copilot-instructions.md")
        return ok
    return f"/{target.lower()}/" in "/" + path.lower()


class Buffer:
    def __init__(self):
        self.occurrences, self.texts, self.ranges = [], [], []
        self.bytes, self.start = 0, time.time()

    def full(self):
        return (len(self.occurrences) >= FLUSH_OCCURRENCES or self.bytes >= FLUSH_BYTES
                or time.time() - self.start >= FLUSH_SECONDS)


buf = Buffer()
counter = {"flush": 0}


def ingest(items, target, match_type, a, b, con):
    rows, new = [], []
    t = now()
    for i in items:
        r = i["repository"]
        rows.append({
            "alvo": target, "repo": r["full_name"], "repo_id": r["id"], "fork": r.get("fork"),
            "caminho": i["path"], "nome": i["name"], "sha": i["sha"],
            "nome_exato": is_exact(target, match_type, i["path"], i["name"]),
            "familia": family(i["path"]), "faixa_a": a, "faixa_b": b, "coletado_em": t})
        new.append((i["sha"], r["full_name"]))
    con.executemany("INSERT OR IGNORE INTO shas VALUES (?,?,'pendente')", new)
    con.commit()
    with lock:
        buf.occurrences.extend(rows)


def process_range(con, target, q, match_type, a, b):
    qr = f"{q} size:{a}..{b}"
    r = search(qr)
    if r is None:
        return False
    if "erro422" in r:
        log(f"[{target}] 422 at {a}..{b}: {r['erro422'][:100]}")
        con.execute("UPDATE faixas SET status='erro', atualizado=? WHERE alvo=? AND a=? AND b=?",
                    (now(), target, a, b))
        con.commit()
        return True
    total, items = r.get("total_count", 0), r.get("items", [])

    if total > CAP and a < b:
        k = min(math.ceil(total / LEAF_TARGET), b - a + 1)
        step = (b - a + 1) / k
        cuts = sorted({a + int(round(step * j)) for j in range(k)} | {b + 1})
        children = [(cuts[j], cuts[j + 1] - 1) for j in range(len(cuts) - 1)]
        con.execute("DELETE FROM faixas WHERE alvo=? AND a=? AND b=?", (target, a, b))
        con.executemany("INSERT OR IGNORE INTO faixas VALUES (?,?,?,'pendente',NULL,NULL,?)",
                        [(target, x, y, now()) for x, y in children])
        con.commit()
        log(f"[{target}] {a}..{b} total={total} -> {len(children)} ranges")
        return True

    fetched = list(items)
    page = 1
    while len(items) == 100 and page < 10 and not stop.is_set():
        page += 1
        r = search(qr, page)
        if r is None:
            return False
        items = r.get("items", []) if "erro422" not in r else []
        fetched.extend(items)
    status = "feita"
    if total > CAP:
        # a single byte size exceeds the cap: the two sort orders give up to 2,000
        status = "truncada"
        for order in ("asc", "desc"):
            for p in range(1, 11):
                r = search(qr, p, order)
                if r is None:
                    return False
                if "erro422" in r or not r.get("items"):
                    break
                fetched.extend(r["items"])
                if len(r["items"]) < 100:
                    break
        seen, unique = set(), []
        for i in fetched:
            key = (i["repository"]["full_name"], i["path"])
            if key not in seen:
                seen.add(key)
                unique.append(i)
        fetched = unique
        log(f"[{target}] byte {a} truncated: total={total}, fetched={len(fetched)}")
    if stop.is_set():
        return False
    ingest(fetched, target, match_type, a, b, con)
    con.execute("UPDATE faixas SET status='processada' WHERE alvo=? AND a=? AND b=?", (target, a, b))
    con.commit()
    with lock:
        buf.ranges.append((status, total, len(fetched), target, a, b))
    return True


def discover():
    con = connect()
    try:
        for target, q, match_type in TARGETS:
            while not stop.is_set():
                pending = con.execute(
                    "SELECT a, b FROM faixas WHERE alvo=? AND status='pendente' ORDER BY a LIMIT 1",
                    (target,)).fetchone()
                if not pending:
                    break
                if not process_range(con, target, q, match_type, *pending):
                    # persistent failure: wait and try the same range again,
                    # instead of skipping the rest of the target
                    if not stop.is_set():
                        log(f"[{target}] range {pending[0]}..{pending[1]} failed, retrying in 120s")
                        stop.wait(120)
                    continue
                if buf.full():
                    flush(con)
            if stop.is_set():
                break
            log(f"[{target}] discovery finished")
    finally:
        discovery_done.set()
        con.close()


# ---------------------------------------------------------------- download

def download_batch(batch):
    parts = []
    for j, (sha, repo) in enumerate(batch):
        owner, name = repo.split("/", 1)
        parts.append(f'b{j}: repository(owner:{json.dumps(owner)}, name:{json.dumps(name)}) '
                     f'{{ object(oid:"{sha}") {{ ... on Blob {{ byteSize isBinary isTruncated text }} }} }}')
    q = "query { rateLimit { remaining resetAt } " + " ".join(parts) + " }"
    body, _ = request("https://api.github.com/graphql", json.dumps({"query": q}).encode())
    if body is None:
        return None, None
    return body.get("data") or {}, body.get("errors") or []


gql_attempts = [0]


def download():
    con = connect()
    try:
        while not stop.is_set():
            batch = con.execute("SELECT sha, repo FROM shas WHERE status='pendente' LIMIT ?",
                                (GQL_BATCH,)).fetchall()
            if not batch:
                if discovery_done.is_set():
                    break
                stop.wait(15)
                continue
            data, errors = download_batch(batch)
            if data is None:
                if stop.is_set():
                    break
                stop.wait(30)
                continue
            if not any(k.startswith("b") for k in data) and gql_attempts[0] < 3:
                gql_attempts[0] += 1
                log(f"GraphQL returned no data ({str(errors)[:120]}), retrying")
                stop.wait(30)
                continue
            gql_attempts[0] = 0
            rows, failures, t = [], [], now()
            for j, (sha, repo) in enumerate(batch):
                o = (data.get(f"b{j}") or {}).get("object")
                if not o:
                    failures.append(sha)
                    continue
                rows.append({"sha": sha, "repo": repo, "byte_size": o.get("byteSize"),
                             "is_binary": o.get("isBinary"), "is_truncated": o.get("isTruncated"),
                             "texto": o.get("text"), "baixado_em": t})
            con.executemany("UPDATE shas SET status='falhou' WHERE sha=?", [(s,) for s in failures])
            con.executemany("UPDATE shas SET status='buffer' WHERE sha=?", [(l["sha"],) for l in rows])
            con.commit()
            with lock:
                buf.texts.extend(rows)
                buf.bytes += sum(len(l["texto"] or "") for l in rows)
            rl = data.get("rateLimit") or {}
            if rl.get("remaining") is not None and rl["remaining"] < 100:
                reset = datetime.fromisoformat(rl["resetAt"].replace("Z", "+00:00"))
                wait = max(5, (reset - datetime.now(timezone.utc)).total_seconds() + 5)
                log(f"GraphQL close to the limit, waiting {wait:.0f}s")
                stop.wait(wait)
            else:
                stop.wait(2)
    finally:
        con.close()


# ---------------------------------------------------------------- upload

QUEUE_DIR = os.path.join(STATE_DIR, "queue")
upload_queue = queue.Queue()


def flush(con, final=False):
    # only serialises and enqueues; the upload runs in another thread so the
    # search does not stall while the Hub receives the files
    with lock:
        occurrences, texts, ranges = buf.occurrences, buf.texts, buf.ranges
        buf.occurrences, buf.texts, buf.ranges = [], [], []
        buf.bytes, buf.start = 0, time.time()
    if not (occurrences or texts or ranges):
        return
    # disk filled up by another program: hold the batch in memory and wait for space
    warned = False
    while shutil.disk_usage(QUEUE_DIR).free < 1_000_000_000 and not stop.is_set():
        if not warned:
            log("less than 1 GB of free disk, waiting for space")
            warned = True
        stop.wait(300)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    files = []
    for name, rows in (("ocorrencias", occurrences), ("textos", texts)):
        if rows:
            rel = f"{name}/{stamp}.parquet"
            pd.DataFrame(rows).to_parquet(os.path.join(QUEUE_DIR, rel), compression="zstd", index=False)
            files.append(rel)
    upload_queue.put({"files": files, "shas": [l["sha"] for l in texts], "ranges": ranges,
                      "n": (len(occurrences), len(texts))})


def upload():
    # if the upload breaks, the process stops and the supervisor restarts it;
    # the ranges and texts not uploaded go back to pending and are redone
    try:
        _upload()
    except Exception:
        import traceback
        log("Traceback in upload: " + traceback.format_exc()[-600:])
        stop.set()
        os._exit(1)


def _upload():
    from huggingface_hub import HfApi
    con = connect()
    while True:
        batches = [upload_queue.get()]
        while not upload_queue.empty():
            batches.append(upload_queue.get())
        if any(l is None for l in batches):
            batches = [l for l in batches if l is not None]
            finished = True
        else:
            finished = False
        if batches:
            files = [a for l in batches for a in l["files"]]
            ranges = [f for l in batches for f in l["ranges"]]
            # snapshot of the ranges, including the ones this upload completes
            rg = pd.read_sql("SELECT * FROM faixas", con).set_index(["alvo", "a", "b"])
            for st, tot, fe, tg, a, b in ranges:
                if (tg, a, b) in rg.index:
                    rg.loc[(tg, a, b), ["status", "total", "obtidos"]] = [st, tot, fe]
            rg.reset_index().to_parquet(os.path.join(QUEUE_DIR, "estado/faixas.parquet"), index=False)
            n_occ, n_txt = sum(l["n"][0] for l in batches), sum(l["n"][1] for l in batches)
            t0, t = time.time(), 0
            while True:
                try:
                    HfApi().upload_folder(folder_path=QUEUE_DIR, repo_id=HF_REPO, repo_type="dataset",
                                          allow_patterns=files + ["estado/faixas.parquet"],
                                          commit_message=f"collection: {n_occ} occurrences, {n_txt} texts")
                    break
                except Exception as e:
                    t += 1
                    log(f"upload to the Hub failed ({type(e).__name__}: {str(e)[:100]}), attempt {t}")
                    time.sleep(min(600, 30 * t))
            for a in files:
                os.remove(os.path.join(QUEUE_DIR, a))
            con.executemany("UPDATE shas SET status='baixado' WHERE sha=?",
                            [(x,) for l in batches for x in l["shas"]])
            con.executemany("UPDATE faixas SET status=?, total=?, obtidos=?, atualizado=? WHERE alvo=? AND a=? AND b=?",
                            [(st, tot, fe, now(), tg, a, b) for st, tot, fe, tg, a, b in ranges])
            con.commit()
            counter["flush"] += 1
            log(f"upload {counter['flush']}: {n_occ} occurrences, {n_txt} texts, {len(ranges)} ranges, "
                f"{len(batches)} batches, {time.time() - t0:.0f}s")
        if finished:
            break
    con.close()


# ---------------------------------------------------------------- status

def status():
    con = connect()
    print(pd.read_sql("""
        SELECT alvo AS target,
               sum(status='pendente') AS pending, sum(status='feita') AS done,
               sum(status='truncada') AS truncated, sum(status='erro') AS errors,
               sum(obtidos) AS occurrences, sum(CASE WHEN status='truncada' THEN total-obtidos END) AS lost
        FROM faixas GROUP BY alvo ORDER BY min(rowid)""", con).to_string(index=False))
    print()
    print(pd.read_sql("SELECT status, count(*) AS shas FROM shas GROUP BY status", con).to_string(index=False))


def main():
    global TOKEN
    os.makedirs(STATE_DIR, exist_ok=True)
    if sys.argv[1:] == ["status"]:
        return status()
    TOKEN = read_token()
    con = connect()
    seed(con)
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: (log("stopping..."), stop.set()))
    # batches enqueued before a crash were not uploaded: their ranges went
    # back to pending in seed(), so the files are discarded
    for sub in ("ocorrencias", "textos", "estado"):
        os.makedirs(os.path.join(QUEUE_DIR, sub), exist_ok=True)
        for f in os.listdir(os.path.join(QUEUE_DIR, sub)):
            os.remove(os.path.join(QUEUE_DIR, sub, f))
    log("start")
    tu = threading.Thread(target=upload)
    tu.start()
    td = threading.Thread(target=download, daemon=True)
    td.start()
    discover()
    # discovery is over (or was stopped): wait for the download to drain the queue
    while td.is_alive():
        td.join(timeout=FLUSH_SECONDS)
        if td.is_alive():
            flush(con)
    flush(con, final=True)
    upload_queue.put(None)
    tu.join()
    # run.sh greps the last log line for " done$" to know the collection finished
    log("done" if not stop.is_set() else "interrupted; run again to resume")


if __name__ == "__main__":
    # any exception in discovery brings the whole process down; without this
    # the upload thread kept it alive and the supervisor did not restart it
    try:
        main()
    except Exception:
        import traceback
        log("Traceback in discovery: " + traceback.format_exc()[-600:])
        os._exit(1)
