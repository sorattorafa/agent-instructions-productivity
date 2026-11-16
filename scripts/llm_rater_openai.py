#!/usr/bin/env python3
"""
Third model rater for the validation of the productivity patterns: sends the
instructions of validation/llm/PROMPT.md, unchanged, and the items to an
OpenAI model through the Chat Completions API, in batches. The model never
sees the labels of the other raters.

Inputs:  validation/llm/PROMPT.md, validation/llm/{precision,recall}_items.jsonl
Outputs: validation/llm/openai/{precision,recall}.json  ([{"id": .., "answer": "y"|"n"}])
         validation/llm/openai/run.json                  (model id, batch sizes, date)
The API key is read from OPENAI_API_KEY or ~/.config/openai/key.
Usage: python scripts/llm_rater_openai.py [model]   (resumes if interrupted)
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

SRC = os.path.join("validation", "llm")
OUT = os.path.join(SRC, "openai")
MODEL = "gpt-5.5-2026-04-23"
BATCH = {"precision": 43, "recall": 10}
TASK = {"precision": "Do Task 1 (precision) for the items below.", "recall": "Do Task 2 (recall) for the items below."}


def key() -> str:
    k = os.environ.get("OPENAI_API_KEY")
    return k or open(os.path.expanduser("~/.config/openai/key")).read().strip()


def ask(model: str, text: str) -> list[dict]:
    body = {"model": model, "messages": [{"role": "user", "content": text}]}
    for attempt in range(8):
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key()}"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.load(r)
            t = d["choices"][0]["message"]["content"].strip()
            t = t[t.index("["):t.rindex("]") + 1]  # the JSON array, without a code fence around it
            return json.loads(t)
        except urllib.error.HTTPError as e:
            msg = e.read()[:200].decode(errors="replace")
            print(f"  HTTP {e.code} (attempt {attempt + 1}): {msg}", flush=True)
            if e.code not in (429, 500, 502, 503):
                raise
            time.sleep(40 * (attempt + 1))
        except (KeyError, ValueError, TimeoutError) as e:
            print(f"  bad answer (attempt {attempt + 1}): {type(e).__name__}", flush=True)
            time.sleep(10)
    sys.exit("the model did not answer; run again to resume")


def main(model: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    prompt = open(os.path.join(SRC, "PROMPT.md")).read()
    for task in ("precision", "recall"):
        items = [json.loads(l) for l in open(os.path.join(SRC, f"{task}_items.jsonl"))]
        path = os.path.join(OUT, f"{task}.json")
        done = {r["id"]: r["answer"] for r in json.load(open(path))} if os.path.exists(path) else {}
        todo = [i for i in items if i["id"] not in done]
        for k in range(0, len(todo), BATCH[task]):
            group = todo[k:k + BATCH[task]]
            text = (prompt + "\n\n---\n\n" + TASK[task] + " Items, one JSON object per line:\n\n"
                    + "\n".join(json.dumps(i, ensure_ascii=False) for i in group))
            ans = {int(a["id"]): str(a["answer"]).strip().lower()[:1] for a in ask(model, text)}
            ids = {i["id"] for i in group}
            if set(ans) != ids or not set(ans.values()) <= {"y", "n"}:
                sys.exit(f"{task}: answer does not cover the batch (missing {sorted(ids - set(ans))[:5]}); run again")
            done.update(ans)
            json.dump([{"id": i, "answer": done[i]} for i in sorted(done)], open(path, "w"))
            print(f"{time.strftime('%H:%M:%S')} {task} {len(done)}/{len(items)}", flush=True)
    json.dump({"model": model, "batch": BATCH, "date": time.strftime("%Y-%m-%d")},
              open(os.path.join(OUT, "run.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if sys.argv[1:] else MODEL)
