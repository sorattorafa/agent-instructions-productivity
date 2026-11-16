#!/usr/bin/env python3
"""
Dataset characterization (replication): origin of the 30 most copied contents of each artifact, by explicit
rules over observable evidence (no attribution by guesswork).

Skills: the name (front matter) coincides with a folder of a public
  collection (listings checked on 14 Nov 2026 on GitHub). A match of name,
  not of text: the copies may be old versions of the collection.
Context, in order:
  pointer       short text (< 300 characters) that only refers to AGENTS.md /
                CLAUDE.md (the whole file is the name, or "see AGENTS.md")
  generated     block written by a memory tool (claude-mem)
  empty         template with no content (only a Cursor rule header)
  platform      the text identifies itself with a framework or platform by
                a marker or title of its own (BEGIN:nextjs-agent-rules,
                LOVABLE:BEGIN, <laravel-boost-guidelines>, "Expo HAS CHANGED",
                figma-make-app, "Use Bun instead of Node.js"...)
  other         community template, or no attribution possible from the text
Input:  data/reuse/top_contents.csv
Output: data/reuse/origin_top.csv, origin_summary.csv
"""
import re

import pandas as pd

N = 30
COLLECTIONS = {
    "anthropics/skills": {"academy-guide", "algorithmic-art", "brand-guidelines", "canvas-design", "claude-api",
                          "discernment-nudge", "doc-coauthoring", "docx", "frontend-design", "internal-comms",
                          "mcp-builder", "pdf", "pptx", "skill-creator", "slack-gif-creator", "theme-factory",
                          "web-artifacts-builder", "webapp-testing", "xlsx"},
    "obra/superpowers": {"brainstorming", "diagnosing-superpowers", "dispatching-parallel-agents", "executing-plans",
                         "finishing-a-development-branch", "receiving-code-review", "requesting-code-review",
                         "subagent-driven-development", "systematic-debugging", "test-driven-development",
                         "using-git-worktrees", "using-superpowers", "verification-before-completion",
                         "writing-plans", "writing-skills"},
    "vercel-labs/agent-skills": {"composition-patterns", "deploy-to-vercel", "react-best-practices",
                                 "react-native-skills", "react-view-transitions", "vercel-cli-with-tokens",
                                 "vercel-optimize", "web-design-guidelines", "writing-guidelines"},
    "mattpocock/skills": {"grill-me"},
}
PLATFORMS = [
    ("Next.js", r"BEGIN:nextjs-agent-rules"),
    ("Lovable", r"LOVABLE:BEGIN"),
    ("Laravel Boost", r"<laravel-boost-guidelines>"),
    ("Expo", r"Expo HAS CHANGED|This is an Expo/React Native mobile application"),
    ("Figma Make", r"figma-make-app|running inside Figma Make"),
    ("Bun", r"Use Bun instead of Node\.js|Default to using Bun instead of Node\.js"),
    ("Astro", r"astro dev --background"),
    ("Cloudflare Workers", r"# Cloudflare Workers STOP"),
    ("Angular", r"You are an expert in TypeScript, Angular"),
    ("shadcn", r"# shadcn instructions"),
    ("HyperFrames", r"# HyperFrames Composition Project"),
]


def classify_context(text: str, n_chars: float) -> tuple[str, str]:
    t = text or ""
    if n_chars == n_chars and n_chars < 300 and re.search(r"(AGENTS|CLAUDE)\.md", t):
        return "pointer", "short and refers to another context file"
    if "<claude-mem-context>" in t:
        return "generated", "claude-mem"
    if re.fullmatch(r"\s*---\s*(description:\s*)?(globs:\s*)?(alwaysApply:\s*(true|false)\s*)?---\s*", t):
        return "empty", "rule header with no content"
    for skill_name, rx in PLATFORMS:
        if re.search(rx, t):
            return "platform", skill_name
    return "other", ""


def classify_skill(skill_name: str) -> tuple[str, str]:
    for col, skill_names in COLLECTIONS.items():
        if skill_name in skill_names:
            return "collection", col
    return "other", ""


def main() -> None:
    t = pd.read_csv("data/reuse/top_contents.csv")
    rows = []
    for art in ("SKILL.md", "(context, all)"):
        g = t[t.artifact == art].head(N).reset_index(drop=True)
        for i, r in g.iterrows():
            c, ev = classify_skill(str(r.skill_name)) if art == "SKILL.md" else classify_context(str(r.opening), r.n_chars)
            rows.append({"artifact": art, "position": i + 1, "sha": r.sha, "repos": r.repos,
                           "occurrences": r.occurrences, "skill_name": r.skill_name, "origin_class": c, "evidence": ev,
                           "opening": str(r.opening)[:160]})
    o = pd.DataFrame(rows)
    o.to_csv("data/reuse/origin_top.csv", index=False)
    res = o.groupby(["artifact", "origin_class", "evidence"]).agg(contents=("sha", "count"), repos=("repos", "sum")).reset_index()
    res.to_csv("data/reuse/origin_summary.csv", index=False)
    print(o[["artifact", "position", "repos", "origin_class", "evidence", "opening"]].to_string(index=False, max_colwidth=70))
    print()
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
