#!/usr/bin/env python3
"""
Generates paper/numbers.tex: every number cited in the paper comes from here.

Inputs (run from the repository root):
  data/reuse/summary.csv, data/name_filter.csv, data/size_ranges.parquet,
  data/productivity/productivity.csv, data/topics/*.csv, data/context/markers
Output: <repo>/paper/numbers.tex

Convention: a \\Nxxx macro is a count with a thousands separator; \\Pctxxx is
an already rounded percentage; \\Snapxxx is a dataset revision.
Usage: python scripts/synthesis.py
"""
from __future__ import annotations

import json
import os

import duckdb
import pandas as pd

import lexicon

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(REPO, "paper", "numbers.tex")
# revisions of the datasets on Hugging Face (checked on 13 Nov 2026)
SNAP = {"SnapSkills": "ebab174", "SnapContext": "9266e1c"}
# a content "dominates" a truncated size if it accounts for more than this
DOMINANT_SHARE = 1 / 3

macros: dict[str, str] = {}


def n(v) -> str:
    return f"{int(round(float(v))):,}".replace(",", "{,}")


def pct(v, decimals: int = 1) -> str:
    return f"{float(v):.{decimals}f}"


def put(name: str, value: str) -> None:
    assert name.isalpha(), name
    macros[name] = value


def corpus_numbers() -> None:
    for k, v in SNAP.items():
        put(k, v)
    r = pd.read_csv("data/reuse/summary.csv").set_index("artifact")
    sk, ct = r.loc["SKILL.md"], r.loc["(context, all)"]
    put("NSkillOcc", n(sk.occurrences)); put("NSkillDistinct", n(sk.distinct)); put("NSkillRepos", n(sk.repos))
    put("NCtxOcc", n(ct.occurrences)); put("NCtxDistinct", n(ct.distinct)); put("NCtxRepos", n(ct.repos))
    put("PctRedSkills", pct(sk.pct_redundant)); put("PctRedCtx", pct(ct.pct_redundant))
    put("PctUniqueSkills", pct(sk.pct_unique)); put("PctUniqueCtx", pct(ct.pct_unique))
    for target, name in (("AGENTS.md", "Agents"), ("CLAUDE.md", "Claude")):
        put(f"PctRed{name}", pct(r.loc[target].pct_redundant))
        put(f"NOcc{name}", n(r.loc[target].occurrences))
    put("NTargets", str(len(r) - 2))

    f = pd.read_csv("data/name_filter.csv")
    put("NFilterRemoved", n(f.removed.sum()))
    put("NFilterAgents", n(f.set_index("target").loc["AGENTS.md"].removed))

    con = duckdb.connect()
    con.sql("SET memory_limit = '1500MB'")
    tr = con.sql("SELECT target, a, total, fetched FROM 'data/size_ranges.parquet' WHERE status = 'truncated'").df()
    put("NSizeTrunc", str(len(tr)))
    put("NTruncated", n((tr.total - tr.fetched).sum()))
    # dominant content at each truncated size (size in characters of the text)
    dom = con.sql("""
      WITH t AS (SELECT target, a FROM 'data/size_ranges.parquet' WHERE status = 'truncated'),
           tam AS (SELECT DISTINCT sha, n_chars FROM read_parquet('data/context/markers/*.parquet')),
           x AS (SELECT o.target, m.n_chars, o.sha, count(*) AS c
                 FROM read_parquet('data/collection/occurrences.parquet') o
                 JOIN tam m USING (sha) JOIN t ON t.target = o.target AND t.a = m.n_chars GROUP BY ALL)
      SELECT target, n_chars, max(c) * 1.0 / sum(c) AS f FROM x GROUP BY 1, 2""").df()
    put("NSizeTruncDom", str(int((dom.f > DOMINANT_SHARE).sum())))

    put("NItemsSkills", str(len(lexicon.items("skills"))))
    put("NItemsCtx", str(len(lexicon.items("context"))))


# Labels of the NMF topics (k = 8): the ONLY judgment call in Dataset characterization (topics). Based on
# reading the 10 highest-weight documents and 10 drawn at random per topic
# (examples.csv), 14 Nov 2026. Topics 0 and 5 were renamed on 15 Nov 2026 after
# the typical phrases (phrases.py, phrases_by_topic.csv), approved by the
# authors; "MCP" was dropped from topic 5 because "mcp server" does not recur
# among the 30 typical phrases in the held-out sample.
LABELS = {
    0: "Architecture narrative (where code lives, what it does)",
    1: "Coding standards and conventions",
    2: "Non-coding tasks (marketing, research, analysis)",
    3: "UI and design systems",
    4: "Git and contribution workflow",
    5: "Environment and services setup (env vars, API keys)",
    6: "Agent operating rules (scope, precedence)",
    7: "Phased workflows (plan, execute, review)",
}
TABLE_TARGETS = ["AGENTS.md", "CLAUDE.md", ".claude/agents", ".claude/commands", ".cursor/rules",
             "copilot-instructions.md", ".claude/rules", ".github/instructions", "GEMINI.md",
             "memory-bank", ".cursorrules", ".windsurfrules", ".clinerules", "CLAUDE.local.md"]
NAMES = {"dora": "Dora", "space": "Space", "devex": "Devex", "dx_core4": "DxCore", "dev_productivity": "DevProd"}


def tex(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def corpus_table() -> None:
    r = pd.read_csv("data/reuse/summary.csv").set_index("artifact")
    q = pd.read_csv("data/productivity/productivity_by_target.csv").set_index("target")
    ln = pd.read_csv("data/reuse/file_lengths.csv").set_index("artifact").median_chars
    rows = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
              r"File & Occ. & Distinct & Redund.\,\% & Chars & Fw.\,\textperthousand \\", r"\midrule",
              rf"\code{{SKILL.md}} & {n(r.loc['SKILL.md'].occurrences)} & {n(r.loc['SKILL.md'].distinct)} & "
              rf"{pct(r.loc['SKILL.md'].pct_redundant)} & {n(ln['SKILL.md'])} & \FwPermilSkills \\", r"\midrule"]
    for a in TABLE_TARGETS:
        x = r.loc[a]
        fw = 10 * q.loc[a, "any framework/any"] if a in q.index else 0.0
        name = a.replace("copilot-instructions.md", "copilot-instructions")
        rows.append(rf"\code{{{tex(name)}}} & {n(x.occurrences)} & {n(x.distinct)} & {pct(x.pct_redundant)} & "
                    rf"{n(ln[a])} & {pct(fw, 2)} \\")
    c = r.loc["(context, all)"]
    rows += [r"\midrule", rf"All context files & {n(c.occurrences)} & {n(c.distinct)} & {pct(c.pct_redundant)} & "
                          rf"{n(ln['(context, all)'])} & \FwPermilCtx \\",
               r"\bottomrule",
               r"\multicolumn{6}{@{}l}{Redund.: occurrences that repeat an earlier content. Chars: median length.}\\",
               r"\multicolumn{6}{@{}l}{Fw.: distinct contents naming a framework or metric, per thousand.}",
               r"\end{tabular}"]
    open(os.path.join(REPO, "paper", "tab_corpus.tex"), "w").write("\n".join(rows) + "\n")
    put("MedCharsSkills", n(ln["SKILL.md"])); put("MedCharsCtx", n(ln["(context, all)"]))


N_PHRASES = 3
GENERIC_WORDS = __import__("phrases").GENERIC_WORDS


def topics_table() -> None:
    t = pd.read_csv("data/topics/topics.csv")
    rows = [r"\begin{tabular}{@{}p{0.62\columnwidth}rr@{}}", r"\toprule",
              r"Topic, with top terms and typical phrases & Skills & Context \\", r"\midrule"]
    t = t.assign(d=t.pct_skills - t.pct_context).sort_values("d", ascending=False)
    # typical phrases (phrases.py), explicit rule: the first N_PHRASES of the
    # fit that are also among the 30 typical ones of the held-out sample and
    # that are not a piece of the /init header (a tool's sentence, not subject
    # matter); a phrase is dropped if it ends in a generic word ("unless the")
    # or if it contains or is contained in one already chosen ("unless the user")
    fit_ph = pd.read_csv("data/topics/phrases_by_topic.csv")
    va = pd.read_csv("data/topics/heldout/phrases_by_topic.csv")
    init = "this file provides guidance to claude code claude ai code when working with code in this repository"
    expr = {}
    for i, g in fit_ph.sort_values("position").groupby("topic"):
        ok = set(va[va.topic == i].phrase)
        chosen = []
        for e in g.phrase:
            if e not in ok or e in init or e.split()[-1] in GENERIC_WORDS:
                continue
            if any(e in c or c in e for c in chosen):
                continue
            chosen.append(e)
        expr[i] = chosen[:N_PHRASES]
    for _, x in t.iterrows():
        terms = ", ".join(str(x.terms).split(", ")[:4])
        ex = "; ".join(expr[int(x.topic)])
        # topic label on its own line; terms and phrases indented under it, smaller and grey
        child = (rf"\par{{\scriptsize\color{{black!65}}\raggedright\leftskip=1em\parskip=0pt "
                 rf"\textit{{{tex(terms)}}}\par``{tex(ex)}''\par}}\vspace{{-\baselineskip}}")
        rows.append(rf"{LABELS[int(x.topic)]}{child} & {pct(x.pct_skills)}\% & {pct(x.pct_context)}\% \\[-6pt]")
    rows += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(REPO, "paper", "tab_topics.tex"), "w").write("\n".join(rows) + "\n")
    d = __import__("json").load(open("data/topics/diagnostics.json"))
    put("NTopicK", str(d["k"]))
    for art, suf in (("skill", "Sk"), ("context", "Ct")):
        base = d["raw_sample"][art] - d["short"][art] - d["non_english"][art]
        put(f"PctNearDup{suf}", pct(100 * d["near_duplicates"][art] / base))
    put("NTopicDocs", n(d["matrix"][0]))
    put("NTopicPerArt", n(d["balanced_per_artifact"]))
    put("NOwnerCap", n(sum(d["over_owner_cap"].values())))
    put("NOwnerCapSkills", n(d["over_owner_cap"]["skill"]))
    put("StabMin", f"{min(d['min_stability_cosine_by_topic'].values()):.3f}")
    v = __import__("json").load(open("data/topics/heldout/summary.json"))
    put("NValSk", n(v["heldout_final"]["skill"])); put("NValCt", n(v["heldout_final"]["context"]))
    put("ValMaxDif", f"{v['max_diff_pp']:.1f}")
    put("ValTopMin", str(min(v["shared_top30_by_topic"].values())))
    put("ValTopMax", str(max(v["shared_top30_by_topic"].values())))
    e = pd.read_csv("data/topics/k_selection.csv").set_index("k").loc[d["k"]]
    put("LargestGroup", pct(100 * e.largest_group))
    tp = pd.read_csv("data/topics/topics_by_target.csv").set_index("target")
    put("PctAgentsRules", pct(tp.loc["AGENTS.md", "6"])); put("PctClaudeRules", pct(tp.loc["CLAUDE.md", "6"]))
    put("PctClaudeSetup", pct(tp.loc["CLAUDE.md", "5"])); put("PctAgentsSetup", pct(tp.loc["AGENTS.md", "5"]))
    for i in LABELS:
        x = t.set_index("topic").loc[i]
        put(f"PctTopic{'ABCDEFGH'[i]}Sk", pct(x.pct_skills)); put(f"PctTopic{'ABCDEFGH'[i]}Ct", pct(x.pct_context))


def origin() -> None:
    o = pd.read_csv("data/reuse/origin_top.csv")
    sk, ct = o[o.artifact == "SKILL.md"], o[o.artifact != "SKILL.md"]
    put("NTop", str(len(sk)))
    put("NTopSkCollection", str((sk.origin_class == "collection").sum()))
    for col, name in (("anthropics/skills", "Anthropic"), ("obra/superpowers", "Superpowers")):
        put(f"NTopSk{name}", str((sk.evidence == col).sum()))
    for cl, name in (("platform", "Platform"), ("pointer", "Pointer"), ("generated", "Generated"),
                     ("empty", "Empty"), ("other", "Other")):
        put(f"NTopCt{name}", str((ct.origin_class == cl).sum()))
    put("NTopCtPlatforms", str(ct[ct.origin_class == "platform"].evidence.nunique()))
    put("NTopRepoMaxSk", n(sk.repos.max())); put("NTopRepoMaxCt", n(ct.repos.max()))


def markers() -> None:
    con = duckdb.connect(); con.sql("SET memory_limit = '1500MB'")
    con.sql("CREATE TABLE oc AS SELECT DISTINCT target, sha FROM 'data/context_occurrences.parquet'")
    con.sql("CREATE TABLE mk AS SELECT DISTINCT * EXCLUDE (n_chars) FROM read_parquet('data/context/markers/*.parquet') WHERE sha IN (SELECT sha FROM oc)")
    cols = [c for c in con.sql("SELECT * FROM mk LIMIT 0").columns if c != "sha"]
    for c in cols:
        k = con.sql(f"SELECT count(*) FROM mk WHERE {c}").fetchall()[0][0]
        put("NMark" + "".join(w.capitalize() for w in c.split("_")), n(k))
    tot = con.sql("SELECT count(*) FROM oc WHERE target = 'CLAUDE.md'").fetchall()[0][0]
    init_n = con.sql("SELECT count(*) FROM oc JOIN mk USING (sha) WHERE target = 'CLAUDE.md' AND init_claude").fetchall()[0][0]
    put("PctInitClaude", pct(100 * init_n / tot))


def productivity() -> None:
    r = pd.read_csv("data/productivity/productivity.csv")
    for corpus, suf in (("skills", "Sk"), ("context", "Ct")):
        g = r[r.corpus == corpus].set_index("group")
        for p, name in NAMES.items():
            put(f"NMen{name}{suf}", n(g.loc[p].texts))
        put(f"PctMenDevProd{suf}", pct(g.loc["dev_productivity"].pct, 2))
        for fw in ("DORA", "SPACE", "DevEx"):
            put(f"N{fw.capitalize()}Any{suf}", n(g.loc[f"{fw}/any"].texts))
            put(f"N{fw.capitalize()}Metric{suf}", n(g.loc[f"{fw}/metric"].texts))
        a = g.loc["any framework/any"]
        put(f"NAny{suf}", n(a.texts)); put(f"NAnyRepos{suf}", n(a.repos))
        put(f"PctAny{suf}", pct(a.pct, 2))
        put(f"FwPermil{'Skills' if suf == 'Sk' else 'Ctx'}", pct(10 * a.pct, 2))
        mv, mt = g.loc["metric with value"].texts, g.loc["any metric"].texts
        put(f"NMetricValue{suf}", n(mv)); put(f"NMetricAny{suf}", n(mt)); put(f"PctMetricValue{suf}", pct(100 * mv / mt, 0))
        # copies among the occurrences of the contents that mention measurement
        put(f"NOccAny{suf}", n(a.occurrences))
        put(f"PctRedAny{suf}", pct(100 * (a.occurrences - a.texts) / a.occurrences))
        # split of the texts that count: framework named only, metric only, both
        for grp, name in (("named only", "NamedOnly"), ("metric only", "MetricOnly"),
                          ("named and metric", "NamedMetric")):
            put(f"N{name}{suf}", n(g.loc[grp].texts))
            put(f"Pct{name}{suf}", pct(100 * g.loc[grp].texts / a.texts, 0))
        dims = g[(g.dimension.notna()) & (~g.dimension.isin(["(any)"])) & (g.measure != "mention")]
        put(f"NDimNotMeasured{suf}", str((dims.measure == "not measured").sum()))
    put("NDimTotal", str(len(lexicon.DIMENSIONS)))
    gs = r[r.corpus == "skills"].set_index("group")
    for d, name in (("DORA/change failure rate", "Cfr"), ("DORA/lead time for changes", "LeadTime"),
                    ("SPACE/communication and collaboration", "Comm"), ("DevEx/cognitive load", "Cog"),
                    ("DevEx/feedback loops", "Feedback")):
        put(f"NDim{name}Sk", n(gs.loc[d].texts))
    gc = r[r.corpus == "context"].set_index("group")
    for d, name in (("DORA/change failure rate", "Cfr"), ("DORA/lead time for changes", "LeadTime"),
                    ("SPACE/communication and collaboration", "Comm")):
        put(f"NDim{name}Ct", n(gc.loc[d].texts))
    q = pd.read_csv("data/productivity/productivity_by_target.csv").set_index("target")
    put("PermilClaudeAgentsDir", pct(10 * q.loc[".claude/agents", "any framework/any"], 2))
    put("PermilAgentsMd", pct(10 * q.loc["AGENTS.md", "any framework/any"], 2))
    put("PermilClaudeMd", pct(10 * q.loc["CLAUDE.md", "any framework/any"], 2))


def stats_macros() -> None:
    """Macros from statistical_tests.py (Wilson CI, rate ratio, chi-square)."""
    e = __import__("json").load(open("data/statistics/summary.json"))
    f1 = lambda x: f"{x:.1f}"
    put("NItemsDropped", str(len(lexicon.REMOVED)))
    r = e["productivity"]["skills_vs_context"]
    put("RRSkCt", f1(r["ratio"])); put("RRSkCtLo", f1(r["ci95"][0])); put("RRSkCtHi", f1(r["ci95"][1]))
    r = e["productivity"]["claude_agents_vs_agents_md"]
    put("RRAgDir", f1(r["ratio"])); put("RRAgDirLo", f1(r["ci95"][0])); put("RRAgDirHi", f1(r["ci95"][1]))
    t = e["topics"]["topic_by_artifact"]
    put("ChiTopicArt", n(round(t["chi2"]))); put("DofTopicArt", str(t["dof"])); put("VTopicArt", f"{t['cramers_v']:.2f}")
    put("ResSkNonCoding", f1(t["skill_residual_by_topic"]["2"])); put("ResSkPhased", f1(t["skill_residual_by_topic"]["7"]))
    t = e["topics"]["topic_by_agents_claude"]
    put("ChiTopicAC", n(round(t["chi2"]))); put("DofTopicAC", str(t["dof"])); put("VTopicAC", f"{t['cramers_v']:.2f}")
    put("ResAgRules", f1(t["agents_residual_by_topic"]["6"]))
    put("HalfCITopics", f1(e["topics"]["max_ci95_half_width_pp"]))


def overlap() -> None:
    """Macros from overlap.py: repositories present in both datasets."""
    o = pd.read_csv("data/overlap/overlap.csv").set_index("measure").repos
    put("NReposBoth", n(o["repositories with both"]))
    put("NReposEither", n(o["repositories with either"]))
    r = pd.read_csv("data/reuse/summary.csv").set_index("artifact")
    put("NFilesBothDatasets", n(r.loc["SKILL.md"].occurrences + r.loc["(context, all)"].occurrences))
    put("PctSkillReposWithCtx", pct(o["pct of skill repositories that also have a context file"]))
    put("PctCtxReposWithSkill", pct(o["pct of context-file repositories that also have a skill"]))
    t = pd.read_csv("data/overlap/overlap_by_target.csv").set_index("target")
    put("PctClaudeRulesWithSkill", pct(t.loc[".claude/rules", "pct_with_skills"]))
    put("PctAgentsMdWithSkill", pct(t.loc["AGENTS.md", "pct_with_skills"]))
    k = pd.read_csv("data/overlap/overlap_naming.csv").iloc[0]
    in_ctx = k.context_only + k.in_both
    put("NNamingCtxRepos", n(k.repos_naming_in_context_all)); put("NNamingCtxReposBoth", n(in_ctx))
    put("PctNamingCtxReposBoth", pct(100 * in_ctx / k.repos_naming_in_context_all, 0))
    put("NNamingBothArtifacts", n(k.in_both))


def masking() -> None:
    """Macros from export_dataset.py: what was masked in the released texts."""
    m = __import__("json").load(open("data/dataset/masking.json"))
    put("NReleaseTexts", n(m["distinct_texts"]))
    put("NMaskedEmails", n(m["emails"])); put("NMaskedTokens", n(m["secrets"]))
    put("PctTextsWithEmail", pct(100 * m["texts_with_email"] / m["distinct_texts"]))


def phrases_table() -> None:
    """Table of the phrases most characteristic of each artifact (phrase_contrast.py).
    Shares over every distinct content when the count stage has run, otherwise the
    shares in the sample."""
    full = os.path.exists("data/topics/phrase_counts.csv")
    d = pd.read_csv("data/topics/phrase_counts.csv" if full else "data/topics/phrase_contrast.csv")
    rows = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", r"Phrase & Skills & Context \\", r"\midrule"]
    for art, head in (("skill", "More frequent in skills"), ("context", "More frequent in context files")):
        rows.append(rf"\multicolumn{{3}}{{@{{}}l}}{{\emph{{{head}}}}} \\")
        for _, x in d[d.artifact == art].sort_values("rank").iterrows():
            rows.append(rf"\quad {tex(x.phrase)} & {pct(x.pct_skills, 2 if full else 1)}\% & {pct(x.pct_context, 2 if full else 1)}\% \\")
        if art == "skill":
            rows.append(r"\midrule")
    rows += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(REPO, "paper", "tab_phrases.tex"), "w").write("\n".join(rows) + "\n")
    put("PhraseScope", "every distinct content of each dataset" if full else "the topic-model sample")
    j = __import__("json").load(open("data/topics/phrase_contrast.json"))
    put("PctInitSample", pct(j["pct_context_with_init_header"]))
    put("NPhrasesPerArt", str(int((d.artifact == "skill").sum())))


def loose() -> None:
    """Macros from loose_patterns.py: ceiling given by unqualified expressions."""
    d = pd.read_csv("data/productivity/loose_patterns.csv")
    delivery = ["dora", "mttr", "deployment_frequency", "lead_time", "cycle_time", "change_failure"]
    for c, suf in (("skills", "Sk"), ("context", "Ct")):
        g = d[d.corpus == c].set_index("expression")
        put(f"LooseAny{suf}", pct(g.loc["any"].pct, 2)); put(f"LooseKpi{suf}", pct(g.loc["kpi"].pct, 2))
        put(f"LooseDevExp{suf}", pct(g.loc["developer_experience"].pct, 2))
        put(f"LooseMaxDelivery{suf}", pct(g.loc[delivery].pct.max(), 2))


def model_raters() -> None:
    """Macros from llm_agreement.py: three model raters on every pattern match
    (Task 1) and on files with no match (Task 2)."""
    d = json.load(open("data/validation/llm_summary.json"))
    for task, suf in (("precision", "Prec"), ("recall", "Rec")):
        t = d[task]
        ys = sorted(v["y"] for v in t["share_y"].values())
        put(f"Val{suf}Items", f"{t['items']:,}")
        put(f"Val{suf}Majority", f"{t['majority_y']['y']:,}"); put(f"Val{suf}MajorityPct", pct(100 * t["majority_y"]["p"]))
        put(f"Val{suf}All", f"{t['all_y']['y']:,}"); put(f"Val{suf}AllPct", pct(100 * t["all_y"]["p"]))
        put(f"Val{suf}Agree", pct(100 * t["all_agree"])); put(f"Val{suf}Fleiss", pct(t["fleiss_kappa"], 2))
        put(f"Val{suf}MinY", f"{ys[0]:,}"); put(f"Val{suf}MaxY", f"{ys[-1]:,}"); put(f"Val{suf}MidY", f"{ys[1]:,}")
        ks = [v["kappa"] for v in t["pairs"].values()]
        put(f"Val{suf}CohenMin", pct(min(ks), 2)); put(f"Val{suf}CohenMax", pct(max(ks), 2))


def languages() -> None:
    """Macros from languages.py: share of English and of Chinese among the
    texts counted in the productivity analysis and in the random sample of each corpus."""
    d = pd.read_csv("data/languages/languages.csv")
    for c, suf in (("skills", "Sk"), ("context", "Ct")):
        g = d[d.corpus == c]
        share = lambda grp, lang: pct(g[(g.group == grp) & (g.language == lang)].pct.sum())
        put(f"LangEnRq{suf}", share("productivity", "en")); put(f"LangEnAll{suf}", share("corpus", "en"))
        put(f"LangZhRq{suf}", share("productivity", "zh")); put(f"LangZhAll{suf}", share("corpus", "zh"))


def write_macros() -> None:
    with open(OUTPUT, "w") as fh:
        fh.write("% generated by synthesis.py; do not edit by hand\n")
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"{len(macros)} macros in {OUTPUT}")
    for k, v in macros.items():
        print(f"  {k:22} {v}")


if __name__ == "__main__":
    corpus_numbers()
    origin()
    markers()
    productivity()
    corpus_table()
    topics_table()
    stats_macros()
    overlap()
    masking()
    phrases_table()
    loose()
    model_raters()
    languages()
    write_macros()
