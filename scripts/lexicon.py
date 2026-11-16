#!/usr/bin/env python3
"""
Metric vocabulary: links the text of the .md files to the dimensions of DORA,
SPACE and DevEx through the NAMES OF THE METRICS that the frameworks
themselves define.

Version 2 (12 Nov 2026). Version 1, which mapped the "task territory"
(deploy, test, documentation...) to the dimensions, is an earlier version,
not part of this package; it was dropped from the paper by the authors'
decision: it had no independent validation and its patterns were not derived
from the anchors.

The provenance chain
--------------------
Each item below links three things, all checked by machine:

  1. FRAMEWORK ANCHOR -- a VERBATIM passage from the text in which the
     framework itself lists the metric as an example:
        DORA  -> official DORA guide (source G001 of the authors' multivocal
                 literature review (MLR))
        SPACE -> Forsgren et al. 2021, per-dimension sections and Figure 1 (P1259)
        DevEx -> Noda et al. 2023, Table 1 "Example DevEx metrics" (P1280)
     `verify()` rereads the texts and fails if a passage is not there.
  2. CANONICAL METRIC OF THE MLR -- the name of the metric in the inventory
     of 62 metrics of the authors' multivocal literature review (228
     sources), in data/master/metric_inventory.csv. It shows that the metric
     appears in the literature and in how many sources. `verify()` fails if
     the name disappears.
  3. PATTERN -- the expression that is searched for in the .md file.

The two criteria that decide what becomes a pattern (authors' decision, 12 Nov 2026)
------------------------------------------------------------------------------------
  (a) Only an expression that NAMES A QUANTITY counts: a measurement core
      such as rate, time, frequency, count, number of, duration, turnaround.
      "deployment frequency" counts; "deploy" does not. Verbs and objects of
      instructions ("run the tests", "commit the changes") are left out,
      because they say what the skill does and not what is measured. The
      only exception is construct names that the framework lists as a
      metric and that have no instruction sense (burnout).
  (b) The dimension comes from the framework, never from our own reading. A
      single pattern counts for two dimensions only when the frameworks
      themselves list the metric in both (e.g. code review turnaround in
      SPACE efficiency and flow and in DevEx feedback loops). Cycle time
      does NOT go into lead time for changes: the MLR groups the two under
      Delivery, but DORA defines only lead time.

Precision check (12 Nov 2026). The first pass over 6 of the 31 shards of the
corpus was read through a sample of snippets, item by item. Patterns that
matched another sense (inventory lead time, uptime as the Linux command,
marketing NPS, "number of commits" as a git operation) were restricted to
the measurement form, and two items were dropped; each case is commented on
the item. The final precision is measured again, by sample, over the full
scan.

The names of the dimensions ("cognitive load", "flow state", "feedback loop")
are not metrics and are not included here: mentions of the framework are a
separate measure (corpus_pass.py, productivity.py).

The current DORA has five metrics (G001): "deployment rework rate" was added
to instability. The four classic dimensions were kept, as in the MLR, and
rework rate counts together with change failure rate.

Usage:
  python scripts/lexicon.py          # checks the lexicon
  python scripts/lexicon.py export   # writes data/lexicon/provenance.csv, needs MLR_ROOT

The anchor passages and canonical metrics are re-checked only when the
environment variable MLR_ROOT points to the review data; without it the
anchor check is skipped and the frozen provenance table
data/lexicon/provenance.csv documents the result.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

# The authors' multivocal literature review (MLR; Oct 2026 version, 228
# sources). Outside this repository; its location is given by the environment
# variable MLR_ROOT.
MLR = Path(os.environ["MLR_ROOT"]) if os.environ.get("MLR_ROOT") else None
PROVENANCE_CSV = Path(__file__).resolve().parent.parent / "data" / "lexicon" / "provenance.csv"
INVENTORY = MLR / "data" / "master" / "metric_inventory.csv" if MLR else None
SOURCE_TEXTS = {
    "G001": MLR / "pdfs" / "grey_literature" / "G001_DORA_Metrics_Guide.txt",
    "P1259": next((MLR / "pdfs" / "selecionados" / "Google_Scholar").glob("P1259_*.pdf"), None),
    "P1280": MLR / "pdfs" / "selecionados" / "Google_Scholar" / "P1280_3595878.pdf",
} if MLR else {}
REFERENCE = {
    "G001": "DORA, \"DORA's software delivery performance metrics\" (dora.dev)",
    "P1259": "Forsgren et al., \"The SPACE of Developer Productivity\", ACM Queue 2021",
    "P1280": "Noda et al., \"DevEx: What Actually Drives Productivity\", ACM Queue 2023",
}

DIMENSIONS = [
    ("DORA", "deployment frequency"), ("DORA", "lead time for changes"),
    ("DORA", "change failure rate"), ("DORA", "time to restore service"),
    ("SPACE", "satisfaction and well-being"), ("SPACE", "performance"),
    ("SPACE", "activity"), ("SPACE", "communication and collaboration"),
    ("SPACE", "efficiency and flow"),
    ("DevEx", "feedback loops"), ("DevEx", "cognitive load"), ("DevEx", "flow state"),
]
FRAMEWORKS = ("DORA", "SPACE", "DevEx")

# ---------------------------------------------------------------------------
# key       -- short identifier of the item (becomes a column in the output)
# pattern   -- RE2 regex, applied case-insensitively and with word boundaries
# mlr       -- canonical metric(s) from the MLR inventory (None = the framework
#              lists the metric but the MLR does not have it as canonical)
# anchors   -- (framework, dimension, source, verbatim passage from the source)
# ---------------------------------------------------------------------------
ITEMS: list[dict] = [
    # ----------------------------- DORA ------------------------------------
    {"key": "deploy_freq",
     "pattern": r"deploy(?:ment)? frequency|frequency of (?:deploy(?:ment)?s|releases)"
               r"|release frequency|deploy(?:ment)?s per (?:day|week|month)",
     "mlr": ["deployment frequency"],
     "anchors": [("DORA", "deployment frequency", "G001",
                  "Deployment frequency: The number of deployments over a given period")]},
    {"key": "lead_time",
     # bare "lead time" matches inventory, purchasing and record releases; it
     # only counts with a software-delivery qualifier.
     "pattern": r"lead[ -]time for changes|change lead[ -]time"
               r"|(?:deploy(?:ment)?|commit|pr|pull request|merge request|code|ci|cd"
               r"|engineering|development) lead[ -]time"
               r"|lead[ -]time (?:to|for) (?:production|deploy(?:ment)?|merge)",
     "mlr": ["lead time for changes"],
     "anchors": [("DORA", "lead time for changes", "G001",
                  "Change lead time: The amount of time it takes for a change"),
                 ("DevEx", "feedback loops", "P1280", "Deployment lead time")]},
    {"key": "change_fail",
     "pattern": r"change[ -]fail(?:ure)?[ -]rate|deployment (?:failure|rework) rate"
               r"|(?:failed|failing) deployments? rate",
     "mlr": ["change failure rate"],
     "anchors": [("DORA", "change failure rate", "G001",
                  "Change fail rate: The ratio of deployments that require immediate intervention"),
                 ("DORA", "change failure rate", "G001",
                  "Deployment rework rate: The ratio of deployments that are unplanned")]},
    {"key": "restore",
     # "Indisputable" criterion (12 Nov 2026): bare "MTTR" and "mean time to
     # repair/resolve/respond" were also vulnerability remediation time;
     # only the names of the DORA metric were kept.
     "pattern": r"mean[ -]time[ -]to[ -](?:restore|recover|recovery)"
               r"|time to restore service|failed deployment recovery time",
     "mlr": ["mean time to restore"],
     "anchors": [("DORA", "time to restore service", "G001",
                  "Failed deployment recovery time: The time it takes to recover")]},

    # ----------------------------- SPACE -----------------------------------
    {"key": "satisfaction",
     # "employee/job satisfaction" came from HR and compliance in general.
     # "team satisfaction" was team management in any field.
     "pattern": r"(?:developer|engineer|engineering) satisfaction",
     "mlr": ["satisfaction"],
     "anchors": [("SPACE", "satisfaction and well-being", "P1259",
                  "Employee satisfaction. The degree of satisfaction among employees")]},
    {"key": "burnout",
     # bare "burnout" matches a burnt-out resistor and anyone's mental
     # health; only burnout of developers, or as an indicator, counts.
     "pattern": r"(?:developer|engineer|engineering) burn[- ]?out",
     "mlr": ["burnout / stress / frustration"],
     "anchors": [("SPACE", "satisfaction and well-being", "P1259",
                  "Burnout. Exhaustion caused by excessive and prolonged workplace stress")]},
    {"key": "retention",
     # "employee retention" and "turnover rate" came from due diligence and
     # financial markets; only retention of developers counts.
     # "engineer retention" is a verb in video scripts ("engineer retention
     # inside a piece of content"); "developer retention" is user churn in
     # DevRel. Only turnover of the engineering team was kept.
     "pattern": r"engineering (?:team )?(?:attrition|turnover|retention)"
               r"|(?:engineer|developer) (?:attrition|turnover)",
     "mlr": ["engagement / motivation / retention"],
     "anchors": [("SPACE", "satisfaction and well-being", "P1259", "zRetention")]},
    {"key": "uptime",
     "removed": "service uptime and SLA contracts measure the system, not the work of developing",
     # bare "uptime" is the Linux command or the age of a container; only
     # service availability as a number or a goal counts.
     "pattern": r"(?:service|system|platform|application|app|api|production) uptime"
               r"|uptime (?:sla|slo|target|percentage|percent|metric|metrics|goal)",
     "mlr": ["incidents / instability"],
     "anchors": [("SPACE", "performance", "P1259", "Reliability (uptime)")]},
    {"key": "defects",
     # "bug count" was a count of bugs within a task (triage, exercise).
     # "defect rate/density" and "bug rate" were also about factories and suppliers.
     "pattern": r"escaped defects|defect escape rate|(?:bugs|defects) per (?:release|sprint|kloc)",
     "mlr": ["defects / rollbacks"],
     "anchors": [("SPACE", "performance", "P1259",
                  "Quality. Reliability, absence of bugs, ongoing service health")]},
    # The "impact" item (customer satisfaction, NPS, CSAT, feature usage --
    # SPACE performance, "Impact") was dropped in the 12 Nov 2026 check: in
    # the snippets it came from marketing and customer-support skills, where
    # it measures the business and not the work of developing.
    {"key": "activity_count",
     # "number of commits" and "commit count" came from git operations and
     # changelogs (counting what is about to be merged); only the count per
     # period or per person counts, which is the measurement form.
     "pattern": r"commits per (?:day|week|month|sprint|developer|engineer|author|contributor)"
               r"|(?:prs|pull requests|merge requests) (?:merged |opened )?per"
               r" (?:day|week|month|sprint|developer|engineer)"
               r"|(?:pr|pull request) throughput"  # "commit frequency" was agent configuration
               r"|lines of code (?:changed|added|written|deleted) per"
               r"|story points (?:completed|delivered|shipped) per|story points per sprint",
     "mlr": ["commit count", "pull-request count / time", "lines of code changed",
             "build count / time", "story points completed"],
     "anchors": [("SPACE", "activity", "P1259",
                  "Volume or count of design documents and specs, work items, pull requests, "
                  "commits, and code reviews"),
                 ("SPACE", "activity", "P1259", "Count of build, test, deployment/release")]},
    {"key": "integration",
     # "review quality" was almost always a quality gate of the skill
     # itself, not a measure; only integration time was kept.
     # "at PR merge time" (the instant) and "time to merge" (verb) were
     # common; only the merge-time statistic counts.
     "pattern": r"(?:median|average|avg|mean|p50|p90) (?:pr |pull request )?(?:merge time|time to merge)",
     "mlr": ["code-review interaction"],
     "anchors": [("SPACE", "communication and collaboration", "P1259",
                  "How quickly work is integrated"),
                 ("SPACE", "communication and collaboration", "P1259",
                  "Quality of reviews of work contributed by team members"),
                 ("SPACE", "communication and collaboration", "P1259", "PR merge times")]},
    {"key": "onboarding_time",
     # "ramp-up time" was about load testing; dropped.
     # "developer onboarding time" was DevRel (onboarding of SDK users).
     "pattern": r"(?:engineer|new hire|contributor) onboarding time"
               r"|time to (?:first|1st) (?:commit|pr|pull request|merged pr)",
     "mlr": ["onboarding / mentoring"],
     "anchors": [("SPACE", "communication and collaboration", "P1259",
                  "Onboarding time for and experience of new members")]},
    {"key": "handoffs",
     "removed": "\"number of handoffs\" was also session handover between agents and logistics",
     # "handoff rate/count" was about chatbots (handover to a human agent).
     "pattern": r"number of hand[- ]?offs",
     "mlr": None,
     "anchors": [("SPACE", "efficiency and flow", "P1259", "Number of handoffs in a process")]},
    {"key": "interruptions",
     # bare "interruption rate/frequency" was about voice and dialogue AI.
     "pattern": r"(?:developer|engineer|team) interruptions"
               r"|(?:interruptions|context switch(?:es)?) per (?:day|hour|week)",
     "mlr": ["interruptions / context switches"],
     "anchors": [("SPACE", "efficiency and flow", "P1259", "Interruptions: quantity, timing"),
                 ("DevEx", "flow state", "P1280",
                  "Perceived ability to focus and avoid interruptions")]},
    {"key": "focus_time",
     "removed": "deep work/focus time came from personal-productivity skills, not from development",
     # "block focus time" is a calendar instruction, not a measure; only
     # focus time as a quantity counts.
     "pattern": r"(?:total|daily|weekly|average|hours of) focus time"
               r"|focus time (?:per|hours|ratio|percentage|metric)"
               r"|deep work hours",  # "hours of focused work" was task size
     "mlr": ["flow state / focus time"],
     "anchors": [("SPACE", "efficiency and flow", "P1259", "uninterrupted focus time"),
                 ("DevEx", "flow state", "P1280",
                  "Number of blocks of time without meetings or interruptions")]},
    {"key": "wait_time",
     # "queue wait time" was about workflow engines; dropped.
     "pattern": r"(?:review|ci|pr|build) wait(?:ing)? time",  # "value-added time" was operations in general  # "flow efficiency" was about UX
     "mlr": ["feedback loops"],
     "anchors": [("SPACE", "efficiency and flow", "P1259", "Time measures through a system")]},
    {"key": "review_turnaround",
     # bare "time to review" and "pickup time" were everyday language (and
     # car rental bookings); they only count with explicit code review.
     "pattern": r"(?:code|pr|pull request) review turnaround|review turnaround time"
               r"|time to first review|(?:code|pr) review (?:latency|response time)"
               r"|(?:pr|pull request) pickup time",
     "mlr": ["code-review interaction"],
     # Figure 1 of SPACE lists "Code review timing" under efficiency and
     # flow, but the cell comes out scrambled in the PDF extraction and
     # cannot be checked by machine; only the DevEx anchor is kept.
     "anchors": [("DevEx", "feedback loops", "P1280", "Code review turnaround time")]},
    {"key": "unplanned",
     "pattern": r"frequency of unplanned|unplanned (?:work|tasks|requests) (?:rate|ratio|percentage|frequency)",
     "mlr": ["work-mix balance (planned vs unplanned)"],
     "anchors": [("DevEx", "flow state", "P1280", "Frequency of unplanned tasks or requests")]},

    # ----------------------------- DevEx -----------------------------------
    {"key": "ci_time",
     "pattern": r"time to (?:generate )?ci results|ci (?:duration|run ?time|wall[- ]?clock time|feedback time)"
               r"|(?:ci|cd|build|deploy(?:ment)?) pipeline (?:duration|run ?time)"
               r"|build duration"  # "build times/speed" was also compile-time
               # "test duration" was about A/B testing
               r"|test suite (?:duration|run ?time)|test execution time"
               r"|feedback loop (?:time|speed|latency)",
     "mlr": ["feedback loops", "build count / time"],
     "anchors": [("DevEx", "feedback loops", "P1280", "generate CI results"),
                 ("DevEx", "feedback loops", "P1280", "automated test speed")]},
    # The "codebase complexity" item (perceived complexity of codebase, ease
    # of debugging) was dropped in the 12 Nov 2026 check: "codebase
    # complexity" appeared as a factor for sizing the task, not as a measure.
    {"key": "knowledge_friction",
     # "documentation improvements" was a changelog/task category.
     # bare "time to answer" was everyday language.
     "pattern": r"time to (?:get )?answers? to (?:technical|developer|engineering) questions"
               r"|manual steps (?:required )?to deploy"
               r"|number of manual steps|frequency of documentation improvements",
     "mlr": ["DevEx driver items (survey)"],
     "anchors": [("DevEx", "cognitive load", "P1280",
                  "Time it takes to get answers to technical questions"),
                 ("DevEx", "cognitive load", "P1280", "Manual steps required to deploy a change"),
                 ("DevEx", "cognitive load", "P1280", "Frequency of documentation improvements")]},
    {"key": "incident_freq",
     # "safety incident rate" (construction, industry) is excluded by the qualifier.
     "pattern": r"(?:production|service|software|outage|sev[0-9]?|on[- ]?call) incidents? (?:count|frequency|rate)"
               # "incident count" and "number of incidents" were also security
               # and suppliers; production incidents and on-call were kept.
               r"|number of production incidents"
               r"|on[- ]?call (?:load|burden|disruption|interruptions)",
     "mlr": ["incidents / instability"],
     "anchors": [("DevEx", "flow state", "P1280", "Frequency of incidents requiring team attention"),
                 ("DevEx", "flow state", "P1280", "Perceived disruptiveness of being on-call"),
                 ("SPACE", "activity", "P1259", "Count or volume of incidents")]},
]


# Patterns dropped after reading samples of their matches (12 Nov 2026):
# the expression often carried another sense, given next to each item.
DROPPED_ROUND1 = {
    "burnout": "burnout appears as a phenomenon or risk, not as a measure",
    "retention": "turnover/attrition appears as an event or risk",
    "interruptions": "personal calendar limits and marketing claims",
    "incident_freq": "on-call burden/load with no quantity",
    "activity_count": "commit/PR quotas given to the agent",
    "wait_time": "timeouts and waiting periods of the agent",
    "onboarding_time": "project setup and migration time",
    "ci_time": "CI runtime as an environment, build duration of a database index",
    "review_turnaround": "review of contracts and of orders, not of code",
    "unplanned": "industrial maintenance",
}
for _i in ITEMS:
    if _i["key"] in DROPPED_ROUND1 and not _i.get("removed"):
        _i["removed"] = DROPPED_ROUND1[_i["key"]]

# Patterns kept for skills but dropped for context files, where the matches
# often carried another sense.
SKILLS_ONLY = {
    "restore": "MTTR of an agent workflow",
    "knowledge_friction": "'no manual steps' as a deploy requirement",
    "defects": "specific defects, not a rate",
}

# Patterns dropped in a second inspection, on a fresh random sample of
# matches drawn after the first selection (15 Nov 2026).
DROPPED_ROUND2 = {
    "deploy_freq": "architecture decision factor, not a measure",
    "restore": "MTTR of the agent itself, stated focus with no measure",
    "satisfaction": "external customers, brand metric, generic goal",
    "defects": "defect density and specific cases, not an escape rate",
}
for _i in ITEMS:
    if _i["key"] in DROPPED_ROUND2 and not _i.get("removed"):
        _i["removed"] = DROPPED_ROUND2[_i["key"]]

# Removed items stay in the file, with the reason, so that the decision is
# traceable, but they are not counted.
REMOVED = [i for i in ITEMS if i.get("removed")]
ITEMS = [i for i in ITEMS if not i.get("removed")]


def items(corpus: str) -> list[dict]:
    """Items that count in each corpus (skills or context files)."""
    if corpus == "context":
        return [i for i in ITEMS if i["key"] not in SKILLS_ONLY]
    return ITEMS


def regex(item: dict) -> str:
    """The pattern of the item in the form in which it is passed to DuckDB (RE2)."""
    return rf"(?i)\b(?:{item['pattern']})\b"


def item_dimensions(item: dict) -> list[tuple[str, str]]:
    seen = []
    for fw, dim, _, _ in item["anchors"]:
        if (fw, dim) not in seen:
            seen.append((fw, dim))
    return seen


def _norm(t: str) -> str:
    t = re.sub(r"-\s*\n\s*", "", t)
    return re.sub(r"\s+", " ", t).lower()


def _source_text(source: str) -> str:
    path = SOURCE_TEXTS[source]
    if path is None or not Path(path).exists():
        raise SystemExit(f"text of source {source} not found at {path}")
    if str(path).endswith(".pdf"):
        raw = subprocess.run(["pdftotext", str(path), "-"], capture_output=True,
                               text=True, check=True).stdout
    else:
        raw = Path(path).read_text(errors="ignore")
    return _norm(raw)


def inventory() -> dict[str, int]:
    """Canonical metric of the MLR -> number of sources that report it."""
    with open(INVENTORY, newline="") as fh:
        return {l["metric"]: int(l["n_sources"]) for l in csv.DictReader(fh)}


def provenance() -> list[dict]:
    """One row per anchor, for kept and removed items (replication package).
    Needs the review data (MLR_ROOT) for the number of sources per metric."""
    inv = inventory()
    rows = []
    for it in ITEMS + REMOVED:
        for fw, dim, source, snippet in it["anchors"]:
            rows.append({
                "item": it["key"], "framework": fw, "dimension": dim,
                "source": source, "reference": REFERENCE[source], "snippet": snippet,
                "mlr_metric": "; ".join(it["mlr"] or []),
                "mlr_sources": "; ".join(str(inv.get(m, "")) for m in (it["mlr"] or [])),
                "pattern": it["pattern"],
                "counted_in": ("" if it.get("removed") else
                               "skills" if it["key"] in SKILLS_ONLY else "skills; context"),
                "removed": it.get("removed", "")})
    return rows


def verify(verbose: bool = True) -> list[str]:
    """Checks the lexicon. Pattern syntax and dimension names are always
    checked. With MLR_ROOT set, also fails if an anchor passage is not in the
    framework text or a canonical metric is not in the review inventory;
    without it, that part is skipped (the frozen result of the check is
    data/lexicon/provenance.csv)."""
    problems = []
    for it in ITEMS:
        re.compile(regex(it).replace("(?i)", ""))  # valid syntax
        for fw, dim, _, _ in it["anchors"]:
            if (fw, dim) not in DIMENSIONS:
                problems.append(f"{it['key']}: unknown dimension {fw}/{dim}")
    inv = {}
    if MLR is not None:
        texts = {f: _source_text(f) for f in SOURCE_TEXTS}
        inv = inventory()
        for it in ITEMS:
            for fw, dim, source, snippet in it["anchors"]:
                if _norm(snippet) not in texts[source]:
                    problems.append(f"{it['key']}: passage missing from {source}: {snippet!r}")
            for m in it["mlr"] or []:
                if m not in inv:
                    problems.append(f"{it['key']}: canonical metric missing from the review: {m!r}")
    if verbose:
        print(f"[lexicon] {len(ITEMS)} items, {sum(len(i['anchors']) for i in ITEMS)} anchors"
              + (f", {len(inv)} canonical metrics in the review" if MLR is not None else
                 "; MLR_ROOT not set: anchor passages and canonical metrics not re-checked"
                 f" (see {PROVENANCE_CSV.name})"))
        for fw, dim in DIMENSIONS:
            its = [i["key"] for i in ITEMS if (fw, dim) in item_dimensions(i)]
            print(f"  {fw:5s} {dim:32s} {len(its)} items: {', '.join(its)}")
        print()
        print("\n".join(f"  PROBLEM {p}" for p in problems) or "[lexicon] all checks passed")
    return problems


if __name__ == "__main__":
    if sys.argv[1:] == ["export"]:
        # writes data/lexicon/provenance.csv (needs MLR_ROOT)
        rows = provenance()
        PROVENANCE_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(PROVENANCE_CSV, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} rows in {PROVENANCE_CSV}")
        sys.exit(0)
    sys.exit(1 if verify() else 0)
