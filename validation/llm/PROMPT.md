# Instructions given to each model rater

The same text is given, unchanged, to every model. Each model labels alone
and never sees the labels of another rater.

## Background

Three frameworks define how to measure software delivery and developer
productivity:

- DORA: deployment frequency, lead time for changes, change failure rate,
  time to restore service;
- SPACE: satisfaction, performance, activity, communication, efficiency;
- DevEx: feedback loops, cognitive load, flow state.

A study counts files written for coding agents that name these frameworks or
their metrics. You check what search patterns found. In doubt, answer `n`.

## Task 1: precision

Each item is a snippet that a pattern matched. The matched term is between
`>>>` and `<<<`. Judge only the highlighted term.

If `kind` is `metric named`: is the highlighted term this metric of software
delivery, with the sense given in `meaning`?

- `y`: the metric in a list of metrics; a target or benchmark for it; an
  instruction to measure it; a definition of it. A number is not required.
- `n`: another sense (for example lead time of a supplier or a factory);
  another domain (contracts, manufacturing, marketing); a time or quota of
  the agent itself rather than of the team's delivery; a snippet cut too
  short to tell.

If `kind` is `framework cited`: does the snippet refer to this framework?

- `y`: the framework, its metrics, report, research or survey.
- `n`: something else with the same name (for example DORA as the EU
  Digital Operational Resilience Act; a product, package or person).

## Task 2: recall

Each item is the text of a whole file (cut at 6,000 characters) that matched
no pattern. Does the file talk about measuring the work of developing or
delivering software, in any words?

- `y`: it names a delivery or productivity metric in other words (how long
  pull requests wait for review, deployments per week, MTTR); it sets a
  target for the team's delivery or quality over time; it asks the agent to
  compute or report such a measure; it cites a productivity framework.
- `n`: coding rules, style, architecture or commands with nothing about
  measuring; performance of the software itself (latency, memory); test
  coverage or lint thresholds as a gate for one change; business metrics
  (revenue, users, conversion).

## Answer format

For every item return its `id` and one answer, `y` or `n`, as a JSON array:
`[{"id": 1, "answer": "y"}, ...]`. Return every id you were given, once.
