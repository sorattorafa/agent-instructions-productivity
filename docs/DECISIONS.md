# Method decisions

A log of the decisions that shape the numbers, in the order they were made.
Counts quoted here are the ones produced by the pipeline at the time; the
paper always takes its numbers from `paper/numbers.tex`.

## Collection (4–11 Nov 2026)

- **Targets.** Fourteen context-file types: `CLAUDE.local.md`, `GEMINI.md`,
  `.windsurfrules`, `.clinerules`, `.cursorrules`,
  `.github/copilot-instructions.md`, `.github/instructions/`, `CLAUDE.md`,
  `AGENTS.md`, `.claude/rules/`, `.claude/agents/`, `.claude/commands/`,
  `.cursor/rules/`, `memory-bank/`.
- **Discovery.** GitHub code search by file name or path, partitioned by file
  size on a geometric grid; a range with more than 1,000 results is split
  into parts of about 800. A single byte size that alone exceeds the cap is
  fetched in ascending and descending order (up to 2,000 results) and marked
  `truncated`; the number of results not listed is kept.
- **Texts.** GraphQL, 50 blobs per query, deduplicated by blob SHA.
- **Size.** The API's `total_count` underestimates even in narrow ranges
  (about 10% for `CLAUDE.md`), so sizes are the number of results fetched.
- **Rate limits.** Search pace follows the API's own counter; the pause
  between searches adapts to secondary limits (+50% on a block, −3% on
  success, between 6.2 s and 30 s).

## Corpus definition (13–14 Nov 2026)

- **Exact file name.** Code search's `filename:` also matches prefixes and
  suffixes (`agents.mdx`, `subagents.md`, `AGENTS.md.template`). Only
  occurrences whose file name equals the target (case-insensitive) are kept;
  folder targets must contain the folder in the path.
- **Not inside `skills/`.** A context-type file inside a `skills/` folder is
  part of a vendored skill, not repository context, and is excluded.
- **Unit of analysis.** Distinct contents (blob SHA). Redundancy is
  (occurrences − distinct) / occurrences, the definition of the GitSkills
  paper; the pipeline reproduces its 50.5% for skills as a check.

## Dataset characterization: origin of the most copied contents (14 Nov 2026)

Rules on evidence observable in the text, applied in order (`origin.py`):
pointer to another context file; block written by a memory tool; empty rule
header; platform or framework marker; otherwise "other". For skills, the
skill name matches a folder of a public collection (listings checked on
GitHub on 14 Nov 2026).

## Dataset characterization: topic model (13–15 Nov 2026)

- One NMF over both artifacts, balanced, so topics are shared and comparable.
- **Sample.** Texts whose blob SHA starts with fixed two-hex-digit prefixes
  (skills `00`–`07`, context `00`–`04`): a uniform draw anyone can redo
  without a seed.
- **Language filter.** At least 40 tokens, share of English function words
  ≥ 0.08, and less than 5% non-ASCII characters (0.20 discarded 44% of the
  context files, which are bulleted instructions with few function words).
- **Near duplicates.** Texts with the same normalized first 300 characters
  count once in the model (template variants otherwise form their own
  topics). They are still counted in the replication figures.
- **Cap per account.** At most 10 texts per repository owner: one account
  with 1,748 mass-generated skills formed a topic of its own.
- ***k*.** Highest UMass coherence among candidates whose largest group stays
  under 25%: *k* = 8.
- **Stability.** Refits on three 80% subsamples, components matched with the
  Hungarian method on cosine similarity. (Changing only the random state
  tests nothing: the `nndsvda` initialisation is deterministic.)
- **Labels.** The only judgement in the topic description: assigned after reading the 10
  highest-weight and 10 random documents per topic, then revised against the
  typical phrases (`phrases.py`). Topics are not mapped to productivity
  dimensions.
- **Held-out validation.** Disjoint SHA prefixes (skills `08`–`27`, context
  `05`–`18`), same filters, projected on the fitted topics.

## RQ1 and RQ2: naming of frameworks and metrics (12–15 Nov 2026)

- **Why metric names.** An earlier lexicon mapped task vocabulary (deploy,
  test, documentation) to framework dimensions. It was dropped: its patterns
  did not derive from the frameworks' own texts and it had no independent
  validation.
- **Provenance chain.** Each item links a verbatim passage of the framework's
  defining text → a canonical metric of the multivocal literature review →
  the pattern (`lexicon.py`, `data/lexicon/provenance.csv`).
- **What counts.** Only expressions that name a quantity (rate, time,
  frequency, count). The dimension is the one the framework assigns; a
  pattern counts for two dimensions only when the frameworks themselves list
  the metric in both. Cycle time is not lead time for changes.
- **Strict criterion.** In doubt, discard.
- **Named mentions.** Strict patterns: the acronym alone does not count; it
  must be tied to metrics, framework, research or survey. "Developer
  productivity" is reported as a stated goal, not as measurement.
- **First reading of the matches (12 Nov).** Random snippets per pattern
  were read; patterns whose matches often carried another sense were dropped
  or kept for skills only.
- **Second reading (15 Nov).** Fresh random snippets, drawn after the first
  selection, led to dropping four more metric patterns (deployment
  frequency, time to restore, developer satisfaction, escaped defects) and
  the DevEx mention pattern for context files.
- **What is judged.** Only the matched expression, against the meaning of
  its pattern.
- **What is reported.** Files that match the patterns; a false positive
  can only raise the counts.
- **Census vs sample.** Reuse rates and productivity counts are counts over the whole
  collected corpus, so effect sizes are reported, not sampling error.
  Intervals and tests apply to the topic-model sample.

## Languages (15 Nov 2026)

Language of the texts that mention measurement, against the SHA-prefix sample of each
corpus. Code blocks, inline code, URLs and HTML tags are removed; a text with
at least 10% CJK letters is assigned by script (kana → Japanese, hangul →
Korean, otherwise Chinese), because Chinese prose full of English terms
misleads the classifier; otherwise `py3langid` on the first 5,000 characters.
