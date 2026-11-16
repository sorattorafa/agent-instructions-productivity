# Agent instructions and developer productivity — replication package

Replication package for a study under double-blind review (MSR 2027 Mining
Challenge).

The study reads two kinds of files that developers write for coding agents:

- **agent skills** (`SKILL.md`), from the public GitSkills dataset
  (`mvaccargiu/gitskills` on the Hugging Face Hub, revision `ebab174`);
- **context files** (`AGENTS.md`, `CLAUDE.md`, `.cursor/rules`, and eleven
  other file types), collected for this study with GitHub code search
  (the collector is in `collector/`).

Contributions:

- **A dataset of context files** that complements GitSkills, with its
  collector, and a characterization of both datasets (how much is copied,
  where the most copied contents come from, what the files talk about).
- **An analysis of both datasets**:
  - **RQ1** — What do skills and context files tell the agent?
  - **RQ2** — Do they name DORA, SPACE or DevEx or their metrics, and do
    they only cite them or state a value?

Every number in the paper is a LaTeX macro written by `scripts/synthesis.py`
from the files in `data/`; nothing is typed by hand.

## Repository layout

```
scripts/      analysis pipeline (Python)
collector/    GitHub collector for the context files
data/         pipeline outputs (small files tracked; large intermediates regenerable)
paper/        generated numbers and tables (numbers.tex, tab_corpus.tex, tab_topics.tex);
              the paper source is withheld during review
run_pipeline.sh   runs the steps in order
requirements.txt  versions used for the results in the paper
```

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Python 3.13 was used. `pdftotext` (poppler) is only needed for the optional
lexicon source check (see "Lexicon provenance"). The paper compiles with
`tectonic` (0.16.9) or any LaTeX distribution with `IEEEtran`.

All scripts run **from the repository root** (they read and write `data/`
with relative paths):

```bash
python scripts/<name>.py [arguments]
```

### Datasets

| Variable | Default | What it is |
|---|---|---|
| `GITSKILLS_REPO` | `mvaccargiu/gitskills` | public GitSkills dataset (skills) |
| `CONTEXT_DATASET_REPO` | `ANONYMIZED/agent-context-files` | context-file dataset collected for this study |

The context-file dataset id is anonymized for review. Set
`CONTEXT_DATASET_REPO` to the dataset id given in the paper's Data
Availability section before running any step that reads the texts remotely.
The scripts read both datasets remotely with DuckDB (`hf://`); a Hugging Face
token is needed only if the dataset is private (`huggingface-cli login`).

The context-file dataset keeps the **field names it was collected with**,
which are in Portuguese. The scripts alias them to English as soon as they
read them:

| Dataset path / column | Meaning |
|---|---|
| `ocorrencias/*.parquet` | occurrences: one row per (repository, path) found by the search |
| `textos/*.parquet` | texts: one row per distinct blob (`sha`), with its content |
| `estado/faixas.parquet` | state of the size ranges used to partition the search |
| `alvo` | target file type (e.g. `AGENTS.md`, `.cursor/rules`) |
| `caminho`, `nome` | path and file name |
| `nome_exato` | whether the file name matches the target exactly |
| `familia` | tool family (claude, cursor, copilot, ...) |
| `faixa_a`, `faixa_b` | bounds of the size range, in bytes |
| `coletado_em`, `baixado_em` | collected at, downloaded at (UTC) |
| `texto` | file content |
| `obtidos`, `atualizado` | results fetched for a range; last update |
| status `feita` / `truncada` | range done / range truncated at the 1,000-result cap |

## Pipeline

`./run_pipeline.sh remote` runs the passes over the two datasets (hours; each
step has a checkpoint and resumes when run again). `./run_pipeline.sh local`
runs everything after them (about 30 minutes on a laptop with 8 GB of RAM).
Run one heavy step at a time.

| # | Script | What it does | Main outputs |
|---|---|---|---|
| 0 | `collector/collector.py` | collects context files from GitHub (code search partitioned by file size; text via GraphQL) and uploads them to the dataset | the context-file dataset |
| 1 | `scripts/lexicon.py` | the metric-name lexicon: each pattern is tied to a verbatim passage of the framework and to a canonical metric of a multivocal literature review; the script checks the lexicon | `data/lexicon/provenance.csv` |
| 2 | `scripts/framework_search.py` | first-round search for framework mentions; the pipeline uses its deduplicated occurrences and the size ranges | `data/collection/occurrences.parquet`, `data/size_ranges.parquet` |
| 3 | `scripts/vocabulary.py skills\|context` | applies the lexicon to every distinct text | `data/vocabulary/<corpus>/partials/` |
| 4 | `scripts/vocabulary_summary.py skills\|context` | first counts per item and a random sample of snippets per pattern, to read what each one matches | `data/vocabulary/<corpus>/*.csv` |
| 5 | `scripts/corpus_pass.py skills\|context` | one pass per corpus: explicit framework mentions, markers of tool-generated text, and the SHA-prefix sample for the topic model | `data/<corpus>/{mentions,markers,sample}/` |
| 6 | `scripts/name_filter.py` | keeps context occurrences whose file name is exactly the target and that are not inside a `skills/` folder | `data/context_occurrences.parquet`, `data/name_filter.csv` |
| 7 | `scripts/reuse.py` | Dataset characterization (replication): redundancy per file type and the most copied contents | `data/reuse/summary.csv`, `top_contents.csv` |
| 8 | `scripts/origin.py` | Dataset characterization (replication): origin of the 30 most copied contents, by explicit rules | `data/reuse/origin_top.csv`, `origin_summary.csv` |
| 9 | `scripts/topics.py` | Dataset characterization (topics): one NMF topic model over a balanced sample of skills and context files; choice of *k*; stability | `data/topics/` |
| 10 | `scripts/phrases.py` | typical 2–3 word phrases per topic (support for topic labels) | `data/topics/phrases_by_topic.csv`, `coverage_top100.csv` |
| 11 | `scripts/heldout_sample.py skills\|context` | downloads a disjoint SHA-prefix sample the model did not see | `data/<corpus>/heldout/` |
| 12 | `scripts/heldout_validation.py` | projects the held-out sample on the fitted topics and compares shares and phrases | `data/topics/heldout/` |
| 13 | `scripts/productivity.py` | RQ1 and RQ2: texts that name a framework or a metric, per corpus, framework, dimension and file type | `data/productivity/productivity.csv`, `productivity_by_target.csv` |
| 15 | `scripts/statistical_tests.py` | rate ratios, chi-square with Cramér's V, Wilson intervals | `data/statistics/summary.json` |
| 16 | `scripts/languages.py download\|detect` | language of the texts that mention measurement, against a random sample of each corpus | `data/languages/` |
| 17 | `scripts/synthesis.py` | writes every number and both tables of the paper | `paper/numbers.tex`, `tab_corpus.tex`, `tab_topics.tex` |
| 18 | `scripts/figures.py` | draws the two figures of the paper from the same outputs | `paper/fig/fig_dataset.pdf`, `fig_productivity.pdf` |

## What is in `data/`

Tracked (small): every table the paper is computed from —
`reuse/`, `topics/` (including `heldout/`), `productivity/`, `statistics/`,
`languages/*.csv|json`, `lexicon/provenance.csv`, `name_filter.csv`,
`size_ranges.parquet`, the per-text flags of the lexicon
(`vocabulary/<corpus>/partials/`) and of the framework mentions
(`<corpus>/mentions/`), and the checkpoints (`done*.json`, `totals.json`).

### What is not in the repository

These are large and are regenerated by the scripts from the two datasets
(listed in `.gitignore`):

| Path | Size | Regenerate with |
|---|---|---|
| `data/collection/occurrences.parquet` | 143 MB | `framework_search.py` |
| `data/context_occurrences.parquet` | 131 MB | `name_filter.py` |
| `data/reuse/skills_occurrences.parquet` | 109 MB | `reuse.py` |
| `data/<corpus>/markers/` | 113 MB | `corpus_pass.py` |
| `data/<corpus>/sample/` | 270 MB | `corpus_pass.py` |
| `data/<corpus>/heldout/` | 1.1 GB | `heldout_sample.py` |
| `data/languages/texts_*.parquet` | small, full file texts | `languages.py download` |

The samples are defined by the first two hex digits of the blob SHA
(`corpus_pass.py`, `heldout_sample.py`), so anyone can redraw exactly the
same texts without a random seed.

## How the patterns were chosen

The counts in the paper are files that match the patterns in
`scripts/lexicon.py` (metric names) and `scripts/corpus_pass.py` plus
`scripts/productivity.py` (framework names). The patterns are deliberately
conservative. Each one was narrowed after reading random samples of its
matches (`vocabulary_summary.py` and `productivity.py` draw them), and
patterns whose matches often carried another sense were dropped; every
dropped pattern stays in `scripts/lexicon.py` with the reason
(`DROPPED_ROUND1`, `DROPPED_ROUND2`, `SKILLS_ONLY`) and in
`data/lexicon/provenance.csv`. A false positive can only raise the counts,
so the reported rates are upper bounds for the patterns used.

## Validation by model raters

Three language models of different vendors labelled every pattern match and
1,000 files with no match, each alone and with the same instruction. No
person labelled these items. The instruction, the items, the labels of each
model and the agreement between them are in `validation/llm/` (see its
`README.md`).

| Script | Does | Writes |
|---|---|---|
| `scripts/validation_items.py` | builds the items of the two tasks | `validation/llm/*_items.jsonl` |
| `scripts/llm_rater_gemini.py`, `scripts/llm_rater_openai.py` | send the instruction and the items to each API (a key is needed) | `validation/llm/gemini/`, `validation/llm/openai/` |
| `scripts/llm_agreement.py` | shares, Cohen's and Fleiss' kappa | `data/validation/llm_summary.json` |

## Lexicon provenance

`scripts/lexicon.py` holds every pattern, kept or removed, with the reason
for each removal. `data/lexicon/provenance.csv` lists, for each item: the
framework and dimension, the verbatim passage of the framework text that
names the metric, the source of that passage, the canonical metric of the
multivocal literature review and the number of its sources that report it.

The script can re-check the passages against the framework texts and the
review inventory when `MLR_ROOT` points to the review data (not distributed
here: the review is under submission and the framework texts are
copyrighted). Without `MLR_ROOT` it checks pattern syntax and dimension names
only.

## Ethics and data

Only public repositories were read, through GitHub's official APIs and within
their rate limits. The study is about file contents, not people. The released
data contain repository names, paths and blob hashes, as GitSkills does.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0), the same license
as the GitSkills dataset this study builds on. It covers the scripts, the
derived data and the documentation; the full text is in
`LICENSE`. File contents quoted in snippets and samples remain under the
licenses of their source repositories.
