# Validation by model raters

The patterns were checked by three language models of different vendors.
No person labelled these items. Each model received the same instruction
(`PROMPT.md`), labelled alone and never saw the labels of another model.

| Rater | Model | How it was run |
|---|---|---|
| `claude/` | `claude-opus-5-5` | agent runs, one per slice of the item files (one output file per slice) |
| `gemini/` | `gemini-3.5-flash` | `scripts/llm_rater_gemini.py`, temperature 0 |
| `openai/` | `gpt-5.5-2026-04-23` | `scripts/llm_rater_openai.py` |

## Items

Built by `scripts/validation_items.py`.

- **Task 1** (`precision_items.jsonl`, 3,274 items): every snippet that a
  pattern matched in the two datasets, with the matched term highlighted.
- **Task 2** (`recall_items.jsonl`, 1,000 items): files that matched no
  pattern, 500 per artifact, drawn from the SHA-prefix sample; each text cut
  at 6,000 characters.

The ids of the first round (1-344 in Task 1, 1-200 in Task 2) are fixed by
`validation/precision_sample.csv` and `validation/recall_sample.csv`; the
other items were labelled in a second round with the same instruction.
E-mail addresses in the texts are replaced by `<redacted>`; one item held a
placeholder with the format of an access token, replaced in the published
file after the labelling.

The texts are files from public repositories and are untrusted: some contain
instructions aimed at whatever reads them. The raters only return labels.

## Results

Computed by `scripts/llm_agreement.py` into `data/validation/llm_summary.json`.

| | Task 1: matches accepted | Task 2: files that talk about measuring |
|---|---|---|
| Claude | 3,217 | 7 |
| Gemini | 3,229 | 6 |
| OpenAI | 3,212 | 17 |
| At least two of three | 3,229 (98.6%) | 7 (0.7%) |
| All three | 3,196 (97.6%) | 5 (0.5%) |
| Items with the same label from all | 98.9% | 98.7% |
| Fleiss' kappa | 0.77 | 0.56 |
| Cohen's kappa, Claude and Gemini | 0.80 | 0.92 |
| Cohen's kappa, Claude and OpenAI | 0.73 | 0.49 |
| Cohen's kappa, Gemini and OpenAI | 0.78 | 0.43 |

In Task 2 a positive file is rare, so a few disagreements move kappa a lot
while the raw agreement stays high.
