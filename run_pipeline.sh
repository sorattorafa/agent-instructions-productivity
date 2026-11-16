#!/bin/sh
# Runs the pipeline in order, from the repository root.
#   ./run_pipeline.sh remote   passes over the two datasets (hours; needs network and the
#                              dataset ids, see README); every step resumes if interrupted
#   ./run_pipeline.sh local    everything after the remote passes (about 30 min, 8 GB of RAM)
# One heavy job at a time: the steps are sequential on purpose.
set -e
PY=${PYTHON:-python}
case "$1" in
  remote)
    $PY scripts/lexicon.py
    $PY scripts/framework_search.py          # deduplicated context occurrences, size ranges
    $PY scripts/vocabulary.py skills
    $PY scripts/vocabulary.py context
    $PY scripts/corpus_pass.py skills        # mentions, markers, topic-model sample
    $PY scripts/corpus_pass.py context
    $PY scripts/name_filter.py
    $PY scripts/reuse.py                     # also fetches the text of the most copied contents
    $PY scripts/heldout_sample.py skills
    $PY scripts/heldout_sample.py context
    ;;
  local)
    $PY scripts/lexicon.py
    $PY scripts/name_filter.py
    $PY scripts/origin.py
    $PY scripts/file_lengths.py
    $PY scripts/topics.py
    $PY scripts/phrases.py
    $PY scripts/heldout_validation.py
    $PY scripts/productivity.py
    $PY scripts/overlap.py
    $PY scripts/statistical_tests.py
    # texts of the files counted in the productivity analysis (remote; skipped if already downloaded)
    [ -f data/languages/texts_skills.parquet ] || $PY scripts/languages.py download skills
    [ -f data/languages/texts_context.parquet ] || $PY scripts/languages.py download context
    $PY scripts/languages.py detect
    $PY scripts/synthesis.py
    $PY scripts/figures.py
    ;;
  *)
    echo "usage: $0 remote|local"; exit 1 ;;
esac
