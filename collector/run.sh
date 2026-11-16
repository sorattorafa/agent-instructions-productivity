#!/bin/zsh
# supervisor: restarts the collector if it dies; stops when the collection ends.
# to stop it: kill $(cat collector/state/run.pid)
# Uses the `python` found on PATH: activate the virtualenv with the
# dependencies (pandas, pyarrow, huggingface_hub) before running this script.
cd "$(dirname "$0")"
mkdir -p state
echo $$ > state/run.pid
# caffeinate only exists on macOS; elsewhere the collector runs without it
if command -v caffeinate >/dev/null 2>&1; then
  # keeps the Mac awake while the supervisor lives, including the pause between restarts
  caffeinate -is -w $$ &
  run=(caffeinate -is python collector.py)
else
  run=(python collector.py)
fi
while true; do
  "${run[@]}" >> state/collector.log 2>&1
  # the collector's last log line ends in " done" when the collection finished
  tail -1 state/collector.log | grep -q " done$" && break
  echo "$(date '+%m-%d %H:%M:%S') supervisor: collector exited, restarting in 60s" >> state/collector.log
  sleep 60
done
