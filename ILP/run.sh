#!/bin/bash
set -uo pipefail

FASTLAS="./FastLAS/FastLAS2/implementation/build/FastLAS"
TASK="task_pain_full_bias_combined.las"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="results/pain_rules_${TIMESTAMP}.txt"
HEARTBEAT="results/heartbeat_${TIMESTAMP}.log"
CRASHLOG="results/crash_${TIMESTAMP}.log"

echo "PID: $$" >> "$HEARTBEAT"

# Falls der Prozess ein Signal bekommt (Terminal-Muell, kill, etc.),
# wird das HIER geloggt, bevor er stirbt.
trap 'echo "[$(date)] Signal empfangen (PID $$)" >> "$CRASHLOG"' SIGHUP SIGINT SIGTERM SIGQUIT

echo "Start: $(date)" | tee -a "$HEARTBEAT"
echo "Task: $TASK" | tee -a "$HEARTBEAT"
echo "Freier Speicher beim Start:" | tee -a "$HEARTBEAT"
free -h | tee -a "$HEARTBEAT"
echo

# Heartbeat im Hintergrund: alle 60s eine Zeile, damit wir bei einem Abbruch sehen, WANN zuletzt alles ok war.
(
  while true; do
    sleep 60
    echo "[$(date)] noch aktiv - $(free -h | awk '/Mem:/ {print "frei: "$7}')" >> "$HEARTBEAT"
  done
) &
HEARTBEAT_PID=$!

cleanup() {
  kill "$HEARTBEAT_PID" 2>/dev/null
  echo "Ende: $(date), Exit-Code: ${STATUS:-unbekannt/Signal} ===" | tee -a "$HEARTBEAT"
}
trap cleanup EXIT

$FASTLAS --nopl "$TASK" > "$OUTPUT" 2>&1
STATUS=$?

echo
if [ $STATUS -ne 0 ]; then
    echo "FastLAS failed (Exit-Code $STATUS)."
elif grep -q "UNSATISFIABLE" "$OUTPUT"; then
    echo "FastLAS finished, but no rules were learned (UNSATISFIABLE)."
else
    echo "Done :)"
    echo "Output saved to $OUTPUT"
fi
