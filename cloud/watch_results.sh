#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: watch_results.sh NAME REMOTE_RESULTS_DIR LOCAL_DIR [INTERVAL_SECONDS]" >&2
  exit 2
fi

NAME=$1
REMOTE_DIR=$2
LOCAL_DIR=$3
INTERVAL=${4:-60}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
FAILURES=0

while [ "$FAILURES" -lt 4 ]; do
  if bash "$SCRIPT_DIR/remote.sh" "$NAME" fetch "$REMOTE_DIR" "$LOCAL_DIR" > /dev/null 2>&1; then
    FAILURES=0
  else
    FAILURES=$((FAILURES + 1))
  fi
  sleep "$INTERVAL"
done
echo "stopped watching $NAME after repeated connection failures at $(date -u +%H:%M:%S)"
