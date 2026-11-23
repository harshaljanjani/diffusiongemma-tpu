#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -lt 6 ]; then
  echo "usage: launch_session.sh NAME MACHINE_TYPE MAX_RUN_DURATION DISK_GB SESSION ZONE [ZONE...]" >&2
  exit 2
fi

NAME=$1
MACHINE_TYPE=$2
MAX_RUN_DURATION=$3
DISK_GB=$4
SESSION=$5
shift 5

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
REMOTE="bash $SCRIPT_DIR/remote.sh $NAME"

until bash "$SCRIPT_DIR/create_vm.sh" "$NAME" "$MACHINE_TYPE" "$MAX_RUN_DURATION" "$DISK_GB" "$@"; do
  echo "no capacity at $(date -u +%H:%M:%S), retrying"
  sleep 60
done
echo "created at $(date -u +%H:%M:%S)"
$REMOTE authorize > /dev/null 2>&1

until $REMOTE run "test -f /opt/dgemma/BOOTSTRAP_DONE" 2> /dev/null; do
  sleep 15
done
echo "bootstrapped at $(date -u +%H:%M:%S)"

$REMOTE upload
$REMOTE run "cd /opt/dgemma/repo && (sudo nohup bash -c 'bash cloud/run_$SESSION.sh > /opt/dgemma/$SESSION.log 2>&1; cp /opt/dgemma/$SESSION.log results/$SESSION/; sleep 150; shutdown -h now' > /dev/null 2>&1 &)"
echo "session started at $(date -u +%H:%M:%S)"
bash "$SCRIPT_DIR/watch_results.sh" "$NAME" "/opt/dgemma/repo/results/$SESSION" "$REPO_DIR/results" 60
$REMOTE fetch "/opt/dgemma/$SESSION.log" "$REPO_DIR/results/$SESSION" > /dev/null 2>&1
echo "finished at $(date -u +%H:%M:%S)"
