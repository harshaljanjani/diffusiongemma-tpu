#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: create_vm.sh NAME MACHINE_TYPE MAX_RUN_DURATION DISK_GB ZONE [ZONE...]" >&2
  exit 2
fi

NAME=$1
MACHINE_TYPE=$2
MAX_RUN_DURATION=$3
DISK_GB=$4
shift 4

GCLOUD=${GCLOUD:-gcloud}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BOOTSTRAP="$SCRIPT_DIR/bootstrap.sh"
if command -v cygpath > /dev/null; then
  BOOTSTRAP=$(cygpath -w "$BOOTSTRAP")
fi

for ZONE in "$@"; do
  echo "trying $MACHINE_TYPE in $ZONE"
  if "$GCLOUD" compute instances create "$NAME" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --image-project=ubuntu-os-accelerator-images \
    --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --instance-termination-action=DELETE \
    --max-run-duration="$MAX_RUN_DURATION" \
    --boot-disk-size="${DISK_GB}GB" \
    --scopes=storage-rw,logging-write \
    --metadata-from-file=startup-script="$BOOTSTRAP"; then
    echo "created $NAME in $ZONE"
    echo "$ZONE" > "$SCRIPT_DIR/.zone-$NAME"
    exit 0
  fi
done

echo "no capacity in any requested zone" >&2
exit 1
