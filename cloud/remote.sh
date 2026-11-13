#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: remote.sh NAME {authorize|upload|run COMMAND|fetch REMOTE_PATH LOCAL_DIR|delete}" >&2
  exit 2
fi

NAME=$1
ACTION=$2
shift 2

GCLOUD=${GCLOUD:-gcloud}
SSH_KEY=${DGEMMA_SSH_KEY:?set DGEMMA_SSH_KEY to a private key path}
SSH_USER=${DGEMMA_SSH_USER:-dgemma}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
ZONE=$(cat "$SCRIPT_DIR/.zone-$NAME")
SSH_OPTIONS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20)

native_path() {
  if command -v cygpath > /dev/null; then cygpath -w "$1"; else echo "$1"; fi
}

address() {
  "$GCLOUD" compute instances describe "$NAME" --zone "$ZONE" \
    --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2> /dev/null | tail -n 1
}

case "$ACTION" in
  authorize)
    KEYS_FILE=$(mktemp)
    echo "$SSH_USER:$(cat "$SSH_KEY.pub")" > "$KEYS_FILE"
    "$GCLOUD" compute instances add-metadata "$NAME" --zone "$ZONE" \
      --metadata-from-file=ssh-keys="$(native_path "$KEYS_FILE")"
    rm -f "$KEYS_FILE"
    ;;
  upload)
    ARCHIVE=$(mktemp)
    git -C "$REPO_DIR" archive --format=tar.gz -o "$(native_path "$ARCHIVE")" HEAD
    scp "${SSH_OPTIONS[@]}" "$ARCHIVE" "$SSH_USER@$(address):/tmp/dgemma.tar.gz"
    ssh "${SSH_OPTIONS[@]}" "$SSH_USER@$(address)" \
      "sudo mkdir -p /opt/dgemma/repo && sudo tar -xzf /tmp/dgemma.tar.gz -C /opt/dgemma/repo"
    rm -f "$ARCHIVE"
    ;;
  run)
    ssh "${SSH_OPTIONS[@]}" "$SSH_USER@$(address)" "$@"
    ;;
  fetch)
    mkdir -p "$2"
    scp -r "${SSH_OPTIONS[@]}" "$SSH_USER@$(address):$1" "$2"
    ;;
  delete)
    "$GCLOUD" compute instances delete "$NAME" --zone "$ZONE" --quiet
    rm -f "$SCRIPT_DIR/.zone-$NAME"
    ;;
  *)
    echo "unknown action: $ACTION" >&2
    exit 2
    ;;
esac
