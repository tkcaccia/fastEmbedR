#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-chiamaka}"
REMOTE_HOST="${REMOTE_HOST:-137.158.224.178}"

if [[ $# -ne 1 ]]; then
  echo "Usage: tools/check_chiamaka_detached_benchmark.sh /mnt/sata_ssd/remote_run_dir" >&2
  exit 2
fi

REMOTE_OUT_DIR="$1"

ssh_base=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
  ssh_base=(sshpass -e ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
fi

"${ssh_base[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "REMOTE_OUT_DIR='${REMOTE_OUT_DIR}' bash -s" <<'REMOTE'
set -euo pipefail
if [[ ! -d "$REMOTE_OUT_DIR" ]]; then
  echo "Missing remote run directory: $REMOTE_OUT_DIR" >&2
  exit 1
fi

echo "Directory: $REMOTE_OUT_DIR"
echo
echo "status.txt"
cat "$REMOTE_OUT_DIR/status.txt" 2>/dev/null || echo "No status.txt"
echo

if [[ -f "$REMOTE_OUT_DIR/pid.txt" ]]; then
  pid=$(cat "$REMOTE_OUT_DIR/pid.txt")
  echo "process"
  ps -p "$pid" -o pid,etime,pcpu,pmem,cmd || true
  echo
fi

echo "recent log"
tail -n 80 "$REMOTE_OUT_DIR/run.log" 2>/dev/null || echo "No run.log"
echo

echo "files"
find "$REMOTE_OUT_DIR" -maxdepth 2 -type f | sort | tail -50
REMOTE

