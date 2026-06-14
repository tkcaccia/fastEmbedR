#!/usr/bin/env bash
set -euo pipefail

LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-/Users/stefano/Documents/fastEmbedR/Data}"
REMOTE_USER="${REMOTE_USER:-chiamaka}"
REMOTE_HOST="${REMOTE_HOST:-137.158.224.178}"
REMOTE_DIR="${REMOTE_DIR:-/mnt/sata_ssd/fastEmbedR_Data}"

if [[ ! -d "${LOCAL_DATA_DIR}" ]]; then
  echo "Local data directory not found: ${LOCAL_DATA_DIR}" >&2
  exit 1
fi

ssh_base=(ssh -o ConnectTimeout=20)
rsync_ssh='ssh -o ConnectTimeout=20'
if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
  ssh_base=(sshpass -e ssh -o ConnectTimeout=20)
  rsync_ssh='sshpass -e ssh -o ConnectTimeout=20'
fi

"${ssh_base[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "mkdir -p '${REMOTE_DIR}' && df -h /mnt/sata_ssd"

rsync -avh --info=progress2 -e "${rsync_ssh}" \
  --exclude "_downloads/" \
  "${LOCAL_DATA_DIR}/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

"${ssh_base[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "find '${REMOTE_DIR}' -maxdepth 2 -type f \\( -name '*.RData' -o -name '*.csv' -o -name 'README.md' -o -name 'MATERIALS_AND_METHODS.md' \\) -print | sort && du -sh '${REMOTE_DIR}'"
