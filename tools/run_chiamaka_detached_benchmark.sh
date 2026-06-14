#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-chiamaka}"
REMOTE_HOST="${REMOTE_HOST:-137.158.224.178}"
REMOTE_BASE="${REMOTE_BASE:-/mnt/sata_ssd}"
REMOTE_SOURCE="${REMOTE_SOURCE:-/mnt/sata_ssd/fastEmbedR_source}"
BENCHMARK_NAME="${BENCHMARK_NAME:-benchmark}"
REMOTE_OUT_DIR="${REMOTE_OUT_DIR:-$REMOTE_BASE/fastEmbedR_${BENCHMARK_NAME}_$(date +%Y%m%d_%H%M%S)}"

if [[ $# -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage:
  tools/run_chiamaka_detached_benchmark.sh 'Rscript tools/your_benchmark.R --arg=value'

Environment:
  SSHPASS         Optional password used by sshpass.
  REMOTE_USER     Default: chiamaka
  REMOTE_HOST     Default: 137.158.224.178
  REMOTE_SOURCE   Default: /mnt/sata_ssd/fastEmbedR_source
  REMOTE_BASE     Default: /mnt/sata_ssd
  BENCHMARK_NAME  Default: benchmark
  REMOTE_OUT_DIR  Optional exact output directory.

The command is executed on chiamaka with:
  cd $REMOTE_SOURCE
  source ~/.fastEmbedR/cuvs_env.sh when present
  nohup bash run.sh > run.log 2>&1 &

The remote run writes:
  status.txt
  pid.txt
  run.log
  command.txt
USAGE
  exit 2
fi

REMOTE_COMMAND="$*"

ssh_base=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
  ssh_base=(sshpass -e ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
fi

"${ssh_base[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
  "REMOTE_OUT_DIR='${REMOTE_OUT_DIR}' REMOTE_SOURCE='${REMOTE_SOURCE}' REMOTE_COMMAND='${REMOTE_COMMAND}' bash -s" <<'REMOTE'
set -euo pipefail

mkdir -p "$REMOTE_OUT_DIR"
cd "$REMOTE_OUT_DIR"
printf "%s\n" "$REMOTE_COMMAND" > command.txt

cat > run.sh <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
echo "RUNNING" > status.txt
echo "started_at=$(date --iso-8601=seconds 2>/dev/null || date)" >> status.txt
echo "host=$(hostname)" >> status.txt
echo "cwd=$PWD" >> status.txt

if [[ -f "$HOME/.fastEmbedR/cuvs_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.fastEmbedR/cuvs_env.sh"
fi

cd "$REMOTE_SOURCE"
git fetch origin
git reset --hard origin/main

set +e
bash -lc "$REMOTE_COMMAND"
code=$?
set -e

if [[ $code -eq 0 ]]; then
  echo "DONE" > "$REMOTE_OUT_DIR/status.txt"
else
  echo "FAILED" > "$REMOTE_OUT_DIR/status.txt"
fi
echo "exit_code=$code" >> "$REMOTE_OUT_DIR/status.txt"
echo "finished_at=$(date --iso-8601=seconds 2>/dev/null || date)" >> "$REMOTE_OUT_DIR/status.txt"
exit "$code"
RUN

chmod +x run.sh
nohup env REMOTE_SOURCE="$REMOTE_SOURCE" REMOTE_OUT_DIR="$REMOTE_OUT_DIR" REMOTE_COMMAND="$REMOTE_COMMAND" \
  bash "$REMOTE_OUT_DIR/run.sh" > "$REMOTE_OUT_DIR/run.log" 2>&1 &
pid=$!
echo "$pid" > pid.txt
echo "RUNNING" > status.txt
echo "pid=$pid" >> status.txt
echo "launched_at=$(date --iso-8601=seconds 2>/dev/null || date)" >> status.txt
echo "out_dir=$REMOTE_OUT_DIR"
echo "pid=$pid"
REMOTE

