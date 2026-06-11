#!/bin/sh

set -eu

script_dir=$(cd "$(dirname "$0")" && pwd)
stamp=$(date +%Y%m%d_%H%M%S)

remote_host=${FASTEMBEDR_REMOTE_HOST:-137.158.224.178}
remote_user=${FASTEMBEDR_REMOTE_USER:-chiamaka}
remote_base=${FASTEMBEDR_REMOTE_BASE:-/mnt/sata_ssd}
remote_dir=${FASTEMBEDR_REMOTE_DIR:-"$remote_base/fastEmbedR_dataset_discovery_$stamp"}
remote="$remote_user@$remote_host"

ssh_cmd=${SSH:-ssh}

echo "Creating remote dataset discovery directory: $remote:$remote_dir"
$ssh_cmd "$remote" "mkdir -p '$remote_dir'"

echo "Discovering datasets on remote host..."
$ssh_cmd "$remote" "sh -s '$remote_dir'" < "$script_dir/discover_fastpls_datasets.sh"

echo
echo "Remote discovery files:"
$ssh_cmd "$remote" "ls -lh '$remote_dir'/dataset_discovery_*"
echo
echo "Remote directory: $remote:$remote_dir"
