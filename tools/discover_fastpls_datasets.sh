#!/bin/sh

set -eu

out_dir=${1:-.}
mkdir -p "$out_dir"

paths_file="$out_dir/dataset_discovery_paths.txt"
files_file="$out_dir/dataset_discovery_files.txt"
named_file="$out_dir/dataset_discovery_named.txt"
summary_file="$out_dir/dataset_discovery_summary.txt"

: > "$paths_file"
: > "$files_file"
: > "$named_file"
: > "$summary_file"

add_dir() {
  if [ -d "$1" ]; then
    printf '%s\n' "$1" >> "$paths_file"
  fi
}

add_dir "$HOME/Documents/fastPLS/data"
add_dir "$HOME/Documents/Rdatasets"
add_dir "$HOME/GPUPLS/Data"
add_dir "$HOME/GPUPLS"
add_dir "$HOME/Documents/Stefano"
add_dir "/mnt/sata_ssd"

for d in "$HOME"/fastPLS*; do
  [ -d "$d" ] && printf '%s\n' "$d" >> "$paths_file"
done

sort -u "$paths_file" -o "$paths_file"

while IFS= read -r dir; do
  find "$dir" -maxdepth 3 -type f \
    \( -iname "*.RData" -o -iname "*.Rdata" -o -iname "*.rds" -o -iname "*.csv" \) \
    2>/dev/null
done < "$paths_file" | sort -u > "$files_file"

patterns='MetRef CBMC CITE CCLE CIFAR100 GTEx ImageNet imagenet DINO DINOv2 NMR PRISM SingleCell singlecell TCGA BRCA HNSC methylation Pan-Cancer pancancer simulated Simulated'
for pattern in $patterns; do
  while IFS= read -r dir; do
    find "$dir" -maxdepth 5 -iname "*$pattern*" 2>/dev/null
  done < "$paths_file"
done | sort -u > "$named_file"

{
  echo "fastPLS dataset discovery"
  echo "date: $(date)"
  echo "host: $(hostname)"
  echo
  echo "Search roots:"
  sed 's/^/  /' "$paths_file"
  echo
  echo "RData/RDS/CSV file count:"
  wc -l < "$files_file" | sed 's/^/  /'
  echo
  echo "Named benchmark hits:"
  wc -l < "$named_file" | sed 's/^/  /'
  echo
  echo "First 80 RData/RDS/CSV files:"
  sed -n '1,80p' "$files_file" | sed 's/^/  /'
  echo
  echo "First 80 named benchmark hits:"
  sed -n '1,80p' "$named_file" | sed 's/^/  /'
} > "$summary_file"

cat "$summary_file"
