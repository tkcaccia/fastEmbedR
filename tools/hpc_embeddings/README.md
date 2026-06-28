# HPC Embedding Benchmarks

These scripts run publication-style embedding benchmarks on the HPC datasets in
`/scratch/firenze/NN/Data`.

## Files

- `benchmark_embeddings_float32_publication.R`
  Main R driver. It runs each dataset/method in an isolated child R process,
  captures elapsed time, captures peak RSS memory through `/usr/bin/time -v`
  when available, saves layouts, saves per-method plots, and continues after
  failed/OOM/timeout methods.

- `benchmark_embeddings_float32_cpu12.sh`
  CPU-only Slurm wrapper using 12 CPU cores. It runs:
  `fastEmbedR_opentsne_cpu`, `fastEmbedR_umap_cpu_fuzzy`,
  `fastEmbedR_umap_cpu_binary`, `Rtsne_full`, `KlugerLab_FItSNE`,
  `umap_package`, `uwot_default`, and `uwot_fast_sgd`.

- `benchmark_embeddings_float32_cuda.sh`
  CUDA-only Slurm wrapper using one L40S GPU. It runs:
  `fastEmbedR_opentsne_cuda`, `fastEmbedR_umap_cuda_fuzzy`, and
  `fastEmbedR_umap_cuda_binary`. It prints CUDA/faissR/fastEmbedR diagnostics
  before the benchmark so a missing CUDA backend is visible immediately.

## Input Rule

- fastEmbedR methods load each dataset's `*_float32.RData` file.
- Reference R packages load each dataset's standard `.RData` file.

## Copy To HPC Folder

From the local machine:

```bash
cp /Users/stefano/Documents/umap/tools/hpc_embeddings/benchmark_embeddings_float32_publication.R \
   /Users/stefano/HPC-firenze/NN/
cp /Users/stefano/Documents/umap/tools/hpc_embeddings/benchmark_embeddings_float32_cpu12.sh \
   /Users/stefano/HPC-firenze/NN/
cp /Users/stefano/Documents/umap/tools/hpc_embeddings/benchmark_embeddings_float32_cuda.sh \
   /Users/stefano/HPC-firenze/NN/
```

If `dataset_input_audit.csv` reports missing standard `.RData` files for
reference packages, copy them into the local HPC mirror before syncing:

```bash
bash /Users/stefano/Documents/umap/tools/hpc_embeddings/sync_missing_standard_rdata.sh
```

Then sync `/Users/stefano/HPC-firenze/NN` to the HPC as usual.

## Submit On HPC

```bash
sbatch /scratch/firenze/NN/benchmark_embeddings_float32_cpu12.sh
sbatch /scratch/firenze/NN/benchmark_embeddings_float32_cuda.sh
```

Optional overrides:

```bash
DATASETS=MNIST,FashionMNIST K=30 PERPLEXITY=15 TIMEOUT=10800 \
sbatch /scratch/firenze/NN/benchmark_embeddings_float32_cpu12.sh

DATASETS=MNIST,FashionMNIST K=30 PERPLEXITY=15 TIMEOUT=10800 \
sbatch /scratch/firenze/NN/benchmark_embeddings_float32_cuda.sh
```

## Outputs

Each run creates a timestamped output directory containing:

- `embedding_benchmark_results.csv`
- `embedding_time_barplot.png`
- `embedding_memory_barplot.png`
- `layouts/*.rds`
- `plots/*.png`
- `logs/*.log`
- `worker_results/*.csv`
