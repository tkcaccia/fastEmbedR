# Reproducibility

[Home](../README.md) |
[Installation](installation.md) |
[Bioconductor](bioconductor.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
**Reproducibility** |
[References](references.md) |
[Benchmark repository](https://github.com/tkcaccia/fastEmbedR-benchmark)

This page records the reproducibility contract for the manuscript benchmarks.
It is intentionally explicit because fastEmbedR benchmarks may involve R
packages, native C++ code, optional Metal kernels, optional CUDA kernels, FAISS,
and RAPIDS cuVS linked directly by optional CUDA builds. Benchmark code and
dataset instructions are versioned independently at
[`tkcaccia/fastEmbedR-benchmark`](https://github.com/tkcaccia/fastEmbedR-benchmark).
Reproducible reports must record both the package commit and benchmark commit.

## Release Snapshot

Before a publication or release benchmark, create and archive clean tags for
both repositories, for example:

```bash
git tag -a v0.99.0-manuscript -m "fastEmbedR manuscript benchmark snapshot"
git push origin v0.99.0-manuscript

cd ../fastEmbedR-benchmark
git tag -a v0.99.0-manuscript -m "fastEmbedR benchmark snapshot"
git push origin v0.99.0-manuscript
```

The release tag should then be archived on Zenodo or an equivalent repository.
Record the minted DOI by setting:

```bash
export FASTEMBEDR_MANUSCRIPT_TAG=v0.99.0-manuscript
export FASTEMBEDR_ZENODO_DOI="10.xxxx/zenodo.xxxxxxx"
```

The reproducibility scripts write these values into the benchmark output
directory. Do not invent a DOI before the archive exists.

## Reproducibility Bundle

Run:

```bash
git clone https://github.com/tkcaccia/fastEmbedR-benchmark.git
cd fastEmbedR-benchmark
Rscript tools/write_manuscript_reproducibility.R \
  --out-dir=results/manuscript_reproducibility_current \
  --seed=4 \
  --k=30 \
  --perplexity=15 \
  --threads=12 \
  --timeout=10800
```

The script writes:

- `reproducibility_manifest.txt`;
- `reproducibility_manifest.json` when `jsonlite` is installed;
- `comparator_identity.csv` with the exact R/Python package version and the
  FIt-SNE Git commit, or the executable SHA-256 when its source checkout is
  unavailable;
- `sessionInfo.txt`.

The manifest records:

- exact Git commit hash;
- `git describe --tags --always --dirty`;
- release tag and archival DOI fields;
- benchmark command lines;
- random seed, k, perplexity, timeout, and thread count;
- R version, package versions, and `sessionInfo()`;
- resolved `R CMD config` values for `CC`, `CFLAGS`, `CXX17`,
  `CXX17FLAGS`, `CPPFLAGS`, `LDFLAGS`, and `SHLIB_CXX17LD`;
- C/C++ compiler, Xcode, and Metal compiler versions where available;
- the generated package `src/Makevars` and compiler-related environment
  variables;
- hardware information from `Sys.info()`, `lscpu`, `free -h`, and
  `nvidia-smi` where available;
- CUDA information from `nvcc --version`, `CUDA_HOME`, `CUDAHOSTCXX`,
  `FASTEMBEDR_CUDA_ARCH`, `FASTEMBEDR_CUDA_FLAGS`, `LD_LIBRARY_PATH`, and
  fastEmbedR native-backend probes;
- directly linked FAISS GPU/cuVS availability recorded by fastEmbedR;
- paths to the benchmark driver and wrapper scripts.

Release-locked runs additionally require the reviewed package tag and commit,
source-archive SHA-256, benchmark commit, container SHA-256, and permanent
result-archive DOI. The benchmark refuses an enforced release lock when any of
these fields is empty or when the installed package binary has the wrong
SHA-256. The final gate also rejects a successful comparator whose exact
version, Git commit, or executable checksum was not captured.

Each dataset has one deterministic stratified quality sample. Its one-based
row identifiers are written as both RDS and CSV in `fastEmbedR-input`, and the
CSV path and SHA-256 are copied into every result row. The same rows are reused
for all methods, backends, timing repetitions, and embedding seeds.

CUDA timing stops only after the native call or direct Python fit has
synchronized its device stream. GPU memory is sampled every 0.1 seconds with
`nvidia-smi`. The monitor records device-wide memory immediately before the
isolated worker starts, the observed device-wide peak, and their nonnegative
difference. This subtraction removes the measured allocator/runtime baseline
but cannot attribute allocations made by another process, so publication GPU
jobs request an exclusive device and retain all three values.

The publication benchmark driver
[`tools/hpc_embeddings/benchmark_embeddings_float32_publication.R`](https://github.com/tkcaccia/fastEmbedR-benchmark/blob/main/tools/hpc_embeddings/benchmark_embeddings_float32_publication.R)
also writes this bundle automatically into every HPC benchmark output
directory.

GPU-specific test evidence follows the
[real-hardware backend validation contract](backend-validation.md). The
release commands fail when Metal or CUDA is unavailable, when a result records
a backend different from the one requested, or when a backend-specific test
fails. These artifacts are strict hardware smoke/correctness evidence; full
performance validation is a separate release-locked benchmark, and configured
device architectures without execution are only build-level compatibility
targets.

## Benchmark Commands

Small reference validation:

```bash
Rscript tools/validate_reference_implementations.R \
  --out-dir=results/reference_validation_current \
  --threads=2 \
  --seed=4 \
  --perplexity=10 \
  --k=31
```

MNIST70k GitHub example:

```bash
Rscript tools/benchmark_github_mnist70k.R \
  --n=70000 \
  --k=30 \
  --perplexity=30 \
  --threads=4 \
  --run-metal=true \
  --run-cuda=false \
  --run-references=true \
  --out-dir=results/mnist70k_github_current
```

Publication CPU benchmark:

```bash
DATASETS=MNIST,FashionMNIST,flow18,mass41,imagenet,FlowRepository_FR-FCM-ZYRM_files,MetRef \
K=30 PERPLEXITY=15 TIMEOUT=10800 \
sbatch /scratch/firenze/NN/benchmark_embeddings_float32_cpu12.sh
```

Publication CUDA benchmark:

```bash
DATASETS=MNIST,FashionMNIST,flow18,mass41,imagenet,FlowRepository_FR-FCM-ZYRM_files,MetRef \
K=30 PERPLEXITY=15 TIMEOUT=10800 \
sbatch /scratch/firenze/NN/benchmark_embeddings_float32_cuda.sh
```

## Environment

The lightweight R-side environment file is:

```text
tools/reproducibility/benchmark_environment.yml
```

FAISS, RAPIDS cuVS, CUDA, cuFFT, and GPU driver versions are system-level
dependencies owned by the CUDA installation. They are therefore
not vendored into `fastEmbedR`; the exact versions actually used in a run are
captured in `reproducibility_manifest.txt` and `reproducibility_manifest.json`.

## Compiler Comparability

The package inherits optimization flags from R and adds only `-pthread` to the
portable C++ core. Benchmark reports must therefore include the resolved
compiler configuration, not only the package version. In particular, results
from a Conda R/GCC toolchain and a system R/GCC toolchain are not treated as a
controlled algorithm comparison even when they run on the same physical CPU.

The package does not require or recommend global `-march=native`,
`-ffast-math`, or `-O3`. CUDA deployment architectures are recorded
separately because package kernels and linked FAISS/cuVS/RAFT libraries must
all support the target GPU. See
[Installation And Native Compiler Configuration](installation-backends.md)
for the complete build contract.

## Seeds

The manuscript benchmark default seed is `4`. The small reference validation
also uses seed `4`. The GitHub MNIST example records its seed in the output
CSV and plot metadata.
