# Bioconductor Readiness

[Home](../README.md) |
[Installation](installation.md) |
**Bioconductor** |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
[Reproducibility](reproducibility.md) |
[References](references.md)

This page records the dependency boundary used for a Bioconductor-friendly
submission of `fastEmbedR`.

## Current Review Status

`fastEmbedR` is under review in
[BiocContributions issue 142](https://github.com/Bioconductor/BiocContributions/issues/142).
The local version 0.99.13 source check completed with 0 errors and 0 warnings.
`R CMD check --as-cran --no-manual` reports only the expected new-submission
note. BiocCheck reports five notes and no errors or warnings. The package code,
manuals, namespace, and vignette sources now pass its coding-practice and
formatting checks. The remaining findings are described below.

> No Bioconductor dependencies detected. Note that some infrastructure
> packages may not have Bioconductor dependencies.

This is a submission-scope question rather than a compilation or correctness
failure. `fastEmbedR` is intended as infrastructure for dimensionality
reduction of large biological data, including single-cell, flow-cytometry,
gene-expression, and other high-dimensional assays. Its native CPU path is
deliberately standalone, however, and no Bioconductor package provides a
legitimate mandatory runtime API dependency. An artificial dependency will
therefore not be added merely to silence the check. Eligibility and any
required scope clarification should be decided with the assigned
Bioconductor reviewer. Version 0.99.10 therefore declares the accurate
`Infrastructure` biocView. BiocCheck treats dependency-independent
infrastructure packages as a note rather than a warning; no artificial package
dependency is added.

The remaining BiocCheck notes are triaged as follows:

- the automatically suggested `ATACSeq` and `DNASeq` views are not used because
  fastEmbedR is not specific to either assay;
- the package is deliberately independent of Bioconductor runtime packages and
  does not add an artificial dependency solely to silence the infrastructure
  note;
- `Authors@R` includes the verified maintainer ORCID; a `fnd` role will be
  added only if a genuine funder or grant is verified;
- the function-length recommendation is tracked as a staged maintainability
  refactor because splitting backend orchestration requires CPU, Metal, and
  CUDA regression testing and is not a formatting-only change;
- the local mailing-list subscription probe cannot be completed without
  Bioconductor administrative credentials. The maintainer must verify the
  subscription externally.

The previously reported line-width and indentation notes are resolved: all
files covered by BiocCheck use lines of at most 80 columns and indentation in
multiples of four spaces. Directly constructed condition messages were also
rewritten so the coding-practice check passes.

## Dependency Classes

| Class | Dependency | Role | Required For Core Build |
| --- | --- | --- | --- |
| R package | `Rcpp` | R/C++ interface and exported native routines. | yes |
| R package | `float` | Optional float32 R matrices and reduced host memory use. | suggested |
| R package | `jsonlite` | Optional benchmark and reproducibility metadata serialization. | suggested |
| R package | `knitr`, `rmarkdown` | Vignette and documentation rendering. | suggested |
| R package | `testthat` | Unit tests. | suggested |
| R package | `igraph` | Graph-clustering correctness tests against a reference implementation. | suggested |
| R package | `RhpcBLASctl` | Optional CPU BLAS and OpenMP thread control. | suggested |
| System library | C++17 compiler | Native CPU code and numerical helper compilation. | yes |
| System library | Apple Metal framework | Native Metal KNN and embedding backends on macOS. | optional |
| System library | CUDA Toolkit, cuFFT, cuBLAS, cuSOLVER, and RAPIDS cuVS | Native CUDA KNN and embedding backend. | optional |
| System library | RAPIDS RAFT/RMM or FAISS GPU | CUDA TSVD initialization or alternative exact KNN, respectively. | optional |

Reference packages such as `Rtsne`, `uwot`, and `umap` are installed only in
the separate benchmark environment; they are not part of the package
dependency graph.

`fastEmbedR` does not vendor the full FAISS, cuVS, RAFT, or cuML libraries, or
`uwot`, `Rtsne`, or Python openTSNE source. Its compact FAISS-derived HNSW and
Faiss-mlx-informed Metal files retain their permissive licenses under
`inst/LICENSES/`.

## Native KNN Boundary

fastEmbedR owns the internal CPU/Metal KNN and direct cuVS CUDA KNN used
by one-call embeddings. The following are available without another KNN R
package:

- `tsne_knn()` works from supplied neighbor indices and distances;
- `umap_knn()` works from supplied neighbor indices and distances;
- `prepare_tsne_knn()` and `prepare_umap_knn()` can prepare reusable
  native embedding inputs;
- `knn_graph()` and `graph_cluster()` provide native graph construction and
  community detection;
- `evaluate_embedding()` and plotting helpers use package-native routines.

CPU, Metal, and a correctly compiled CUDA one-call build are self-contained at
the R package level.
CUDA requests fail explicitly when cuVS is not linked.

## Backend Policy

The public embedding backend argument is intentionally small:

```r
backend = "cpu"
backend = "metal"
backend = "cuda"
```

An explicit GPU request must use the requested GPU backend. The package must
not run on CPU while reporting Metal or CUDA. Optional GPU code is compiled and
tested when the corresponding toolchain is available.

## Submission Checklist

- No `Remotes` field in `DESCRIPTION`.
- No vendored FAISS/cuVS/RAPIDS source or binary libraries in the R package.
- Large data, benchmark outputs, and container images are excluded by
  `.Rbuildignore`.
- Examples and vignettes use small built-in data or guard optional packages
  with `requireNamespace()`.
- Optional reference benchmarks (`Rtsne`, `uwot`, `umap`) are isolated in the
  external benchmark environment rather than declared as package dependencies.
- CUDA and Metal failures are explicit, not silent CPU fallbacks.
- The maintainer email should be registered on the Bioconductor Support Site
  before submission.

## Minimal Bioconductor Check

A CPU-only check should be possible without FAISS/cuVS installed:

```sh
LC_ALL=C \
FASTEMBEDR_USE_CUDA=0 R CMD build .

LC_ALL=en_US.UTF-8 \
FASTEMBEDR_USE_CUDA=0 R CMD check --as-cran fastEmbedR_0.99.13.tar.gz
```

GPU-enabled builds should be validated separately on machines with the relevant
toolchains, because Bioconductor build machines should not be assumed to have
CUDA or Apple Metal.

The local submission preflight used for this repository is:

```sh
LC_ALL=en_US.UTF-8 \
FASTEMBEDR_USE_CUDA=0 \
R CMD check --as-cran --no-manual fastEmbedR_0.99.13.tar.gz

LC_ALL=en_US.UTF-8 \
Rscript -e 'BiocCheck::BiocCheck("fastEmbedR_0.99.13.tar.gz")'
```

The final submission check builds and rebuilds the vignette. The local
BiocCheck run used the same source archive; an external Bioconductor index
probe can still depend on network availability.

Current Bioconductor-specific follow-up items:

- register and validate the maintainer email on the Bioconductor Support Site;
- obtain the assigned reviewer's decision on eligibility without a mandatory
  Bioconductor dependency;
- keep the verified maintainer ORCID in `Authors@R`;
- add a `fnd` role only when funder metadata are verified;
- gradually split long backend-orchestration functions while preserving
  CPU, Metal, and CUDA behavior.
