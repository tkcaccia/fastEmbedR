## Test environments

- Local macOS Sonoma 14.5, aarch64-apple-darwin23, R 4.6.0

## R CMD check results

0 errors | 0 warnings | 1 note

- This is a new submission.

## Optional system libraries

The package can optionally use FAISS, RAPIDS cuVS, CUDA, and Apple Metal when
they are available at build time. These libraries are not vendored and are not
required for the default CRAN build. If optional GPU or FAISS libraries are not
found, the package builds CPU/stub paths and explicit unavailable backend
requests fail clearly at run time.

## Downstream and optional packages

Optional graph clustering support uses `igraph` and `leidenbase` only when they
are installed. Examples and tests guard optional functionality with
`requireNamespace()`.
