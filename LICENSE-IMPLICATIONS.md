# License Implications Report

Date: 2026-06-18

This report summarizes the practical license implications for the current
`fastEmbedR` repository. It is a development compliance note, not legal advice.

## Current Package License

`fastEmbedR` currently declares:

```text
License: MIT + file LICENSE
```

The intended permissive-license posture is:

- core package code is implemented in package-local R, C++, Objective-C++,
  Metal, CUDA, and Fortran sources;
- GPL packages may be used only as optional benchmark/reference tools, not as
  Imports, LinkingTo dependencies, vendored source, or required runtime code;
- optional external libraries such as FAISS, cuVS, CUDA, cuFFT, and Apple Metal
  are linked or used only when available and are reported explicitly.

## Package Core Versus Benchmark Code

The package core consists of files installed by `R CMD build`: `R/`, `src/`,
`inst/`, vignettes, DESCRIPTION, NAMESPACE, and license files. These files must
be compatible with MIT distribution.

Benchmark and development scripts in `tools/`, external result folders, and
paper-generation scripts may call reference implementations such as `uwot`,
`umap`, `Rtsne`, `tsne`, and FIt-SNE for comparison. Those scripts must make it
clear that the reference packages are optional benchmark dependencies and not
part of the `fastEmbedR` core implementation.

## Third-Party Provenance And Compatibility

The detailed provenance log is in `inst/NOTICE` and
`inst/ALGORITHMIC_REFERENCES.md`. Current status:

| Source | License | Current use | MIT implication |
|---|---|---|---|
| UMAP paper / umap-learn | BSD-3-Clause implementation; algorithm paper | Mathematical reference | Compatible as algorithmic reference; no Python source vendored or called. |
| `uwot` | GPL (>= 3) | External R benchmark and behavioural reference only | Do not copy, vendor, link, or require source/runtime code in MIT package core. |
| `Rtsne` | BSD-style | KNN-input t-SNE validation/reference behaviour | Compatible as reference; old Barnes-Hut C++ files are not vendored. |
| FAISS | MIT | Required by companion `faissR`; optional KNN provider via wrapper | Compatible. FAISS source is not vendored in `fastEmbedR`. |
| RAPIDS cuVS | Apache-2.0 | Optional KNN provider through `faissR`; CUDA design reference | Compatible with MIT use as external dependency/reference. |
| DLPack | Apache-2.0 | Minimal C ABI compatibility header for cuVS bridge where needed | Compatible with notice retained. |
| openTSNE | BSD-3-Clause | Design reference for native openTSNE-style optimizer/transform | Compatible. Python/Cython source is not vendored or called. |
| t-SNE-CUDA | BSD-3-Clause | GPU architecture and FFT-grid design reference | Compatible. Source is not vendored or called. |
| AppleSiliconFFT | MIT | Design/source reference for native Metal Stockham FFT kernels | Compatible; retain MIT attribution in NOTICE. |
| mlx-vis | Apache-2.0 | Apple GPU design reference | Compatible as design reference; no MLX/Python runtime. |
| annembed | MIT OR Apache-2.0 | Design reference | Compatible as design reference. |
| opt-SNE / Multicore-opt-SNE | BSD-3-Clause | Automatic t-SNE parameter design reference | Compatible as design reference. |

## Fast Power Approximation

The UMAP optimizers use package-local positive-power approximations based on
IEEE-754 exponent interpolation. The documented provenance is:

- Nicol N. Schraudolph, "A Fast, Compact Approximation of the Exponential
  Function", Neural Computation, 1999.
- Additional permissive prior art reviewed: Harrison Ainsworth / HXA7241
  fast power approximation material under a new-BSD-style license.

The implemented helpers are local expressions written for `fastEmbedR`:

- `src/fast_knn_umap.cpp::umap_pow`
- `src/fast_knn_umap.cpp::umap_powf_fast`
- `src/embedding_metal_impl.mm::fast_positive_pow`
- `src/embedding_cuda_kernels.cpp::fast_positive_pow`

They are not copied from `uwot`, blog union snippets, or vendored third-party
source. If maximum legal simplicity is ever preferred over speed, these helpers
can be replaced with `std::pow`/backend-native `pow` after benchmarking.

## Required Ongoing Rules

- Do not claim a GPL package implementation is inside `fastEmbedR`.
- Do not copy or closely adapt `uwot` source while keeping `fastEmbedR` MIT.
- Keep `uwot`, `umap`, `Rtsne`, and similar R packages in `Suggests` or
  benchmark scripts only.
- Keep optional GPU libraries explicit: no silent CPU fallback reported as GPU.
- Preserve upstream notices when permissive code is copied or substantially
  adapted, especially MIT/BSD/Apache code.
- Keep generated benchmark outputs, private credentials, Kaggle tokens, and
  private datasets out of the repository.

## Publication Note

For a CRAN/R Journal-oriented permissive package, the strongest posture is:

- `DESCRIPTION` declares `MIT + file LICENSE`;
- core UMAP/openTSNE implementation is package-local and independently written;
- benchmark scripts clearly label external reference packages;
- `inst/NOTICE`, `inst/ALGORITHMIC_REFERENCES.md`, and this report stay current;
- license/provenance scans are run before release.
