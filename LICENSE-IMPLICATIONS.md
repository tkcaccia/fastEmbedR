# License Implications Report

Date: 2026-08-25

This report summarizes the practical license implications for the current
`fastEmbedR` repository. It is a development compliance note, not legal advice.

## Current Package License

`fastEmbedR` currently declares:

```text
License: MIT + file LICENSE
```

The intended permissive-license posture is:

- core package code is implemented in package-local R, C++, Objective-C++,
  Metal, and CUDA sources;
- GPL packages may be used only as optional benchmark/reference tools, not as
  Imports, LinkingTo dependencies, vendored source, or required runtime code;
- optional external libraries such as cuVS, CUDA, cuFFT, and Apple Metal
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
`inst/ALGORITHMIC_REFERENCES.md`; the machine-readable source/dependency map is
`inst/THIRD_PARTY_DEPENDENCIES.json`. Current status:

| Source | License | Current use | MIT implication |
|---|---|---|---|
| UMAP paper / umap-learn | BSD-3-Clause implementation; algorithm paper | Mathematical reference | Compatible as algorithmic reference; no Python source vendored or called. |
| `uwot` | GPL (>= 3) | External R benchmark and behavioural reference only | Do not copy, vendor, link, or require source/runtime code in MIT package core. |
| `Rtsne` | BSD-style | KNN-input t-SNE validation/reference behaviour | Compatible as reference; old Barnes-Hut C++ files are not vendored. |
| FAISS | MIT | CPU HNSW and Metal IVF derivative | Compatible; exact derivative files and FAISS notice are retained. FAISS source/binaries are not bundled or linked. |
| Faiss-mlx | Apache-2.0 | Metal fused list-scan/top-k derivative | Compatible; Apache-2.0 source-specific terms and notice remain in force. No MLX/Python runtime. |
| `faissR` | MIT | Pinned source for the distilled native CUDA adapter | Compatible; `fastEmbedR` does not import, link, or call the `faissR` R package. |
| RAPIDS cuVS | Apache-2.0 | Optional direct CUDA C API linkage | Compatible as an external linked dependency; source/binaries are not bundled. |
| DLPack | Apache-2.0 | Reduced C ABI header redistributed for the cuVS bridge | Compatible with source header and complete notice retained. |
| openTSNE | BSD-3-Clause | Design reference for native openTSNE-style optimizer/transform | Compatible. Python/Cython source is not vendored or called. |
| t-SNE-CUDA | BSD-3-Clause | GPU architecture and FFT-grid design reference | Compatible. Source is not vendored or called. |
| AppleSiliconFFT | MIT | Adapted 512-point Stockham Metal organization | Compatible with source notice and complete MIT license retained. |
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
- Keep `uwot`, `umap`, `Rtsne`, and similar comparator packages in the
  separate benchmark environment and scripts, not in the package dependency
  graph unless a shipped package feature directly requires them.
- Keep optional GPU libraries explicit: no silent CPU fallback reported as GPU.
- Preserve upstream notices when permissive code is copied or substantially
  adapted, especially MIT/BSD/Apache code.
- Keep exact upstream commits, package files, upstream files, and linkage or
  redistribution status synchronized in `inst/THIRD_PARTY_DEPENDENCIES.json`.
- Run `Rscript tools/check_provenance_inventory.R` before each release.
- Keep generated benchmark outputs, private credentials, Kaggle tokens, and
  private datasets out of the repository.

## Publication Note

For a CRAN/R Journal-oriented permissive package, the strongest posture is:

- `DESCRIPTION` declares `MIT + file LICENSE`;
- core UMAP/openTSNE implementation is package-local and independently written;
- benchmark scripts clearly label external reference packages;
- `inst/NOTICE`, `inst/COPYRIGHTS`, `inst/THIRD_PARTY_DEPENDENCIES.json`,
  `inst/ALGORITHMIC_REFERENCES.md`, and this report stay current;
- license/provenance scans are run before release.
