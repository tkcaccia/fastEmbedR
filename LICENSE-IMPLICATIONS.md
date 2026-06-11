# License Implications Report

Date: 2026-06-11

This report summarizes the practical license implications for the current
fastEmbedR repository. It is a development compliance note, not legal advice.

## Current Package License

fastEmbedR currently declares:

```text
License: GPL (>= 3)
```

This is the correct license posture for the current codebase because the UMAP
implementation was intentionally developed to match and adapt behaviour from
GPL-compatible R implementations, especially `uwot`.

Practical implication:

- The package can be published as a free R package on GitHub, CRAN, and in an
  R Journal workflow.
- Users may use the package in academic, clinical, and commercial settings.
- If someone distributes fastEmbedR or a modified derivative, they must keep
  the distribution GPL-compatible and provide the corresponding source code.
- The project should not be advertised as MIT/permissive while GPL-adapted
  implementation code remains in the package.

## Repository Scripts

The benchmark and development scripts in `tools/`, examples, vignettes, native
sources, and R sources are part of the GitHub distribution unless a file says
otherwise. They therefore inherit the repository/package GPL-3-or-later
licensing policy.

The R build excludes `tools/`, `examples/`, `.github/`, and this report through
`.Rbuildignore`, so they are not part of the CRAN-style source tarball unless
that choice is changed later. They are still distributed by GitHub pushes.

## Third-Party Provenance And Compatibility

The detailed provenance log is in `inst/NOTICE` and
`inst/ALGORITHMIC_REFERENCES.md`. Current status:

| Source | License | Current use | Implication |
|---|---|---|---|
| `uwot` | GPL (>= 3) | UMAP behaviour and fast-SGD scheduling studied/adapted | Keeps fastEmbedR GPL-compatible; incompatible with MIT-only distribution. |
| `Rtsne` | BSD-style | R-level KNN API/defaults studied; Barnes-Hut C++ not vendored | Compatible if notices are kept; avoid copying the old Delft advertising-clause C++ files without explicit review. |
| FAISS | MIT | Optional external C++ link backend; no vendored FAISS source | Compatible with GPL-3-or-later. Keep attribution if source is copied later. |
| RAPIDS cuML/cuVS | Apache-2.0 | Optional external cuVS link backend and design reference | Apache-2.0 is GPLv3-compatible. Preserve notices if source is copied later. |
| `mlx-vis` | Apache-2.0 | NN-descent schedule ideas adapted into native R/C++/Metal code | Compatible with GPLv3 if copied with notices; current code is native and records provenance. |
| `annembed` | MIT OR Apache-2.0 | Design reference only | Compatible. Keep original notices if implementation code is copied later. |
| KeOps | MIT | Design reference for blocked map-reduce exact t-SNE repulsion | Compatible. No runtime dependency or vendored code currently. |
| TorchDR | BSD-3-Clause | InfoTSNE/negative-sampling design reference | Compatible. No Python/PyTorch runtime dependency currently. |
| openTSNE | BSD-3-Clause | Transform and Barnes-Hut/negative-force design reference | Compatible. No Python/Cython source vendored currently. |
| t-SNE-CUDA | BSD-3-Clause | GPU t-SNE architecture design reference | Compatible. No CannyLab source vendored currently. |

## Optional Native Backends

The optional CUDA, Metal, FAISS, and cuVS paths must fail clearly when the
required library/backend is unavailable. This is a license and reproducibility
issue as much as a technical issue: a benchmark row must not claim that GPL
package code used a GPU implementation when it silently fell back to CPU or to
an unreported external library.

Current policy:

- FAISS and cuVS are linked only when explicitly available at build time.
- FAISS and cuVS source code is not vendored.
- Metal and CUDA code in `src/` is fastEmbedR package code and is therefore
  distributed under GPL-3-or-later.
- Apple Metal and CUDA toolkit/system SDK headers are treated as external
  system/build dependencies, not copied package source.

## What We Can Safely Do Next

- Continue adapting GPL-compatible code from `uwot`, as long as fastEmbedR
  remains GPL-3-or-later and provenance comments are kept.
- Copy or adapt MIT/BSD-3/Apache-2.0 implementation pieces only when useful,
  preserving upstream copyright headers and license notices next to the copied
  code.
- Keep `inst/NOTICE` updated whenever an implementation moves from
  "inspired by" to "adapted from" or "copied from".
- Keep generated benchmark outputs, private credentials, Kaggle tokens, and
  private datasets out of the repository.

## What We Should Avoid

- Do not claim the package is MIT/permissive while GPL-adapted code remains.
- Do not copy Rtsne's old Barnes-Hut C++ files without a focused license
  compatibility review.
- Do not vendor FAISS, cuVS, cuML, mlx-vis, openTSNE, TorchDR, KeOps, or
  t-SNE-CUDA source without retaining their notices and updating
  `inst/NOTICE`.
- Do not include private datasets, remote machine credentials, Kaggle tokens,
  or generated result folders in GitHub.

## Publication Note

For an R Journal or CRAN-style submission, GPL-3-or-later is acceptable for a
free R package. The strongest publication posture is:

- `DESCRIPTION` declares `GPL (>= 3)`.
- `README.md` briefly explains provenance.
- `inst/NOTICE` records third-party design/code provenance.
- `LICENSE-IMPLICATIONS.md` remains a GitHub-facing development note and is
  excluded from the R build.
