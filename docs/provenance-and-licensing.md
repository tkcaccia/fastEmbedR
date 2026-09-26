# Source Provenance And Licensing

`fastEmbedR` is distributed under the MIT license. Repository-level licensing
statements are backed by file-level records so that adapted, linked, system,
and validation-only components can be audited independently.

## Audit Records

The source repository contains the following records:

- [`LICENSE`](../LICENSE) and [`LICENSE.md`](../LICENSE.md): the package's MIT
  license declaration and full license text.
- [`inst/NOTICE`](../inst/NOTICE): component-by-component provenance and
  attribution, including how each component is used.
- [`inst/COPYRIGHTS`](../inst/COPYRIGHTS): copyright holders, licenses, and
  mapped package files for adapted source.
- [`inst/LICENSES/`](../inst/LICENSES/): required upstream license and notice
  copies.
- [`inst/THIRD_PARTY_DEPENDENCIES.json`](../inst/THIRD_PARTY_DEPENDENCIES.json):
  a machine-readable SPDX inventory.
- [`inst/ALGORITHMIC_REFERENCES.md`](../inst/ALGORITHMIC_REFERENCES.md):
  scientific and software references that informed implementation and
  validation.

The machine-readable inventory records the upstream repository and pinned
revision, maps adapted code to package and upstream files, identifies the SPDX
license, and states whether source or binaries are redistributed. The release
audit in `tools/check_provenance_inventory.R` fails when a required revision,
file mapping, SPDX header, notice, or license copy is missing.

## Adapted And Linked Components

The native nearest-neighbor implementation retains notices for FAISS-derived
HNSW and IVF organization and the exact-search behavior adapted from faissR.
Metal list scanning retains the Faiss-mlx Apache-2.0 notice. Reduced DLPack
declarations and the AppleSiliconFFT-derived Stockham organization retain
their upstream notices. The precise upstream commits and package-file mappings
are recorded in the inventory and notice files above.

CUDA libraries and Apple system frameworks are resolved from the build or
runtime environment and are not redistributed as package binaries. Optional
cuVS, RAFT, RMM, CUDA, Metal, MPS, and Accelerate components retain
their upstream licenses and remain subject to their respective platform terms.

Software used only as a benchmark comparator or numerical reference remains
outside the production call graph. Benchmark datasets retain their original
licenses and access conditions. The separate
[`fastEmbedR-extra`](https://github.com/tkcaccia/fastEmbedR-extra)
repository licenses its scripts and documentation; it does not relicense
third-party software or datasets.

## Verification

For a release audit, inspect the human-readable notice together with the SPDX
inventory and run:

```r
Rscript tools/check_provenance_inventory.R
```

This page is a provenance summary, not legal advice. The installed license and
notice files are the authoritative distribution records.
