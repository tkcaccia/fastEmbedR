# Third-party license copies

This directory contains complete upstream license texts required for source
that is adapted or redistributed by `fastEmbedR`, plus the cuVS license for an
important optional linked backend. The exact source files, upstream commits,
and relationship to fastEmbedR are recorded in `../NOTICE`, `../COPYRIGHTS`,
and `../THIRD_PARTY_DEPENDENCIES.json`.

| File | Component | Revision | Relationship |
|---|---|---|---|
| `FASTEMBEDR-LICENSE` | fastEmbedR original source | package release | Full MIT text retained in installed/source archives; `LICENSE` remains the R-standard MIT declaration |
| `FAISS-LICENSE` | FAISS 1.14.3 | `0ca9df4792b173d573044ee14ca0704780176e82` | CPU HNSW and Metal IVF derivative |
| `FAISS-MLX-LICENSE` | MLXPorts/Faiss-mlx | `d092af559375144fc719cd88a10e414f92c625fa` | Metal list-scan/top-k derivative |
| `FAISSR-LICENSE` | faissR 0.99.15 | `f37ea97c5774200025b1480770b8ecbf1d2d7919` | Native CUDA adapter derivative |
| `DLPACK-LICENSE` | DLPack 1.0 | `bbd2f4d32427e548797929af08cfe2a9cbb3cf12` | Reduced vendored C ABI header |
| `APPLESILICONFFT-LICENSE` | AppleSiliconFFT | `5d0d51dbd983691ee99822ed74bc3f9a47136511` | Metal 512-point Stockham derivative |
| `NETWORKIT-LICENSE` | NetworKit | `7b74f6af90bc0865c6c0937a206df63df331b712` | Conservatively retained design-reference notice |
| `CUVS-LICENSE` | RAPIDS cuVS | build-resolved | Optional linked CUDA dependency; not bundled |

Dependencies that are only linked from the user's build environment, system
frameworks, design references, and benchmark-only software are listed with
their SPDX expression and upstream license URL in the JSON inventory. Their
source trees and binaries are not redistributed in the R source package.
