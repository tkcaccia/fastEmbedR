# References

[Home](../README.md) |
[Installation](installation.md) |
[Implementation](implementation.md) |
[Examples](examples.md) |
[Benchmarks](benchmarks.md) |
[API](usage-api.md) |
**References**

References are listed in AACR journal style. Software projects are included
when they influenced design, benchmarking, or backend engineering.

1. van der Maaten L, Hinton G. Visualizing data using t-SNE. J Mach Learn Res 2008;9:2579-2605.
2. van der Maaten L. Accelerating t-SNE using tree-based algorithms. J Mach Learn Res 2014;15:3221-3245.
3. Linderman GC, Rachh M, Hoskins JG, Steinerberger S, Kluger Y. Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq data. Nat Methods 2019;16:243-245.
4. Policar PG, Strazar M, Zupan B. openTSNE: a modular Python library for t-SNE dimensionality reduction and embedding. J Open Source Softw 2019;4:1576.
5. Chan DM, Rao R, Huang F, Canny JF. t-SNE-CUDA: GPU-accelerated t-SNE and its applications to modern data. arXiv 2018;1807.11824.
6. Belkina AC, Ciccolella CO, Anno R, Halpert R, Spidlen J, Snyder-Cappione JE. Automated optimized parameters for T-distributed stochastic neighbor embedding improve visualization and analysis of large datasets. Nat Commun 2019;10:5415.
7. McInnes L, Healy J, Melville J. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv 2018;1802.03426.
8. Johnson J, Douze M, Jegou H. Billion-scale similarity search with GPUs. IEEE Trans Big Data 2021;7:535-547.
9. RAPIDS Development Team. RAPIDS cuVS: GPU-accelerated vector search and clustering [software]. Available from: https://github.com/rapidsai/cuvs.
10. Melville J. uwot: The Uniform Manifold Approximation and Projection method for dimensionality reduction [software]. Available from: https://github.com/jlmelville/uwot.
11. Krijthe JH. Rtsne: T-distributed stochastic neighbor embedding using Barnes-Hut implementation [software]. Available from: https://github.com/jkrijthe/Rtsne.
12. amine m. AppleSiliconFFT: FFT kernels for Apple Silicon GPUs [software]. Available from: https://github.com/aminems/AppleSiliconFFT.
13. McInnes L, Healy J, Saul N, Grossberger L. UMAP: Uniform Manifold Approximation and Projection. J Open Source Softw 2018;3:861.

## Software Provenance Notes

- `uwot` and `Rtsne` are benchmark/reference implementations only. Their source
  code is not vendored into `fastEmbedR`.
- FAISS and RAPIDS cuVS are used through the companion `faissR` package for
  nearest-neighbour search.
- openTSNE, FIt-SNE, t-SNE-CUDA, and opt-SNE informed the t-SNE optimization
  design. The public package implementation is native C++/Metal/CUDA code.
- AppleSiliconFFT informed the native Metal FFT-grid engineering. The package
  keeps its own backend surface and does not expose experimental FFT variants.
