# Benchmark Gallery

This gallery collects plots intended for later manuscript figures. The MNIST
70k test outputs are documented only on the [Examples](examples.md) page so the
current runtime table, machine specification, and figures stay in one place.

## How To Add A Dataset Panel

For every new dataset added to this gallery, include:

- the embedding plot for CPU, Metal, and CUDA where available;
- the reference R package plot when relevant, for example `uwot` or `Rtsne`;
- a timing table with nearest-neighbour, embedding, and projection/transform
  columns separated;
- trustworthiness and label KNN accuracy when labels are available;
- a short note about whether the method used full embedding or landmarking.

Recommended next gallery datasets:

- Fashion-MNIST 70k;
- Shuttle;
- Covertype;
- opt-SNE paper Flow18 and Mass41 cytometry datasets;
- SingleCell/MetRef on the CUDA workstation;
- CIFAR-style image features.

The multi-dataset runner is documented in
[extended-benchmark-suite.md](extended-benchmark-suite.md).

## Shuttle 58k: Extended Local Benchmark

The extended runner now includes a Shuttle benchmark with CPU, Metal,
MPSGraph diagnostic, `Rtsne_neighbors`, and `uwot::umap(fast_sgd = TRUE)`.

![Shuttle 58k extended embedding gallery](assets/extended-shuttle58k-embedding-gallery.png)

![Shuttle 58k extended timing](assets/extended-shuttle58k-timing-stacked.png)

See [extended-benchmark-suite.md](extended-benchmark-suite.md) for the table
and interpretation.
