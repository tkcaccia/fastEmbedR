\name{benchmark_embedding_datasets}
\alias{benchmark_embedding_datasets}
\title{Benchmark embeddings across multiple public datasets}
\usage{
benchmark_embedding_datasets(datasets = c("iris", "pendigits", "fashion_mnist"),
  subsets = c(iris = NA, pendigits = NA, fashion_mnist = 2000),
  implementations = c("fastknnumap_sgd", "umap", "rtsne"),
  pca_dims = 50L, repeats = 1L,
  cache_dir = file.path(tempdir(), "fastembedr-data"), output_csv = NULL, ...)
}
\arguments{
\item{datasets}{Character vector of dataset names.}
\item{subsets}{Named numeric vector or list with per-dataset row subsets. Use \code{NA} for the full loaded dataset.}
\item{implementations}{Implementations passed to \code{benchmark_knn_umap}, including \code{"umap"} and \code{"rtsne"}.}
\item{pca_dims}{Optional PCA dimensionality for high-dimensional datasets.}
\item{repeats}{Number of seed repeats. Repeats after the first include a Procrustes stability score against the first run.}
\item{cache_dir}{Directory for cached downloaded files.}
\item{output_csv}{Optional path to save the combined metrics table.}
\item{...}{Additional arguments passed to \code{benchmark_knn_umap}.}
}
\value{A list with combined \code{metrics} and per-dataset \code{results}.}
\description{
Runs the same benchmark over multiple public datasets and combines runtime,
silhouette, and KNN-preservation metrics. This is intended for comparing the
native fastknnumap implementations with R reference packages such as
\code{umap} and \code{Rtsne}.
}
