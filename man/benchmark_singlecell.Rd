\name{benchmark_singlecell}
\alias{benchmark_singlecell}
\title{Benchmark fastknnumap on the single-cell example}
\usage{
benchmark_singlecell(data_path = "/Users/stefano/Documents/GPUPLS/Data/singlecell.RData",
  k = 30L, n_features = 20L, mode = c("hybrid", "spectral", "sgd"),
  n_epochs = 100L, spectral_n_iter = 25L, silhouette_sample = NULL,
  seed = 4L, run_umap = TRUE)
}
\arguments{
\item{data_path}{Path to \code{singlecell.RData}.}
\item{k}{Number of nearest neighbors requested, including self.}
\item{n_features}{Number of leading feature columns to use.}
\item{mode}{Embedding mode passed to \code{fast_knn_umap}.}
\item{n_epochs}{Number of optimization epochs for \code{"sgd"} or \code{"hybrid"} modes.}
\item{spectral_n_iter}{Number of spectral power iterations.}
\item{silhouette_sample}{Optional sample size for silhouette scoring. Use \code{NULL} for a full silhouette.}
\item{seed}{Random seed.}
\item{run_umap}{Compare against \code{umap}'s precomputed-KNN path.}
}
\value{A list with timings, silhouette scores, labels, and layouts.}
\description{
Loads the single-cell example data, uses the first \code{n_features} columns,
computes nearest neighbors with \code{nn}, embeds with
\code{fast_knn_umap}, and optionally compares against \code{umap}'s
precomputed-KNN path.
}
