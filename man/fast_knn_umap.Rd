\name{fast_knn_umap}
\alias{fast_knn_umap}
\alias{umap_knn}
\title{Fast UMAP from precomputed nearest neighbors}
\usage{
fast_knn_umap(indices, distances, n_components = 2L, n_epochs = NULL,
  min_dist = 0.1, spread = 1, local_connectivity = 1,
  set_op_mix_ratio = 1, negative_sample_rate = 5L, learning_rate = 1,
  a = NULL, b = NULL, repulsion_strength = 1,
  mode = c("sgd", "spectral", "hybrid"), init = c("spectral", "random"),
  init_sdev = NULL, prune_epochs = TRUE, spectral_n_iter = 50L,
  seed = 42L, verbose = FALSE)

umap_knn(indices, distances, n_components = 2L, n_epochs = NULL,
  min_dist = 0.1, spread = 1, local_connectivity = 1,
  set_op_mix_ratio = 1, negative_sample_rate = 5L, learning_rate = 1,
  a = NULL, b = NULL, repulsion_strength = 1,
  mode = c("sgd", "spectral", "hybrid"), init = c("spectral", "random"),
  init_sdev = NULL, prune_epochs = TRUE, spectral_n_iter = 50L,
  seed = 42L, verbose = FALSE)
}
\arguments{
\item{indices}{Integer matrix of nearest-neighbor indices, one row per point.}
\item{distances}{Numeric matrix of nearest-neighbor distances matching \code{indices}.}
\item{n_components}{Output dimensionality.}
\item{n_epochs}{Number of optimization epochs.}
\item{min_dist}{UMAP minimum distance.}
\item{spread}{UMAP spread parameter.}
\item{local_connectivity}{Local connectivity for KNN distance smoothing.}
\item{set_op_mix_ratio}{Fuzzy set union/intersection mix ratio.}
\item{negative_sample_rate}{Negative samples per positive sample.}
\item{learning_rate}{Initial learning rate.}
\item{a, b}{Optional UMAP curve parameters. If \code{NULL}, they are estimated from \code{min_dist} and \code{spread}.}
\item{repulsion_strength}{Weight applied to negative samples.}
\item{mode}{Embedding mode. \code{"sgd"} runs UMAP SGD, \code{"spectral"} returns the normalized graph-Laplacian spectral embedding, and \code{"hybrid"} uses spectral initialization with fewer default epochs.}
\item{init}{Initialization method. \code{"spectral"} uses a randomized eigensolver on the normalized graph; \code{"random"} uses uniform random coordinates.}
\item{init_sdev}{Optional initialization scaling. Use \code{NULL} for native spectral scaling, \code{"range"} for Python UMAP-compatible 0-10 range scaling, or a positive scalar to scale each initialized dimension to that standard deviation.}
\item{prune_epochs}{Remove graph edges too weak to be sampled at least once over \code{n_epochs}.}
\item{spectral_n_iter}{Number of power iterations for spectral initialization.}
\item{seed}{Integer random seed.}
\item{verbose}{Print optimization progress.}
}
\value{A numeric embedding matrix.}
\description{
Builds a fuzzy UMAP graph from precomputed KNN index and distance matrices and
optimizes the low-dimensional embedding in C++.
}
