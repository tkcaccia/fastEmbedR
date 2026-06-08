\name{knn_embed}
\alias{knn_embed}
\alias{knn_tsne}
\alias{knn_pacmap}
\alias{knn_trimap}
\alias{knn_localmap}
\title{Neighbor-embedding objectives from precomputed KNN}
\usage{
knn_embed(indices, distances, objective = c("tsne", "pacmap", "trimap", "localmap"),
  n_components = 2L, n_epochs = 200L, negative_sample_rate = 5L,
  learning_rate = 0.1, n_threads = 1L, init = c("spectral", "random"),
  spectral_n_iter = 25L, seed = 4L, verbose = FALSE)

knn_tsne(indices, distances, ...)

knn_pacmap(indices, distances, ...)

knn_trimap(indices, distances, ...)

knn_localmap(indices, distances, ...)
}
\arguments{
\item{indices}{Integer KNN index matrix without the self-neighbor column.}
\item{distances}{Numeric KNN distance matrix matching \code{indices}.}
\item{objective}{One of \code{"tsne"}, \code{"pacmap"}, \code{"trimap"}, or \code{"localmap"}.}
\item{n_components}{Output dimensionality.}
\item{n_epochs}{Number of optimization epochs.}
\item{negative_sample_rate}{Number of negative samples per positive edge.}
\item{learning_rate}{Initial learning rate.}
\item{n_threads}{Number of CPU threads for batch optimization.}
\item{init}{Initialization method.}
\item{spectral_n_iter}{Number of spectral power iterations when \code{init = "spectral"}.}
\item{seed}{Random seed.}
\item{verbose}{Print progress from C++.}
\item{...}{Additional arguments passed to \code{knn_embed}.}
}
\value{A numeric embedding matrix.}
\description{
Runs lightweight KNN-driven neighbor-embedding objectives in C++. These are
practical objective variants for benchmarking from the same KNN input rather
than drop-in reimplementations of full reference packages.
}
