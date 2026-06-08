\name{landmark_knn_umap}
\alias{landmark_knn_umap}
\alias{landmark_knn_graph}
\title{Landmark UMAP from precomputed nearest neighbors}
\usage{
landmark_knn_umap(indices, distances, n_landmarks = NULL,
  landmark_ratio = 0.1, landmark_k = 10L, local_k = 5L,
  selection = c("hub", "random"), mode = c("hybrid", "sgd", "spectral"), n_epochs = 100L,
  spectral_n_iter = 25L, seed = 4L, ...)

landmark_knn_graph(indices, distances, n_landmarks = NULL,
  landmark_ratio = 0.1, landmark_k = 10L, local_k = 5L,
  selection = c("hub", "random"), seed = 4L)
}
\arguments{
\item{indices}{Integer KNN index matrix without the self-neighbor column.}
\item{distances}{Numeric KNN distance matrix matching \code{indices}.}
\item{n_landmarks}{Number of landmark/hub points to select.}
\item{landmark_ratio}{Landmark fraction used when \code{n_landmarks} is \code{NULL}.}
\item{landmark_k}{Maximum landmark neighbors retained per point.}
\item{local_k}{Number of original local KNN edges retained per point.}
\item{selection}{Landmark selection strategy.}
\item{mode}{Embedding mode passed to \code{fast_knn_umap}.}
\item{n_epochs}{Number of optimization epochs.}
\item{spectral_n_iter}{Number of spectral power iterations.}
\item{seed}{Random seed.}
\item{...}{Additional arguments passed to \code{fast_knn_umap}.}
}
\value{
\code{landmark_knn_umap} returns an embedding matrix with landmark metadata
attached as attributes. \code{landmark_knn_graph} returns a list containing
point-landmark KNN matrices and selected landmark indices.
}
\description{
Selects landmark points from a precomputed KNN graph, builds a sparse
point-landmark graph, and embeds all points through the existing C++ UMAP
optimizer. Hub selection prioritizes points that appear frequently in the KNN
matrix while encouraging one-hop coverage.
}
