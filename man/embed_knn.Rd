\name{embed_knn}
\alias{embed_knn}
\title{Embed from precomputed KNN}
\usage{
embed_knn(indices, distances = NULL, method = "umap", n_components = 2L,
  seed = 4L, verbose = FALSE,
  backend = c("auto", "cpu", "gpu", "metal", "cuda"), ...)
}
\arguments{
\item{indices}{A KNN object returned by \code{nn()}, or an integer KNN index
matrix. If a self-neighbor first column is present it is removed automatically.}
\item{distances}{Numeric KNN distance matrix matching \code{indices}. Leave as
\code{NULL} when \code{indices} is an \code{nn()} result.}
\item{method}{Embedding method. Use \code{"umap"}, \code{"tsne"}, or
\code{"infotsne"}.}
\item{n_components}{Output dimensionality.}
\item{seed}{Random seed.}
\item{verbose}{Print native optimizer progress.}
\item{backend}{Execution backend. \code{"gpu"} explicitly requests a real
native GPU backend and errors if CUDA or Metal is unavailable.}
\item{...}{Additional method-specific parameters. For \code{method = "tsne"}
or \code{"infotsne"}, supported parameters include \code{perplexity},
\code{max_iter}, \code{Y_init}, learning-rate controls, exaggeration controls,
and \code{n_threads}.}
}
\value{
A numeric embedding matrix with resolved settings stored in
\code{attr(layout, "fastEmbedR_config")}.
}
\description{
Use an existing KNN graph as input. This is the preferred comparison path
because implementations can consume the same neighbor indices and distances.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
knn <- nn(x, k = 16)
layout <- embed_knn(knn, method = "umap", seed = 4)
plot(layout, pch = 21, bg = iris$Species)

layout_tsne <- embed_knn(knn, method = "tsne", max_iter = 250, seed = 4)
plot(layout_tsne, pch = 21, bg = iris$Species)

layout_info <- embed_knn(knn, method = "infotsne", max_iter = 250,
  n_negatives = 30, seed = 4)
plot(layout_info, pch = 21, bg = iris$Species)
}
