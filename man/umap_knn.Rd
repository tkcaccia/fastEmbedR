\name{umap_knn}
\alias{umap_knn}
\title{Run UMAP from precomputed KNN}
\usage{
umap_knn(
  indices,
  distances = NULL,
  n_components = 2L,
  seed = 42L,
  verbose = FALSE,
  backend = c("cpu", "cuda", "metal"),
  n_threads = NULL,
  graph_mode = c("binary", "fuzzy")
)
}
\arguments{
\item{indices}{A KNN object returned by \code{\link{nn}}, or an integer KNN
index matrix. A self-neighbour first column is removed automatically.}

\item{distances}{Numeric KNN distance matrix matching \code{indices}. Leave
\code{NULL} when \code{indices} is an \code{nn()} result.}

\item{n_components}{Output dimensionality.}

\item{seed}{Random seed.}

\item{verbose}{Print native optimizer progress.}

\item{backend}{Execution backend: \code{"cpu"}, \code{"cuda"}, or
\code{"metal"}. GPU requests must resolve to a real native backend; the
package does not label CPU work as GPU.}

\item{n_threads}{Number of CPU worker threads for CPU UMAP.}

\item{graph_mode}{Graph weighting mode. \code{"binary"} uses a symmetric
unit-weight graph. \code{"fuzzy"} uses standard UMAP fuzzy graph weights.}

}
\value{
A numeric embedding matrix with resolved settings in
\code{attr(layout, "fastEmbedR_config")}.
}
\description{
\code{umap_knn()} embeds observations from a supplied nearest-neighbour graph
using the stable native UMAP path. It is the KNN-input UMAP entry
point used by \code{\link{embed_knn}} with \code{method = "umap"}.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
knn <- faissR::nn(x, k = 16)
layout <- umap_knn(knn, seed = 1)
plot(layout, pch = 21, bg = iris$Species)
}
