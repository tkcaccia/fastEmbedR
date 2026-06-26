\name{prepare_umap_knn}
\alias{prepare_umap_knn}
\title{Precompute reusable UMAP graph state from KNN}
\usage{
prepare_umap_knn(
  indices,
  distances = NULL,
  backend = c("cpu", "cuda", "metal"),
  n_threads = NULL,
  graph_mode = c("binary", "fuzzy")
)
}
\arguments{
\item{indices}{A KNN object returned by \code{faissR::nn()}, or an integer KNN index matrix.}
\item{distances}{Numeric KNN distance matrix matching \code{indices}. Leave \code{NULL} when \code{indices} is a KNN object.}
\item{backend}{Execution backend used to resolve size-aware defaults: \code{"cpu"}, \code{"cuda"}, or \code{"metal"}.}
\item{n_threads}{Number of CPU worker threads used during graph construction.}
\item{graph_mode}{Graph weighting mode. \code{"binary"} uses a symmetric unit-weight graph. \code{"fuzzy"} uses standard UMAP fuzzy graph weights.}
}
\value{
A prepared UMAP object containing normalized KNN data, a CSR graph, graph weights, and \code{epochs_per_sample}. Pass it to \code{umap_knn()} for repeated runs without rebuilding the graph.
}
\description{
\code{prepare_umap_knn()} builds the reusable UMAP graph once from a KNN object or index/distance matrices. This is useful for repeated seeds, benchmark grids, and CPU/GPU comparisons where the neighbour graph should remain fixed.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
knn <- faissR::nn(x, k = 15, exclude_self = TRUE)
prep <- prepare_umap_knn(knn)
y <- umap_knn(prep, seed = 1)
plot(y, pch = 21, bg = iris$Species)
}
