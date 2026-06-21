\name{umap}
\alias{umap}
\title{Run UMAP from data or precomputed KNN}
\usage{
umap(
  data,
  n_neighbors = NULL,
  n_components = 2L,
  standardize = FALSE,
  pca_dims = NULL,
  nn = NULL,
  seed = 4L,
  backend = c("cpu", "cuda", "metal"),
  n_threads = NULL,
  keep_knn = FALSE,
  graph_mode = c("binary", "fuzzy"),
  verbose = FALSE
)
}
\arguments{
\item{data}{Numeric matrix/data frame, or a KNN object returned by \code{\link[faissR:nn]{faissR::nn()}}.}

\item{n_neighbors}{Number of non-self neighbours. \code{NULL} chooses the
package default for the data size.}

\item{n_components}{Output dimensionality.}

\item{standardize}{Center and scale columns before KNN when \code{data} is a
matrix. Defaults to \code{FALSE} so one-call results match a KNN object computed
from the supplied matrix.}

\item{pca_dims}{Optional PCA dimension before KNN.}

\item{nn}{Optional precomputed KNN result when \code{data} is a matrix.}

\item{seed}{Random seed.}

\item{backend}{Execution backend: \code{"cpu"}, \code{"cuda"}, or \code{"metal"}. KNN is delegated to \code{faissR::nn_without_self()} with automatic method and tuning selection. CPU and Metal request the faissR CPU backend, while CUDA requests the faissR CUDA backend. GPU requests must resolve to a real native backend; the package does not relabel CPU work as GPU.}

\item{n_threads}{Number of CPU worker threads for KNN and CPU UMAP.}

\item{keep_knn}{If \code{TRUE}, retain KNN matrices in the returned object.}

\item{graph_mode}{Graph weighting mode. \code{"binary"} uses a symmetric
unit-weight graph. \code{"fuzzy"} uses standard UMAP fuzzy graph weights.}

\item{verbose}{Print progress.}
}
\value{
A \code{fastEmbedR_embedding} object.
}
\description{
\code{umap()} computes or reuses a nearest-neighbour graph and then calls
\code{\link{umap_knn}}. Quality metrics are computed explicitly with
\code{\link{evaluate_embedding}} rather than inside the embedding call.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
fit <- umap(x, n_neighbors = 15, seed = 1)
plot(fit)
}
