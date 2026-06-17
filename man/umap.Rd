\name{umap}
\alias{umap}
\title{Run UMAP from data or precomputed KNN}
\usage{
umap(
  data,
  labels = NULL,
  n_neighbors = NULL,
  n_components = 2L,
  standardize = TRUE,
  pca_dims = NULL,
  nn = NULL,
  seed = 4L,
  backend = c("auto", "cpu", "gpu", "metal", "cuda"),
  n_threads = NULL,
  silhouette_sample = NULL,
  preserve_sample = NULL,
  preserve_k = NULL,
  keep_knn = FALSE,
  graph_mode = c("binary", "fuzzy"),
  verbose = FALSE
)
}
\arguments{
\item{data}{Numeric matrix/data frame with observations in rows, or a KNN
object returned by \code{\link{nn}}.}

\item{labels}{Optional labels stored in the returned object and used for
optional scoring.}

\item{n_neighbors}{Number of non-self neighbours. \code{NULL} chooses the
package default for the data size.}

\item{n_components}{Output dimensionality.}

\item{standardize}{Center and scale columns before KNN when \code{data} is a
matrix.}

\item{pca_dims}{Optional PCA dimension before KNN.}

\item{nn}{Optional precomputed KNN result when \code{data} is a matrix.}

\item{seed}{Random seed.}

\item{backend}{Execution backend. GPU requests must resolve to a real native
backend; the package does not label CPU work as GPU.}

\item{n_threads}{Number of CPU worker threads for KNN and CPU UMAP, capped
internally at four threads for the restored stable UMAP path.}

\item{silhouette_sample}{Optional sample size for silhouette scoring.}

\item{preserve_sample}{Optional sample size for local-structure scoring.}

\item{preserve_k}{Number of neighbours used for local-structure scoring.}

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
\code{\link{umap_knn}}. For strict benchmarking, compute KNN once with
\code{\link{nn}} and pass it to \code{umap_knn()}.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
fit <- umap(x, labels = iris$Species, n_neighbors = 15, seed = 1)
plot(fit)
}
