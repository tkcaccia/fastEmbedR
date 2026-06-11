\name{tsne}
\alias{tsne}
\title{Run t-SNE from a data matrix}
\usage{
tsne(data, labels = NULL, n_neighbors = NULL, perplexity = NULL,
  n_components = 2L, standardize = TRUE, pca_dims = NULL, nn = NULL,
  seed = 4L, backend = c("auto", "cpu", "gpu", "metal", "cuda",
  "cuvs", "gpu_cuvs", "cuda_cuvs", "cuda_cuvs_cagra",
  "cuda_cuvs_bruteforce", "cuda_cuvs_exact", "cuda_cuvs_nndescent",
  "cuvs_bruteforce", "cuvs_nndescent"),
  silhouette_sample = NULL, preserve_sample = NULL, preserve_k = NULL,
  keep_knn = FALSE, verbose = FALSE, ...)
}
\arguments{
\item{data}{Numeric matrix or data frame with observations in rows.}
\item{labels}{Optional labels used only for scoring and plotting metadata.}
\item{n_neighbors}{Number of non-self neighbors. If \code{NULL}, uses
\code{3 * perplexity}.}
\item{perplexity}{t-SNE perplexity. If \code{NULL}, uses a bounded default
based on sample size and KNN width.}
\item{n_components}{Output dimensionality from 1 to 3.}
\item{standardize}{Center and scale columns before KNN.}
\item{pca_dims}{Optional PCA dimension before KNN.}
\item{nn}{Optional precomputed KNN output.}
\item{seed}{Random seed.}
\item{backend}{KNN backend. The current t-SNE optimizer is CPU-only.
GPU and RAPIDS cuVS values are used only for the neighbour search and are
recorded separately as \code{nn_backend}; unavailable explicit GPU/cuVS
requests fail rather than falling back silently.}
\item{silhouette_sample}{Optional sample size for silhouette scoring.}
\item{preserve_sample}{Optional sample size for neighborhood scoring.}
\item{preserve_k}{Number of neighbors used for neighborhood scoring.}
\item{keep_knn}{If \code{TRUE}, retain KNN matrices in the returned object.}
\item{verbose}{Print optimizer progress.}
\item{...}{Additional t-SNE optimizer parameters: \code{theta},
\code{max_iter}, \code{Y_init}, \code{stop_lying_iter},
\code{mom_switch_iter}, \code{momentum}, \code{final_momentum}, \code{eta},
\code{exaggeration_factor}, and \code{n_threads}.}
}
\value{
A \code{fastEmbedR_embedding} object.
}
\description{
Run t-SNE through the package's KNN-first workflow. The KNN input path follows
the user-facing behaviour of \code{Rtsne::Rtsne_neighbors()} while using a
native fastEmbedR optimizer. An internal KeOps-inspired exact blocked
reduction is available for development/GPU translation experiments, but the
faster pair-symmetric CPU loop remains the default.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
fit <- tsne(x, labels = iris$Species, perplexity = 10, max_iter = 250,
  silhouette_sample = NULL, preserve_sample = NULL)
plot(fit)
}
