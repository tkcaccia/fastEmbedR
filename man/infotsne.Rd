\name{infotsne}
\alias{infotsne}
\title{Run InfoTSNE from a data matrix}
\usage{
infotsne(data, labels = NULL, n_neighbors = NULL, perplexity = NULL,
  n_components = 2L, standardize = TRUE, pca_dims = NULL, nn = NULL,
  seed = 4L, backend = c("auto", "cpu", "gpu", "metal", "cuda",
  "cuvs", "gpu_cuvs", "cuda_cuvs", "cuda_cuvs_cagra",
  "cuda_cuvs_bruteforce", "cuda_cuvs_exact", "cuda_cuvs_nndescent",
  "cuvs_bruteforce", "cuvs_nndescent"),
  silhouette_sample = NULL, preserve_sample = NULL, preserve_k = NULL,
  keep_knn = FALSE, verbose = FALSE, n_negatives = 300L,
  learning_rate = NULL, early_exaggeration_iter = 250L,
  early_exaggeration_coeff = 12, repulsion_strength = 1, ...)
}
\arguments{
\item{data}{Numeric matrix or data frame with observations in rows.}
\item{labels}{Optional labels used only for scoring and plotting metadata.}
\item{n_neighbors}{Number of non-self neighbors. If \code{NULL}, uses
\code{3 * perplexity}.}
\item{perplexity}{InfoTSNE perplexity. If \code{NULL}, uses a bounded default
based on sample size and KNN width.}
\item{n_components}{Output dimensionality from 1 to 3.}
\item{standardize}{Center and scale columns before KNN.}
\item{pca_dims}{Optional PCA dimension before KNN.}
\item{nn}{Optional precomputed KNN output.}
\item{seed}{Random seed.}
\item{backend}{KNN backend. The current InfoTSNE optimizer is CPU-only.
GPU and RAPIDS cuVS values are used only for the neighbour search and are
recorded separately as \code{nn_backend}; unavailable explicit GPU/cuVS
requests fail rather than falling back silently.}
\item{silhouette_sample}{Optional sample size for silhouette scoring.}
\item{preserve_sample}{Optional sample size for neighborhood scoring.}
\item{preserve_k}{Number of neighbors used for neighborhood scoring.}
\item{keep_knn}{If \code{TRUE}, retain KNN matrices in the returned object.}
\item{verbose}{Print optimizer progress.}
\item{n_negatives}{Number of sampled negatives per point and iteration.}
\item{learning_rate}{Learning rate. Use \code{NULL} for
\code{max(n / early_exaggeration_coeff / 4, 50)}.}
\item{early_exaggeration_iter}{Number of early exaggeration iterations.}
\item{early_exaggeration_coeff}{Early exaggeration multiplier.}
\item{repulsion_strength}{Multiplier for the sampled repulsive term.}
\item{...}{Additional optimizer parameters such as \code{max_iter},
\code{Y_init}, \code{momentum}, \code{final_momentum}, and \code{n_threads}.}
}
\value{
A \code{fastEmbedR_embedding} object.
}
\description{
Run InfoTSNE through the package's KNN-first workflow. The objective is inspired
by TorchDR's InfoTSNE: sparse KNN attraction plus sampled negative repulsion.
The optimizer is native C++ and does not call Python or PyTorch.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
fit <- infotsne(x, labels = iris$Species, perplexity = 10, max_iter = 250,
  n_negatives = 30, silhouette_sample = NULL, preserve_sample = NULL)
plot(fit)
}
