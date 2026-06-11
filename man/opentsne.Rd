\name{opentsne}
\alias{opentsne}
\title{Run native openTSNE-style t-SNE from data or KNN}
\usage{
opentsne(data, labels = NULL, n_neighbors = NULL, perplexity = NULL,
  n_components = 2L, standardize = TRUE, pca_dims = NULL, nn = NULL,
  seed = 4L, backend = c("auto", "cpu", "gpu", "metal", "cuda",
  "cuvs", "gpu_cuvs", "cuda_cuvs", "cuda_cuvs_cagra",
  "cuda_cuvs_bruteforce", "cuda_cuvs_exact", "cuda_cuvs_nndescent",
  "cuvs_bruteforce", "cuvs_nndescent"),
  silhouette_sample = NULL, preserve_sample = NULL, preserve_k = NULL,
  keep_knn = FALSE, verbose = FALSE, learning_rate = "auto",
  early_exaggeration_iter = 250L, early_exaggeration = "auto",
  n_iter = 500L, exaggeration = NULL, initial_momentum = 0.8,
  final_momentum = 0.8, max_step_norm = 5,
  negative_gradient_method = "auto", record_costs = FALSE, ...)
}
\arguments{
\item{data}{Numeric matrix/data frame with observations in rows, or a KNN
object returned by \code{\link{nn}}. For separate KNN matrices use
\code{\link{opentsne_knn}}.}
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
\item{backend}{KNN backend. The native openTSNE-style optimizer itself is
currently CPU-only; unavailable explicit GPU/cuVS neighbour requests fail
rather than falling back silently.}
\item{silhouette_sample}{Optional sample size for silhouette scoring.}
\item{preserve_sample}{Optional sample size for neighborhood scoring.}
\item{preserve_k}{Number of neighbors used for neighborhood scoring.}
\item{keep_knn}{If \code{TRUE}, retain KNN matrices in the returned object.}
\item{verbose}{Print optimizer progress.}
\item{learning_rate}{Positive number or \code{"auto"}. With \code{"auto"},
the native optimizer uses \code{n / exaggeration} separately for each phase.}
\item{early_exaggeration_iter}{Number of early-exaggeration iterations.}
\item{early_exaggeration}{Early-exaggeration multiplier, or \code{"auto"} for
openTSNE's default rule.}
\item{n_iter}{Number of normal optimization iterations after early
exaggeration.}
\item{exaggeration}{Normal-phase exaggeration. \code{NULL} means 1.}
\item{initial_momentum}{Momentum during early exaggeration.}
\item{final_momentum}{Momentum during normal optimization.}
\item{max_step_norm}{Maximum per-point update norm. Use \code{NULL} or
\code{NA} to disable clipping.}
\item{negative_gradient_method}{\code{"auto"}, \code{"bh"}, or
\code{"exact"}. Native FFT interpolation is not ported yet and fails clearly.}
\item{record_costs}{If \code{TRUE}, compute diagnostic KL/cost traces. This
does not affect the embedding, but costs extra time because it evaluates the
objective for reporting.}
\item{...}{Additional native optimizer parameters such as \code{theta},
\code{Y_init}, \code{min_gain}, and \code{n_threads}.}
}
\value{
A \code{fastEmbedR_embedding} object. When \code{data} is a KNN object,
preprocessing and KNN timings are recorded as zero because those steps were
already completed upstream.
}
\description{
\code{opentsne()} uses the same KNN-first workflow as \code{tsne()}, but its
optimizer follows openTSNE's two-phase contract: an early-exaggeration phase
followed by a normal optimization phase, \code{learning_rate = n /
exaggeration} when \code{learning_rate = "auto"}, openTSNE-style gains, and
max-step clipping. The implementation is native fastEmbedR C++ and does not
call Python. If the first argument is an \code{\link{nn}} result, the supplied
neighbour graph is used directly.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
fit <- opentsne(x, labels = iris$Species, perplexity = 10,
  early_exaggeration_iter = 100, n_iter = 250,
  silhouette_sample = NULL, preserve_sample = NULL)
plot(fit)

knn <- nn(x, k = 31)
fit_from_knn <- opentsne(knn, labels = iris$Species, perplexity = 10,
  early_exaggeration_iter = 100, n_iter = 250,
  silhouette_sample = NULL, preserve_sample = NULL)
plot(fit_from_knn)
}
