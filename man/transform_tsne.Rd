\name{transform_tsne}
\alias{transform_tsne}
\title{Transform query points into an existing t-SNE embedding}
\usage{
transform_tsne(reference_layout, knn = NULL, reference_data = NULL,
  new_data = NULL, k = NULL, perplexity = 5,
  initialization = c("median", "weighted", "random"), Y_init = NULL,
  n_iter = 250L, early_exaggeration_iter = 0L, learning_rate = 0.1,
  early_exaggeration = 4, exaggeration = 1.5,
  initial_momentum = 0.8, final_momentum = 0.8,
  max_grad_norm = 0.25, max_step_norm = Inf, n_negatives = NULL,
  exact_repulsion_threshold = 4096L, n_threads = NULL, seed = 4L,
  backend = "auto", verbose = FALSE)
}
\arguments{
\item{reference_layout}{Numeric reference embedding matrix, or a
\code{fastEmbedR_embedding} object.}
\item{knn}{Optional query-to-reference KNN list with \code{indices} and
\code{distances}, usually returned by \code{nn(reference_data, new_data, k)}.}
\item{reference_data}{Reference observations in the same preprocessing space
used to fit \code{reference_layout}. Required only when \code{knn} is
\code{NULL}.}
\item{new_data}{Query observations in the same preprocessing space as
\code{reference_data}. Required only when \code{knn} is \code{NULL}.}
\item{k}{Number of reference neighbors used for transform KNN. If \code{NULL},
uses at least \code{3 * perplexity}, matching the usual t-SNE affinity width.}
\item{perplexity}{Transform perplexity. The openTSNE default is 5.}
\item{initialization}{Initial query placement: \code{"median"} from reference
neighbors, inverse-distance \code{"weighted"}, or \code{"random"}.}
\item{Y_init}{Optional initial query embedding matrix. When supplied, it
overrides \code{initialization}.}
\item{n_iter}{Number of normal transform optimization iterations.}
\item{early_exaggeration_iter}{Number of early-exaggeration iterations.}
\item{learning_rate}{Transform learning rate.}
\item{early_exaggeration}{Early exaggeration multiplier.}
\item{exaggeration}{Normal transform exaggeration multiplier.}
\item{initial_momentum}{Momentum used during early exaggeration.}
\item{final_momentum}{Momentum used during normal optimization.}
\item{max_grad_norm}{Maximum per-query gradient norm. Use \code{Inf} to
disable.}
\item{max_step_norm}{Maximum per-query step norm. Use \code{Inf} to disable.}
\item{n_negatives}{Number of sampled reference points for repulsion when the
reference set is larger than \code{exact_repulsion_threshold}.}
\item{exact_repulsion_threshold}{Use exact query-reference repulsion at or
below this reference count.}
\item{n_threads}{Number of CPU worker threads for the native optimizer.}
\item{seed}{Random seed.}
\item{backend}{Backend used for query KNN when \code{knn} is \code{NULL}.
\code{"metal"} also runs the fixed-reference transform optimizer in native
Metal. CUDA transform is planned but intentionally errors until implemented,
so CPU work is never reported as CUDA.}
\item{verbose}{Print native optimizer progress.}
}
\value{
A numeric matrix with one row per query observation. Attributes record the
query KNN backend, transform optimizer, and resolved parameters.
}
\description{
\code{transform_tsne()} places query observations into a fixed reference t-SNE
embedding. It follows the openTSNE transform design at the algorithm level:
query-to-reference affinities are computed from KNN distances, query points
are initialized from nearby reference points, and optimization moves only the
query points.
}
\details{
This implementation is native C++ and does not call Python or openTSNE at
runtime. It is openTSNE-inspired rather than a vendored copy. With
\code{backend = "metal"}, the fixed-reference transform optimization loop runs
as a native Metal kernel. The current Metal path supports two-dimensional
reference layouts.
}
\examples{
set.seed(1)
x <- scale(as.matrix(iris[, 1:4]))
ref <- x[1:100, , drop = FALSE]
qry <- x[101:110, , drop = FALSE]

fit <- infotsne(ref, n_neighbors = 15, perplexity = 5, max_iter = 25,
  n_negatives = 10, standardize = FALSE, silhouette_sample = NULL,
  preserve_sample = NULL)
query_knn <- nn(ref, qry, k = 15, backend = "cpu")
projected <- transform_tsne(fit$layout, query_knn, perplexity = 5,
  n_iter = 10, n_negatives = 10)
head(projected)
}
