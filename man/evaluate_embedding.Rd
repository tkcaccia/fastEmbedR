\name{evaluate_embedding}
\alias{evaluate_embedding}
\title{Evaluate embedding quality}
\usage{
evaluate_embedding(x_high, embedding, labels = NULL, batch = NULL,
  k = c(15L, 30L, 50L),
  primary_k = NULL,
  reference_nn = NULL,
  sample_size_for_global_metrics = min(5000L, nrow(x_high)),
  sample_size_for_local_metrics = min(5000L, nrow(x_high)),
  use_cache = FALSE,
  cache_dir = file.path("results", "cache"),
  force_recompute = FALSE,
  seed = NA_integer_, method = NA_character_, backend = NA_character_,
  dataset = NA_character_)
}
\arguments{
\item{x_high}{High-dimensional numeric matrix, usually scaled data or PCA scores.}
\item{embedding}{Low-dimensional numeric embedding matrix.}
\item{labels}{Optional biological/class labels.}
\item{batch}{Optional batch labels.}
\item{k}{Neighbor sizes used for structure metrics.}
\item{primary_k}{Neighbor size used for scalar trustworthiness, continuity,
label accuracy, rare-class recall, density summaries, and rank-error metrics.
The default \code{NULL} preserves the historical k = 15 behavior when possible.}
\item{reference_nn}{Optional precomputed high-dimensional nearest-neighbor
list. It may include a self-neighbor column, like \code{nn(x, x, k + 1)}.}
\item{sample_size_for_global_metrics}{Maximum deterministic subsample size for
global distance metrics.}
\item{sample_size_for_local_metrics}{Maximum deterministic subsample size for
local trustworthiness, continuity, neighbour preservation, and density-radius
preservation metrics.}
\item{use_cache}{Cache the high-dimensional reference neighbours used by
quality metrics.}
\item{cache_dir}{Directory for quality-metric reference-neighbour cache files.}
\item{force_recompute}{Ignore cached quality reference neighbours.}
\item{seed}{Random seed recorded in the output and used for subsampling.}
\item{method}{Method name recorded in the output.}
\item{backend}{Backend name recorded in the output.}
\item{dataset}{Dataset name recorded in the output.}
}
\value{
A one-row data frame with local structure metrics, density-radius preservation
metrics, global distance metrics, label-aware metrics, batch-aware metrics, and
metadata columns. A per-class recall table is attached as the
\code{per_class_recall} attribute and encoded in \code{per_class_recall_json}.
}
\description{
Computes embedding quality without all-pairs work on large datasets. Global
distance metrics use a reproducible subsample of at most
\code{sample_size_for_global_metrics} observations. Local structure and
density-radius metrics reuse nearest-neighbour distances and can also be sampled
with \code{sample_size_for_local_metrics}, while the high-dimensional reference
neighbours can be cached and reused across seed repeats.
}
\examples{
x <- as.matrix(iris[, 1:4])
layout <- stats::cmdscale(stats::dist(scale(x)), k = 2)
evaluate_embedding(x, layout, iris$Species, k = c(10, 15), method = "cmdscale")
evaluate_embedding(x, layout, iris$Species, k = c(10, 15, 30), primary_k = 30)
}
