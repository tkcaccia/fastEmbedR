\name{landmark_tsne}
\alias{landmark_tsne}
\title{Landmark t-SNE with openTSNE-style transform}
\usage{
landmark_tsne(data, labels = NULL, landmarks = TRUE,
  reference_method = c("infotsne", "tsne"), n_neighbors = NULL,
  perplexity = NULL, n_components = 2L, standardize = TRUE,
  pca_dims = NULL, seed = 4L, backend = "auto", transform_k = NULL,
  transform_perplexity = 5, transform_iter = 250L,
  transform_early_exaggeration_iter = 0L,
  transform_n_negatives = NULL,
  initialization = c("median", "weighted", "random"),
  silhouette_sample = NULL, preserve_sample = NULL, preserve_k = NULL,
  keep_knn = FALSE, verbose = FALSE, ...)
}
\arguments{
\item{data}{Numeric matrix/data frame with observations in rows.}
\item{labels}{Optional labels used only for metadata and optional scoring.}
\item{landmarks}{\code{TRUE} for an automatic subset, a fraction such as
\code{0.5}, a landmark count, or explicit row indices.}
\item{reference_method}{\code{"infotsne"} for the scalable reference optimizer
or \code{"tsne"} for the Rtsne-neighbors-style exact reference optimizer.}
\item{n_neighbors}{Number of non-self neighbors. If \code{NULL}, the package
uses \code{3 * perplexity}, matching the usual \code{Rtsne_neighbors()}
convention.}
\item{perplexity}{t-SNE perplexity. If \code{NULL}, uses
\code{min(30, floor((n - 1) / 3), floor(k / 3))}.}
\item{n_components}{Output dimensionality, from 1 to 3.}
\item{standardize}{Center and scale columns before KNN.}
\item{pca_dims}{Optional PCA dimension before KNN.}
\item{seed}{Random seed.}
\item{backend}{KNN backend. \code{"metal"} also runs the non-landmark transform
optimizer in native Metal. CUDA transform is planned but unavailable and is not
silently replaced by CPU.}
\item{transform_k}{Number of landmark neighbors used to place non-landmarks.}
\item{transform_perplexity}{Perplexity used by \code{transform_tsne()}.}
\item{transform_iter}{Number of normal transform iterations.}
\item{transform_early_exaggeration_iter}{Number of transform early
exaggeration iterations.}
\item{transform_n_negatives}{Number of sampled reference negatives used by
\code{transform_tsne()} on large landmark sets.}
\item{initialization}{Initial placement for transformed observations.}
\item{silhouette_sample}{Optional sample size for silhouette scoring.}
\item{preserve_sample}{Optional sample size for neighborhood scoring.}
\item{preserve_k}{Number of neighbors used for neighborhood scoring.}
\item{keep_knn}{If \code{TRUE}, retain projection KNN matrices in the returned
object.}
\item{verbose}{Print optimizer progress.}
\item{...}{Additional parameters passed to the reference \code{tsne()} or
\code{infotsne()} call, such as \code{max_iter}, \code{n_negatives},
\code{learning_rate}, or \code{n_threads}.}
}
\value{
A \code{fastEmbedR_embedding} object containing the full layout, timings,
resolved parameters, and landmark metadata.
}
\description{
\code{landmark_tsne()} embeds a subset of observations with \code{tsne()} or
\code{infotsne()}, then places the remaining observations with
\code{transform_tsne()}. This mirrors the practical landmark workflow exposed
by openTSNE while keeping the implementation native to this package.
}
\details{
The reference embedding and transform optimizer are native fastEmbedR code.
The design is informed by openTSNE's fixed-reference transform: non-landmark
points are optimized independently against the fixed landmark embedding and do
not interact with one another during transform. With \code{backend = "metal"},
the transform optimizer runs in native Metal for two-dimensional reference
layouts.
}
\examples{
set.seed(2)
x <- scale(as.matrix(iris[, 1:4]))
fit <- landmark_tsne(x, labels = iris$Species, landmarks = 0.5,
  n_neighbors = 15, perplexity = 5, max_iter = 25, n_negatives = 10,
  transform_iter = 10, transform_n_negatives = 10,
  standardize = FALSE, silhouette_sample = NULL, preserve_sample = NULL)
plot(fit)
}
