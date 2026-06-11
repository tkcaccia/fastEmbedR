\name{opentsne_knn}
\alias{opentsne_knn}
\title{Run native openTSNE-style t-SNE from precomputed KNN}
\usage{
opentsne_knn(indices, distances = NULL, n_neighbors = NULL,
  perplexity = NULL, n_components = 2L, seed = 4L, verbose = FALSE,
  backend = c("auto", "cpu", "gpu", "metal", "cuda"),
  learning_rate = "auto", early_exaggeration_iter = 250L,
  early_exaggeration = "auto", n_iter = 500L, exaggeration = NULL,
  initial_momentum = 0.8, final_momentum = 0.8, max_step_norm = 5,
  negative_gradient_method = "auto", record_costs = FALSE, ...)
}
\arguments{
\item{indices}{A KNN object returned by \code{\link{nn}}, or an integer KNN
index matrix.}
\item{distances}{Numeric KNN distance matrix matching \code{indices}. Leave
\code{NULL} when \code{indices} is an \code{\link{nn}} result.}
\item{n_neighbors}{Optional number of non-self neighbor columns to use from
the supplied KNN graph.}
\item{perplexity}{t-SNE perplexity. If \code{NULL}, uses a bounded default
based on sample size and KNN width.}
\item{n_components}{Output dimensionality from 1 to 3.}
\item{seed}{Random seed.}
\item{verbose}{Print optimizer progress.}
\item{backend}{Execution backend. Native openTSNE-style full t-SNE from KNN is
currently CPU-only; explicit GPU requests fail rather than falling back
silently.}
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
A numeric embedding matrix with resolved settings stored in
\code{attr(layout, "fastEmbedR_config")}.
}
\description{
\code{opentsne_knn()} is the direct KNN-input entry point for the native
openTSNE-style optimizer. It accepts either an object returned by \code{\link{nn}}
or separate KNN index and distance matrices. No neighbour search, scaling, or
PCA is done inside this function.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
knn <- nn(x, k = 31)
layout <- opentsne_knn(knn, perplexity = 10,
  early_exaggeration_iter = 100, n_iter = 250)
plot(layout, pch = 21, bg = iris$Species)
}
