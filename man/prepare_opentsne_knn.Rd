\name{prepare_opentsne_knn}
\alias{prepare_opentsne_knn}
\title{Precompute reusable openTSNE KNN state}
\usage{
prepare_opentsne_knn(
  indices,
  distances = NULL,
  n_neighbors = NULL,
  perplexity = NULL
)
}
\arguments{
\item{indices}{A KNN object returned by \code{faissR::nn()}, or an integer KNN index matrix.}
\item{distances}{Numeric KNN distance matrix matching \code{indices}. Leave \code{NULL} when \code{indices} is a KNN object.}
\item{n_neighbors}{Optional number of non-self neighbour columns to retain.}
\item{perplexity}{Optional t-SNE perplexity used to choose the retained KNN width when \code{n_neighbors} is not supplied.}
}
\value{
A prepared openTSNE KNN object. Pass it to \code{opentsne_knn()} for repeated runs without repeating KNN normalization/materialization.
}
\description{
\code{prepare_opentsne_knn()} strips self-neighbours, trims to a compact non-self KNN width, and stores the materialized KNN matrices once. The current native optimizer still builds t-SNE affinities internally from this compact KNN state.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
knn <- faissR::nn(x, k = 15, exclude_self = TRUE)
prep <- prepare_opentsne_knn(knn, perplexity = 10)
y <- opentsne_knn(prep, early_exaggeration_iter = 50, n_iter = 100)
plot(y, pch = 21, bg = iris$Species)
}
