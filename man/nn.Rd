\name{nn}
\alias{nn}
\title{Exact nearest neighbors from row-wise matrices}
\usage{
nn(data, points = data, k = nrow(data), method = "euclidean",
  search = "standard", eps = 0, square = FALSE, sorted = FALSE,
  radius = 0, trans = TRUE, leafs = 10L, p = 0, parallel = FALSE,
  cores = 0L, backend = c("auto", "cpu", "gpu"))
}
\arguments{
\item{data}{Numeric matrix of reference observations in rows.}
\item{points}{Numeric matrix of query observations in rows. Defaults to \code{data}.}
\item{k}{Number of neighbors to return.}
\item{method}{Distance metric. Supported values are \code{"euclidean"}, \code{"manhattan"}, and \code{"minkowski"}.}
\item{search}{Accepted for compatibility. Only exact \code{"standard"} search is currently implemented.}
\item{eps}{Accepted for compatibility; approximate epsilon search is not yet implemented.}
\item{square}{If \code{TRUE}, return squared Euclidean distances.}
\item{sorted}{If \code{TRUE}, return neighbors sorted by distance. The exact backend currently returns sorted neighbors in all cases.}
\item{radius}{Accepted for compatibility; radius search is not yet implemented.}
\item{trans}{Accepted for compatibility. Observations are always interpreted as rows.}
\item{leafs}{Accepted for compatibility with tree-based APIs.}
\item{p}{Minkowski exponent when \code{method = "minkowski"}.}
\item{parallel}{Use multiple CPU threads over query points.}
\item{cores}{Number of CPU threads when \code{parallel = TRUE}. Values \code{<= 0} use hardware concurrency.}
\item{backend}{Execution backend. \code{"auto"} uses the Metal GPU for sufficiently large Euclidean searches when available and otherwise uses CPU. \code{"gpu"} requests the Metal backend explicitly. \code{"cpu"} always uses the C++ CPU path.}
}
\value{
A list with integer matrix \code{indices} and numeric matrix \code{distances}.
Indices are 1-based.
}
\description{
Package-native exact nearest-neighbor search compatible with the common
\code{Rnanoflann::nn(data, points, k)} use case. It is implemented in C++ and
can parallelize over query points. On macOS it can also use a native Metal GPU
backend for exact Euclidean search.
}
