\name{load_embedding_dataset}
\alias{load_embedding_dataset}
\alias{load_mnist}
\alias{load_pendigits}
\alias{load_shuttle}
\title{Load public embedding benchmark datasets}
\usage{
load_embedding_dataset(name, n = NULL, pca_dims = NULL,
  cache_dir = file.path(tempdir(), "fastembedr-data"), seed = 4L)

load_mnist(dir = file.path(tempdir(), "mnist"), n = 10000L, seed = 4L)

load_pendigits(dir = file.path(tempdir(), "pendigits"), n = NULL, seed = 4L)

load_shuttle(dir = file.path(tempdir(), "shuttle"), n = 10000L, seed = 4L)
}
\arguments{
\item{name}{Dataset name. One of \code{"iris"}, \code{"mnist"}, \code{"fashion_mnist"}, \code{"pendigits"}, or \code{"shuttle"}.}
\item{n}{Optional row subset.}
\item{pca_dims}{Optional PCA dimensionality after loading.}
\item{cache_dir}{Directory for cached downloaded files.}
\item{seed}{Seed used for subsampling and PCA.}
\item{dir}{Directory for cached dataset files.}
}
\value{A list with matrix \code{x}, labels \code{y}, and source metadata.}
\description{
Loads public datasets that can be downloaded automatically without login or
private credentials. MNIST uses public IDX files, Fashion-MNIST uses the
Zalando public files via \code{load_fashion_mnist}, and PenDigits/Shuttle use
the UCI public repository.
}
