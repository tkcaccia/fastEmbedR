\name{load_fashion_mnist}
\alias{load_fashion_mnist}
\alias{benchmark_fashion_mnist}
\title{Fashion-MNIST benchmark helpers}
\usage{
load_fashion_mnist(dir = file.path(tempdir(), "fashion-mnist"),
  n_train = 10000L, seed = 4L)

benchmark_fashion_mnist(n_train = 10000L, pca_dims = 50L,
  cache_dir = file.path(tempdir(), "fashion-mnist"), ...)
}
\arguments{
\item{dir}{Directory for cached IDX gzip files.}
\item{n_train}{Number of training images to load.}
\item{seed}{Seed used when subsampling.}
\item{pca_dims}{Optional PCA dimensionality before KNN.}
\item{cache_dir}{Directory for cached Fashion-MNIST files.}
\item{...}{Additional arguments passed to \code{benchmark_knn_umap}.}
}
\value{\code{load_fashion_mnist} returns a list with \code{x} and \code{y}. \code{benchmark_fashion_mnist} returns a benchmark result.}
\description{
Downloads Fashion-MNIST from the public Zalando repository, optionally applies
PCA, computes KNN once, and benchmarks KNN-based embeddings.
}
