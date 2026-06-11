\name{cuda_available}
\alias{cuda_available}
\title{Check whether the native CUDA backend is available}
\usage{
cuda_available()
}
\value{\code{TRUE} when the package was built with CUDA support and the CUDA runtime reports at least one available device.}
\description{
Checks whether the package can use a native NVIDIA CUDA device. The CUDA
backend accelerates exact Euclidean nearest-neighbor search through
\code{nn(..., backend = "cuda")} or \code{nn(..., backend = "gpu")} and can
run two-dimensional KNN embedding optimizers through
\code{embed_knn(..., backend = "cuda")} when the package is built with CUDA
support.
}
\examples{
cuda_available()

\dontrun{
if (cuda_available()) {
  x <- as.matrix(iris[, 1:4])
  knn <- nn(x, backend = "cuda")
  layout <- embed_knn(knn, method = "umap", backend = "cuda")
}
}
}
