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
\code{nn(..., backend = "cuda")} or \code{nn(..., backend = "gpu")}.
}
