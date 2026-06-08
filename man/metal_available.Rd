\name{metal_available}
\alias{metal_available}
\title{Check whether the native Metal backend is available}
\usage{
metal_available()
}
\value{\code{TRUE} when a Metal device is available to the package.}
\description{
Checks whether the package can see a native macOS Metal device. The current
Metal backend accelerates exact Euclidean nearest-neighbor search through
\code{nn(..., backend = "metal")} or \code{nn(..., backend = "gpu")}.
}
