\name{benchmark_embed}
\alias{benchmark_embed}
\title{Benchmark embeddings with a minimal user interface}
\usage{
benchmark_embed(datasets = c("iris", "pendigits", "fashion_mnist"),
  n = 2000L, methods = c("fast", "tsne"),
  output_csv = NULL, preset = c("balanced", "quick", "accuracy"))
}
\arguments{
\item{datasets}{Public datasets to run. Defaults to a small practical suite.}
\item{n}{Row subset for datasets larger than Iris. Use \code{NULL} for each dataset's default.}
\item{methods}{Methods to compare. Use simple names such as \code{"fast"}, \code{"tsne"}, \code{"pacmap"}, \code{"trimap"}, \code{"localmap"}, or \code{"all"}.}
\item{output_csv}{Optional path to save the combined metrics table.}
\item{preset}{Speed/accuracy preset. \code{"quick"} runs fewer epochs, \code{"balanced"} is the default, and \code{"accuracy"} runs more epochs and repeats for stability.}
}
\value{A list with combined \code{metrics} and per-dataset \code{results}.}
\description{
A compact front door for final users. It runs public dataset benchmarks with
sensible defaults and compares only native fastEmbedR embeddings.
}
