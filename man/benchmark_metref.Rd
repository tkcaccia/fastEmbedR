\name{benchmark_metref}
\alias{benchmark_metref}
\title{Benchmark fastknnumap on the metref example}
\usage{
benchmark_metref(data_path = "/Users/stefano/Documents/GPUPLS/Data/metref_remote_task.RData",
  k = 30L, n_epochs = 200L, min_dist = 0.01,
  negative_sample_rate = 10L, learning_rate = 1.5, seed = 4L)
}
\arguments{
\item{data_path}{Path to \code{metref_remote_task.RData}.}
\item{k}{Number of nearest neighbors requested, including self.}
\item{n_epochs}{Number of native UMAP optimization epochs.}
\item{min_dist}{Native UMAP minimum distance.}
\item{negative_sample_rate}{Native UMAP negative sample rate.}
\item{learning_rate}{Native UMAP learning rate.}
\item{seed}{Native UMAP random seed.}
}
\value{A list with timings, silhouette scores, labels, and layouts.}
\description{
Loads the metref example data, computes nearest neighbors with
\code{nn}, embeds with \code{fast_knn_umap}, and compares speed and
label-based silhouette against \code{umap}'s precomputed-KNN path when
available.
}
