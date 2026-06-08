\name{benchmark_knn_umap}
\alias{benchmark_knn_umap}
\title{Benchmark UMAP implementations from the same KNN output}
\usage{
benchmark_knn_umap(data, labels = NULL, k = 30L, nn = NULL,
  implementations = c("fastknnumap_hybrid", "fastknnumap_landmark",
  "fastknnumap_sgd", "fastknnumap_spectral", "umap", "rtsne", "uwot",
  "uwot_fast_sgd", "knn_tsne", "knn_pacmap", "knn_trimap", "knn_localmap"),
  n_epochs = 200L, hybrid_epochs = 100L, min_dist = 0.01,
  negative_sample_rate = 10L, learning_rate = 1.5,
  spectral_n_iter = 25L, landmark_ratio = 0.1, n_landmarks = NULL,
  landmark_k = 10L, landmark_local_k = 5L, n_threads = 1L,
  silhouette_sample = 5000L,
  preserve_sample = 5000L, preserve_k = NULL, seed = 4L,
  verbose = TRUE)
}
\arguments{
\item{data}{Numeric matrix used to compute KNN once and to satisfy reference implementation APIs.}
\item{labels}{Optional labels for silhouette scoring.}
\item{k}{Number of neighbors requested from \code{nn}, including self.}
\item{nn}{Optional precomputed KNN list with \code{indices} and \code{distances}, including self in the first column.}
\item{implementations}{Character vector of implementations to run.}
\item{n_epochs}{Number of optimization epochs for SGD-style methods.}
\item{hybrid_epochs}{Number of optimization epochs for \code{fastknnumap_hybrid}.}
\item{min_dist}{UMAP minimum distance.}
\item{negative_sample_rate}{Negative sample rate.}
\item{learning_rate}{Initial learning rate.}
\item{spectral_n_iter}{Number of spectral power iterations for fastknnumap.}
\item{landmark_ratio}{Landmark fraction for \code{fastknnumap_landmark}.}
\item{n_landmarks}{Optional explicit number of landmarks.}
\item{landmark_k}{Number of landmark neighbors retained per point.}
\item{landmark_local_k}{Number of original local KNN edges retained for \code{fastknnumap_landmark}.}
\item{n_threads}{Number of CPU threads for KNN objective variants.}
\item{silhouette_sample}{Optional sample size for silhouette scoring.}
\item{preserve_sample}{Optional sample size for KNN preservation scoring.}
\item{preserve_k}{Number of neighbors used for preservation scoring.}
\item{seed}{Random seed.}
\item{verbose}{Print progress.}
}
\value{A list with KNN timing, per-implementation metrics, layouts, labels, and KNN matrices.}
\description{
Computes or accepts one KNN output and feeds the same neighbor index and
distance matrices to several UMAP implementations. Accuracy is reported as
label silhouette when labels are supplied and as KNN preservation against the
input neighbor graph.
}
