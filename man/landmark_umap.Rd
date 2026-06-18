\name{landmark_umap}
\alias{landmark_umap}
\title{Run Landmark UMAP}
\usage{
landmark_umap(
  data,
  landmarks = 0.5,
  n_neighbors = NULL,
  n_components = 2L,
  standardize = TRUE,
  pca_dims = NULL,
  seed = 4L,
  backend = c("cpu", "cuda", "metal"),
  transform_k = NULL,
  n_threads = NULL,
  keep_knn = FALSE,
  verbose = FALSE
)
}
\description{
Embed a landmark subset with \code{\link{umap}} and project the remaining
observations by KNN interpolation against the fixed landmark embedding.
}
\arguments{
\item{data}{Numeric matrix or data frame with observations in rows.}
\item{landmarks}{\code{TRUE} for an automatic subset, a fraction such as
\code{0.5}, a landmark count, or explicit row indices.}
\item{n_neighbors}{Number of non-self neighbours.}
\item{n_components}{Output dimensionality.}
\item{standardize}{Center and scale columns before KNN.}
\item{pca_dims}{Optional PCA dimension before KNN.}
\item{seed}{Random seed.}
\item{backend}{Execution backend: \code{"cpu"}, \code{"cuda"}, or
\code{"metal"}. KNN is selected internally: CPU and Metal use FAISS CPU
IVF-Flat through \code{faissR}, while CUDA uses FAISS GPU IVF-Flat. GPU
requests must use a real native backend and are not silently relabelled as
CPU.}
\item{transform_k}{Number of landmark neighbours used to project
non-landmark observations. Defaults to \code{n_neighbors}.}
\item{n_threads}{Number of CPU worker threads.}
\item{keep_knn}{Keep projection KNN matrices in the returned object.}
\item{verbose}{Print progress.}
}
\value{
A \code{fastEmbedR_embedding} object with the full layout, landmark metadata,
timings, and parameters.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
fit <- landmark_umap(x, landmarks = 0.5, n_neighbors = 15, seed = 1)
plot(fit)
}
