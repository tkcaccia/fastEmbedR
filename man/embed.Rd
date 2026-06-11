\name{embed}
\alias{embed}
\alias{umap}
\title{Run UMAP from a data matrix}
\usage{
umap(data, labels = NULL, ...)
}
\arguments{
\item{data}{Numeric matrix or data frame with observations in rows.}
\item{labels}{Optional labels used for scoring and plotting metadata.}
\item{...}{Arguments passed to the internal UMAP workflow, such as
\code{n_neighbors}, \code{n_components}, \code{standardize}, \code{pca_dims},
\code{nn}, \code{landmarks}, \code{seed}, \code{backend},
\code{silhouette_sample}, \code{preserve_sample}, \code{preserve_k},
\code{keep_knn}, and \code{verbose}.}
}
\value{
A \code{fastEmbedR_embedding} object containing the layout, metrics, timings,
resolved parameters, and optionally the KNN matrices.
}
\description{
Run the package's focused UMAP workflow. The implementation is centered on
native KNN reuse and a CPU UMAP path configured to be comparable with
\code{uwot::umap(..., fast_sgd = TRUE)}.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
fit <- umap(x, labels = iris$Species, n_neighbors = 15,
  silhouette_sample = NULL, preserve_sample = NULL)
plot(fit)
}
