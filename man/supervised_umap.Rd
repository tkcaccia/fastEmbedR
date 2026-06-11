\name{supervised_umap}
\alias{supervised_umap}
\title{Supervised UMAP from labels}
\usage{
supervised_umap(data, labels, target_weight = 0.5,
  target_metric = c("categorical"), n_neighbors = NULL,
  n_components = 2L, standardize = TRUE, pca_dims = NULL,
  seed = 4L, backend = c("auto", "cpu", "gpu", "metal",
    "cuda"), silhouette_sample = NULL,
  preserve_sample = NULL, preserve_k = NULL, keep_knn = FALSE,
  verbose = FALSE)
}
\arguments{
\item{data}{Numeric matrix or data frame with observations in rows.}
\item{labels}{Labels used to adjust the UMAP KNN graph and to score/plot.}
\item{target_weight}{Strength of label supervision in \code{[0, 1]}.}
\item{target_metric}{Currently only \code{"categorical"} is implemented.}
\item{n_neighbors}{Number of non-self neighbors. \code{NULL} chooses the
package's automatic neighborhood size.}
\item{n_components}{Output dimensionality.}
\item{standardize}{If \code{TRUE}, center and scale columns before KNN.}
\item{pca_dims}{Optional PCA dimension before KNN.}
\item{seed}{Random seed.}
\item{backend}{KNN and embedding backend. Use \code{"gpu"} to explicitly
request a native GPU for both stages. \code{"auto"} may select the internal
clustered CPU KNN path for very large self-KNN searches and records that backend
in the result metadata.}
\item{silhouette_sample}{Optional sample size for silhouette scoring. Use \code{NULL} to skip silhouette scoring.}
\item{preserve_sample}{Optional sample size for neighborhood scoring. Use \code{NULL} to skip neighborhood scoring.}
\item{preserve_k}{Number of neighbors used for neighborhood scoring.}
\item{keep_knn}{If \code{TRUE}, retain the supervised non-self KNN matrices.}
\item{verbose}{Print progress.}
}
\value{A \code{fastEmbedR_embedding} object.}
\description{
Runs UMAP with an explicit supervised graph adjustment inspired by cuML's
\code{y}/\code{target_weight} supervised UMAP interface. The regular
\code{umap()} function still treats labels only as scoring/plotting metadata;
use \code{supervised_umap()} when labels should guide the embedding itself.
}
\details{
The implementation starts from the same KNN-input workflow as the rest of the
package. Same-label KNN edges are pulled closer, different-label KNN edges are
pushed farther, and the adjusted KNN graph is passed to the native UMAP
optimizer. This is a lightweight native approximation of supervised categorical
UMAP rather than a dependency on cuML.
}
\examples{
x <- as.matrix(iris[, 1:4])
labels <- iris$Species

fit <- supervised_umap(x, labels, n_neighbors = 10,
  silhouette_sample = NULL, preserve_sample = NULL)
fit
head(fit$layout)
fit$parameters[c("supervised", "target_weight", "target_metric")]
}
