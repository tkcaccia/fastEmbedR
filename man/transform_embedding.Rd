\name{transform_embedding}
\alias{transform_embedding}
\title{Project new points into an existing embedding}
\usage{
transform_embedding(reference_layout, knn = NULL, reference_data = NULL,
  new_data = NULL, k = NULL,
  backend = c("auto", "cpu", "gpu", "cuda", "metal"))
}
\arguments{
\item{reference_layout}{Numeric embedding matrix for the reference observations.
Rows are reference observations and columns are embedding dimensions.}
\item{knn}{Optional KNN list with \code{indices} and \code{distances}, usually
returned by \code{nn(reference_data, new_data, k)}. Indices must be 1-based row
numbers into \code{reference_layout}.}
\item{reference_data}{Numeric matrix/data frame of reference observations.
Required only when \code{knn} is \code{NULL}.}
\item{new_data}{Numeric matrix/data frame of query observations. Required only
when \code{knn} is \code{NULL}.}
\item{k}{Number of reference neighbors to use. With a supplied \code{knn}, this
keeps the first \code{k} columns. With \code{knn = NULL}, this is passed to
\code{nn()}. \code{NULL} chooses the package's automatic neighborhood size.}
\item{backend}{KNN backend used only when \code{knn} is \code{NULL}. Explicit GPU
requests fail clearly when unavailable; they are not silently relabelled as CPU
work.}
}
\value{
A numeric matrix with one row per query observation and the same number of
columns as \code{reference_layout}. Attributes record the KNN backend, neighbor
count, and transform type.
}
\description{
\code{transform_embedding()} places query observations into an existing layout
from their nearest reference observations. It is inspired by UMAP's transform
workflow, but keeps the API KNN-first: pass a query KNN object when you already
have one, or pass reference/query matrices and let \code{fastEmbedR::nn()}
compute it.
}
\details{
The reference and query matrices must be in the same preprocessing space used
to fit the reference layout. For example, if the layout was fit on
\code{scale(data)}, pass scaled reference and query matrices here too.
}
\examples{
x <- scale(as.matrix(iris[, 1:4]))
labels <- iris$Species

fit <- umap(x, labels = labels, n_neighbors = 10, standardize = FALSE,
  silhouette_sample = NULL, preserve_sample = NULL)

# KNN-first workflow: compute query neighbors once, then project.
query_knn <- nn(x, x[1:5, , drop = FALSE], k = 10, backend = "cpu")
projected <- transform_embedding(fit$layout, query_knn)
head(projected)

# Or let transform_embedding compute the query KNN.
projected2 <- transform_embedding(fit$layout,
  reference_data = x,
  new_data = x[1:5, , drop = FALSE],
  k = 10,
  backend = "cpu"
)
head(projected2)
}
