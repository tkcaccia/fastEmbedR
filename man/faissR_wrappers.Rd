\name{faissR_wrappers}
\alias{faissR_wrappers}
\alias{nn}
\alias{nn_without_self}
\alias{candidate_knn}
\alias{knn_graph}
\alias{fast_kmeans}
\alias{knn_fit}
\alias{predict_proba}
\alias{knn_recall}
\alias{faiss_available}
\alias{cuvs_available}
\title{Nearest-neighbour and graph utilities from faissR}
\usage{
nn(...)

nn_without_self(...)

candidate_knn(...)

knn_graph(...)

fast_kmeans(...)

knn_fit(...)

predict_proba(...)

knn_recall(...)

faiss_available()

cuvs_available()
}
\arguments{
\item{...}{Arguments passed directly to the matching \pkg{faissR} function.}
}
\value{
The value returned by the matching \pkg{faissR} function.
}
\description{
These functions are thin wrappers around the companion \pkg{faissR} package.
They are exported from \pkg{fastEmbedR} so a user can write workflows such as
\code{fastEmbedR::nn()} followed by \code{fastEmbedR::opentsne_knn()} or
\code{fastEmbedR::umap_knn()} without switching namespaces.
}
\details{
The implementation, optional FAISS/cuVS backends, CUDA support, and detailed
backend routing live in \pkg{faissR}. \pkg{fastEmbedR} only forwards the call and
keeps the embedding algorithms in this package.
}
\examples{
x <- as.matrix(iris[, 1:4])
knn <- fastEmbedR::nn(x, k = 16, backend = "cpu")
layout <- fastEmbedR::umap_knn(knn, seed = 1)
dim(layout)
}
