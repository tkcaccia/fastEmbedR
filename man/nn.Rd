\name{nn}
\alias{nn}
\alias{print.fastEmbedR_nn}
\title{Nearest neighbors from row-wise matrices}
\usage{
nn(data, points = data, k = NULL,
  backend = c("auto", "cpu", "cpu_approx", "cpu_ivf", "cpu_faiss_ivf",
  "faiss", "cpu_faiss", "cpu_faiss_flat", "faiss_ivf",
  "cpu_faiss_index_ivf", "cuvs", "gpu_cuvs", "cuda_cuvs",
  "cuda_cuvs_cagra", "cuvs_bruteforce", "cuda_cuvs_bruteforce",
  "cuda_cuvs_exact", "cuvs_nndescent", "cuda_cuvs_nndescent",
  "cpu_annoy", "cpu_nndescent", "cpu_clustered", "cpu_vptree", "gpu",
  "cuda", "metal", "gpu_approx", "cuda_approx", "metal_approx",
  "gpu_nndescent", "cuda_nndescent", "metal_nndescent", "gpu_ivf",
  "cuda_ivf", "metal_ivf", "gpu_faiss", "cuda_faiss", "metal_faiss",
  "metal_grid"),
  n_threads = NULL)

\method{print}{fastEmbedR_nn}(x, ...)
}
\arguments{
\item{data}{Numeric matrix of reference observations in rows.}
\item{points}{Numeric matrix of query observations in rows. Defaults to \code{data}.}
\item{k}{Number of neighbors to return. \code{NULL} chooses the package's
automatic neighborhood size and includes the self-neighbor when \code{points}
is \code{data}.}
\item{backend}{Execution backend. \code{"auto"} records the selected fast backend in \code{attr(result, "backend")}. \code{"cpu"} always uses the exact C++ CPU path. \code{"cpu_approx"} chooses the current native CPU approximation. \code{"cpu_ivf"} and \code{"cpu_faiss_ivf"} use package-native IVF-flat search; the latter records a FAISS-style IVF-flat strategy but does not link to external FAISS. \code{"faiss"} uses the real FAISS C++ \code{IndexFlatL2} backend when fastEmbedR was built against FAISS, and \code{"faiss_ivf"} uses real FAISS \code{IndexIVFFlat}. \code{"cuda_cuvs"} uses a RAPIDS-inspired cuVS policy: exact cuVS brute force for small searches and cuVS NN-descent for large self-KNN. \code{"cuda_cuvs_cagra"} forces cuVS CAGRA. \code{"cuda_cuvs_bruteforce"} and \code{"cuda_cuvs_exact"} use exact cuVS brute-force search. \code{"cuda_cuvs_nndescent"} uses cuVS NN-descent for self-KNN. \code{"cpu_annoy"} uses a package-native Annoy-style random-projection forest. \code{"cpu_nndescent"}, \code{"cpu_clustered"}, and \code{"cpu_vptree"} force native NN-descent, legacy clustered-IVF, and VP-tree strategies for self-KNN searches. \code{"gpu"} requests CUDA when available and otherwise Metal. \code{"cuda"} and \code{"metal"} request exact GPU backends explicitly. \code{"gpu_nndescent"}, \code{"cuda_nndescent"}, and \code{"metal_nndescent"} request seeded native NN-descent refinement. \code{"gpu_approx"}, \code{"cuda_approx"}, and \code{"metal_approx"} are convenience aliases for the preferred native GPU NN-descent path. \code{"gpu_ivf"}, \code{"cuda_ivf"}, \code{"metal_ivf"}, \code{"gpu_faiss"}, \code{"cuda_faiss"}, and \code{"metal_faiss"} request native GPU IVF/FAISS-style IVF-flat searches and fail clearly if the requested GPU backend is unavailable. \code{"metal_grid"} requests the experimental FastGraph-style Metal grid-candidate search for low-dimensional or PCA-reduced self-KNN.}
\item{n_threads}{Number of CPU worker threads for CPU backends. GPU backends
ignore this argument.}
\item{x, ...}{A \code{fastEmbedR_nn} object and extra print arguments.}
}
\value{
A list with integer matrix \code{indices} and numeric matrix \code{distances}.
Indices are 1-based. The object has class \code{fastEmbedR_nn} and records the
backend used in \code{attr(out, "backend")} and whether the result is exact in
\code{attr(out, "exact")}.
}
\description{
Package-native nearest-neighbor search compatible with the common
\code{nn(data, points, k)} use case. Explicit \code{backend = "cpu"} uses exact
C++ brute-force search and is mainly a reference path for small data or recall
checks. \code{backend = "auto"} chooses a recorded fast path for large self-KNN
searches: native CUDA/Metal NN-descent when available, then a native CPU approximate
path. Use \code{backend = "cpu_approx"} to force the CPU approximate selector,
or \code{backend = "cpu_ivf"}, \code{"cpu_annoy"}, \code{"cpu_nndescent"},
\code{"cpu_clustered"}, or \code{"cpu_vptree"} to choose a specific native
multi-CPU strategy. \code{backend = "faiss"} and \code{"faiss_ivf"} call the
real FAISS C++ library when it was linked at package build time; otherwise they
fail clearly. \code{backend = "cuda_cuvs"} and related cuVS names call RAPIDS
cuVS when it was linked at package build time; otherwise they fail clearly.
Native
CUDA and Metal GPU backends are exact when requested explicitly; native
IVF/FAISS-style, grid-candidate, and seeded NN-descent GPU searches are approximate.
}
\examples{
x <- as.matrix(iris[, 1:4])

# Default self-query KNN: the first column is the self-neighbor.
knn <- nn(x, backend = "cpu")
knn
head(knn$indices)
head(knn$distances)

# Query a smaller set of points against the same reference matrix.
query <- x[1:5, , drop = FALSE]
query_knn <- nn(x, query, k = 3, backend = "cpu")
query_knn

# Native approximate self-KNN strategies for larger inputs.
ivf_knn <- nn(x, k = 15, backend = "cpu_ivf", n_threads = 4)
annoy_knn <- nn(x, k = 15, backend = "cpu_annoy", n_threads = 4)
vp_knn <- nn(x, k = 15, backend = "cpu_vptree")

# Real FAISS is optional and must be linked at package build time.
if (faiss_available()) {
  faiss_knn <- nn(x, k = 15, backend = "faiss")
}

# RAPIDS cuVS is optional and must be linked at package build time.
if (cuvs_available()) {
  cuvs_knn <- nn(x, k = 15, backend = "cuda_cuvs")
}

# Explicit GPU requests fail clearly when the backend is unavailable.
\dontrun{
knn_metal <- nn(x, backend = "metal")
knn_metal_nndescent <- nn(x, backend = "metal_nndescent")
knn_metal_ivf <- nn(x, backend = "metal_ivf")
knn_metal_grid <- nn(x, backend = "metal_grid")
knn_cuda <- nn(x, backend = "cuda")
knn_cuda_nndescent <- nn(x, backend = "cuda_nndescent")
}
}
