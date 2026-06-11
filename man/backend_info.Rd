\name{backend_info}
\alias{backend_info}
\title{Summarize native backend availability}
\usage{
backend_info()
}
\value{
A data frame with one row per native backend and columns describing KNN
availability, embedding availability, device/runtime hints, and a short note.
The function never intentionally falls back from a requested GPU backend to CPU;
this table is informational only.
}
\description{
\code{backend_info()} reports which native package backends can currently run.
It is a lightweight availability summary for users who want to check FAISS,
RAPIDS cuVS, or GPU support before requesting \code{backend = "faiss"},
\code{"cuda_cuvs"}, \code{"cuda"}, \code{"metal"}, or \code{"gpu"}.
}
\details{
The function does not load or depend on external GPU packages such as
\pkg{cuda.ml}; it only reports the native backends compiled into
\pkg{fastEmbedR} and visible on the current machine. FAISS and RAPIDS cuVS are
reported only when the external libraries were linked at build time. CUDA
builds report CUDA runtime device details when available; non-CUDA builds
report the unavailable state and may include \code{nvidia-smi} device hints if
that command is present.
}
\examples{
backend_info()

\dontrun{
info <- backend_info()
info[info$available, c("backend", "knn_available", "embedding_available")]
}
}
