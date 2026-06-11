\name{cuvs_available}
\alias{cuvs_available}
\title{Check whether the RAPIDS cuVS backend is available}
\usage{
cuvs_available()
}
\value{
\code{TRUE} when fastEmbedR was compiled and linked against RAPIDS cuVS and a
CUDA device is visible to the CUDA runtime.
}
\description{
Reports whether explicit cuVS KNN requests such as
\code{nn(x, backend = "cuda_cuvs")}, \code{nn(x, backend =
"cuda_cuvs_bruteforce")}, and \code{nn(x, backend = "cuda_cuvs_nndescent")}
can call the external RAPIDS cuVS library. If this returns \code{FALSE}, those
backends fail clearly instead of using a CPU fallback with a cuVS label.
}
\examples{
cuvs_available()
}
