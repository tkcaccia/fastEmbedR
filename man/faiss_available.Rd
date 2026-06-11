\name{faiss_available}
\alias{faiss_available}
\title{Check whether the real FAISS C++ backend is available}
\usage{
faiss_available()
}
\value{
\code{TRUE} when fastEmbedR was compiled and linked against FAISS.
}
\description{
Reports whether explicit FAISS KNN requests such as
\code{nn(x, backend = "faiss")} and \code{nn(x, backend = "faiss_ivf")} can
call the external FAISS C++ library. If this returns \code{FALSE}, those
backends fail clearly instead of using a CPU fallback with a FAISS label.
}
\examples{
faiss_available()
}
