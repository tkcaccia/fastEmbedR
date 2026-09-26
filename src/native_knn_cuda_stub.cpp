#include <Rcpp.h>

#include <string>

bool native_cuda_knn_available_impl() {
  return false;
}

Rcpp::List native_cuda_knn_impl(SEXP,
                                int,
                                const std::string&,
                                const std::string&,
                                double,
                                bool,
                                bool) {
  Rcpp::stop(
    "Native CUDA KNN is unavailable. Reinstall fastEmbedR with CUDA and "
    "RAPIDS cuVS enabled (FASTEMBEDR_USE_CUDA=1, "
    "FASTEMBEDR_USE_CUVS=1, and CUVS_HOME=/path/to/rapids)."
  );
}

Rcpp::List native_cuda_query_knn_impl(SEXP,
                                      SEXP,
                                      int,
                                      const std::string&,
                                      const std::string&,
                                      double,
                                      bool) {
  Rcpp::stop(
    "Native CUDA query KNN is unavailable. Reinstall fastEmbedR with CUDA and "
    "RAPIDS cuVS enabled (FASTEMBEDR_USE_CUDA=1, "
    "FASTEMBEDR_USE_CUVS=1, and CUVS_HOME=/path/to/rapids)."
  );
}

Rcpp::List native_cuda_knn_to_host_impl(SEXP) {
  Rcpp::stop("The supplied KNN object is not backed by native fastEmbedR CUDA storage.");
}
