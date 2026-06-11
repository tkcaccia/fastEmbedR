#include <Rcpp.h>

#include <string>

using Rcpp::List;
using Rcpp::NumericMatrix;

bool faiss_is_available_impl() {
  return false;
}

std::string faiss_info_json_impl() {
  return "{\"available\":false,\"reason\":\"package_not_built_with_faiss\"}";
}

List faiss_flat_knn_impl(NumericMatrix,
                         NumericMatrix,
                         int,
                         bool,
                         int) {
  Rcpp::stop(
    "FAISS backend is not available. Reinstall fastEmbedR with a FAISS C++ "
    "library visible to configure, for example FASTEMBEDR_USE_FAISS=1 and "
    "FAISS_HOME=/path/to/faiss."
  );
}

List faiss_ivf_knn_impl(NumericMatrix,
                        NumericMatrix,
                        int,
                        int,
                        int,
                        bool,
                        int) {
  Rcpp::stop(
    "FAISS IVF backend is not available. Reinstall fastEmbedR with a FAISS C++ "
    "library visible to configure, for example FASTEMBEDR_USE_FAISS=1 and "
    "FAISS_HOME=/path/to/faiss."
  );
}
