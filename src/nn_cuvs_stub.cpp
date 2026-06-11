#include <Rcpp.h>

#include <string>

using Rcpp::List;
using Rcpp::NumericMatrix;

bool cuvs_is_available_impl() {
  return false;
}

std::string cuvs_info_json_impl() {
  return "{\"available\":false,\"reason\":\"package_not_built_with_cuvs\"}";
}

List cuvs_bruteforce_knn_impl(NumericMatrix,
                              NumericMatrix,
                              int,
                              bool) {
  Rcpp::stop(
    "cuVS backend is not available. Reinstall fastEmbedR with RAPIDS cuVS "
    "visible to configure, for example FASTEMBEDR_USE_CUVS=1 and "
    "CUVS_HOME=/path/to/cuvs."
  );
}

List cuvs_cagra_knn_impl(NumericMatrix,
                         NumericMatrix,
                         int,
                         bool,
                         int,
                         int,
                         int,
                         int) {
  Rcpp::stop(
    "cuVS CAGRA backend is not available. Reinstall fastEmbedR with RAPIDS "
    "cuVS visible to configure, for example FASTEMBEDR_USE_CUVS=1 and "
    "CUVS_HOME=/path/to/cuvs."
  );
}

List cuvs_nndescent_self_knn_impl(NumericMatrix,
                                  int,
                                  int,
                                  int,
                                  int) {
  Rcpp::stop(
    "cuVS NN-descent backend is not available. Reinstall fastEmbedR with "
    "RAPIDS cuVS visible to configure, for example FASTEMBEDR_USE_CUVS=1 "
    "and CUVS_HOME=/path/to/cuvs."
  );
}
