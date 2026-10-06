#include <Rcpp.h>
#include <faissR_api.h>

// [[Rcpp::export]]
SEXP faissr_hnsw_search_cpp(SEXP data, SEXP query, SEXP n, SEXP p,
                            SEXP k, SEXP target_recall, SEXP n_threads,
                            SEXP distance_storage) {
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  faissR_hnsw_search_v1_fun search = faissR_get_hnsw_search_v1();
  return search(data, query, n, p, k, target_recall, n_threads,
                distance_storage);
}
