#include <Rcpp.h>
#include <faissR_api_v1.h>

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

// [[Rcpp::export]]
Rcpp::List faissr_hnsw_index_build_cpp(SEXP data, int n, int p,
                                        int k, int n_threads,
                                        int m = 0,
                                        int ef_construction = 0) {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  Rcpp::List tuning = faissR_get_hnsw_tune_v1()(
      Rcpp::wrap(n), Rcpp::wrap(p), Rcpp::wrap(k),
      Rcpp::wrap("euclidean"), Rcpp::wrap(0.99));
  if ((m == 0) != (ef_construction == 0)) {
    Rcpp::stop("M and efConstruction must be specified together");
  }
  if (m < 0 || ef_construction < 0 ||
      (m > 0 && (m < 2 || ef_construction < m))) {
    Rcpp::stop("Invalid explicit HNSW build settings");
  }
  const bool explicit_build = m > 0;
  if (!explicit_build) {
    m = Rcpp::as<int>(tuning["m"]);
    ef_construction = Rcpp::as<int>(tuning["ef_construction"]);
  } else {
    tuning["rule"] = "explicit";
  }
  const int ef_search = Rcpp::as<int>(tuning["ef_search"]);
  SEXP pointer = faissR_get_hnsw_index_build_v1()(
      data, Rcpp::wrap(m), Rcpp::wrap(ef_construction),
      Rcpp::wrap(n_threads));
  return Rcpp::List::create(
      Rcpp::Named("pointer") = pointer,
      Rcpp::Named("method") = "faissr_hnsw_persistent",
      Rcpp::Named("M") = m,
      Rcpp::Named("efConstruction") = ef_construction,
      Rcpp::Named("efSearch") = ef_search,
      Rcpp::Named("tuning_rule") = tuning["rule"]);
}

// [[Rcpp::export]]
Rcpp::List faissr_hnsw_index_search_cpp(SEXP index, SEXP query,
                                         int k, int ef_search,
                                         int n_threads) {
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  Rcpp::List result = faissR_get_hnsw_index_search_v1()(
      index, query, Rcpp::wrap(k), Rcpp::wrap(ef_search),
      Rcpp::wrap(n_threads));
  result["backend"] = "cpu";
  result["method"] = "faissr_hnsw_query_reused";
  return result;
}
