#include <Rcpp.h>

#include <string>

Rcpp::List native_hnsw_knn_impl(SEXP data,
                                int k,
                                int n_threads,
                                const std::string& metric,
                                double target_recall);
Rcpp::List native_hnsw_query_impl(SEXP data,
                                  SEXP query,
                                  int k,
                                  int n_threads,
                                  const std::string& metric,
                                  double target_recall);
bool native_metal_knn_available_impl();
Rcpp::List native_metal_knn_impl(SEXP data,
                                 int k,
                                 const std::string& method,
                                 const std::string& metric,
                                 double target_recall);
Rcpp::List native_metal_query_knn_impl(SEXP data,
                                       SEXP query,
                                       int k,
                                       const std::string& method,
                                       const std::string& metric,
                                       double target_recall);
bool native_cuda_knn_available_impl();
Rcpp::List native_cuda_knn_impl(SEXP data,
                                int k,
                                const std::string& method,
                                const std::string& metric,
                                double target_recall,
                                bool keep_gpu);
Rcpp::List native_cuda_query_knn_impl(SEXP data,
                                      SEXP query,
                                      int k,
                                      const std::string& method,
                                      const std::string& metric,
                                      double target_recall,
                                      bool keep_gpu);
Rcpp::List native_cuda_knn_to_host_impl(SEXP knn);

// [[Rcpp::export]]
Rcpp::List native_hnsw_knn_cpp(SEXP data,
                               int k,
                               int n_threads = 1,
                               std::string metric = "euclidean",
                               double target_recall = 0.99) {
  return native_hnsw_knn_impl(data, k, n_threads, metric, target_recall);
}

// [[Rcpp::export]]
Rcpp::List native_hnsw_query_cpp(SEXP data,
                                 SEXP query,
                                 int k,
                                 int n_threads = 1,
                                 std::string metric = "euclidean",
                                 double target_recall = 0.99) {
  return native_hnsw_query_impl(
    data, query, k, n_threads, metric, target_recall
  );
}

// [[Rcpp::export]]
bool native_metal_knn_available_cpp() {
  return native_metal_knn_available_impl();
}

// [[Rcpp::export]]
Rcpp::List native_metal_knn_cpp(SEXP data,
                                int k,
                                std::string method = "auto",
                                std::string metric = "euclidean",
                                double target_recall = 0.99) {
  return native_metal_knn_impl(data, k, method, metric, target_recall);
}

// [[Rcpp::export]]
Rcpp::List native_metal_query_knn_cpp(SEXP data,
                                      SEXP query,
                                      int k,
                                      std::string method = "auto",
                                      std::string metric = "euclidean",
                                      double target_recall = 0.99) {
  return native_metal_query_knn_impl(
    data, query, k, method, metric, target_recall
  );
}

// [[Rcpp::export]]
bool native_cuda_knn_available_cpp() {
  return native_cuda_knn_available_impl();
}

// [[Rcpp::export]]
Rcpp::List native_cuda_knn_cpp(SEXP data,
                               int k,
                               std::string method = "auto",
                               std::string metric = "euclidean",
                               double target_recall = 0.99,
                               bool keep_gpu = true) {
  return native_cuda_knn_impl(
    data, k, method, metric, target_recall, keep_gpu
  );
}

// [[Rcpp::export]]
Rcpp::List native_cuda_query_knn_cpp(SEXP data,
                                     SEXP query,
                                     int k,
                                     std::string method = "auto",
                                     std::string metric = "euclidean",
                                     double target_recall = 0.99,
                                     bool keep_gpu = true) {
  return native_cuda_query_knn_impl(
    data, query, k, method, metric, target_recall, keep_gpu
  );
}

// [[Rcpp::export]]
Rcpp::List native_cuda_knn_to_host_cpp(SEXP knn) {
  return native_cuda_knn_to_host_impl(knn);
}
