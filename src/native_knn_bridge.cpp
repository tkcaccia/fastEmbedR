#include <Rcpp.h>

#include <string>

bool native_cuda_knn_available_impl();
Rcpp::List native_cuda_memory_info_impl();
Rcpp::List native_cuda_knn_impl(SEXP data,
                                int k,
                                const std::string& method,
                                const std::string& metric,
                                double target_recall,
                                bool keep_gpu,
                                bool retain_data);
Rcpp::List native_cuda_query_knn_impl(SEXP data,
                                      SEXP query,
                                      int k,
                                      const std::string& method,
                                      const std::string& metric,
                                      double target_recall,
                                      bool keep_gpu);
Rcpp::List native_cuda_knn_to_host_impl(SEXP knn);
SEXP native_cuda_index_build_impl(SEXP data,
                                  int k,
                                  const std::string& method,
                                  double target_recall);
Rcpp::List native_cuda_index_search_impl(SEXP pointer,
                                         SEXP query,
                                         int k);

// [[Rcpp::export]]
bool native_cuda_knn_available_cpp() {
  return native_cuda_knn_available_impl();
}

// [[Rcpp::export]]
Rcpp::List native_cuda_memory_info_cpp() {
  return native_cuda_memory_info_impl();
}

// [[Rcpp::export]]
Rcpp::List native_cuda_knn_cpp(SEXP data,
                               int k,
                               std::string method = "auto",
                               std::string metric = "euclidean",
                               double target_recall = 0.99,
                               bool keep_gpu = true,
                               bool retain_data = false) {
  return native_cuda_knn_impl(
    data, k, method, metric, target_recall, keep_gpu, retain_data
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

// [[Rcpp::export]]
SEXP native_cuda_index_build_cpp(SEXP data,
                                 int k,
                                 std::string method,
                                 double target_recall = 0.99) {
  return native_cuda_index_build_impl(data, k, method, target_recall);
}

// [[Rcpp::export]]
Rcpp::List native_cuda_index_search_cpp(SEXP pointer, SEXP query, int k) {
  return native_cuda_index_search_impl(pointer, query, k);
}
