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
Rcpp::List native_cuda_knn_to_host_impl(SEXP knn);

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
Rcpp::List native_cuda_knn_to_host_cpp(SEXP knn) {
  return native_cuda_knn_to_host_impl(knn);
}
