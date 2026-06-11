#include <Rcpp.h>

#include <string>

using Rcpp::List;
using Rcpp::NumericMatrix;

bool faiss_is_available_impl();
std::string faiss_info_json_impl();
List faiss_flat_knn_impl(NumericMatrix data,
                         NumericMatrix points,
                         int k,
                         bool exclude_self,
                         int n_threads);
List faiss_ivf_knn_impl(NumericMatrix data,
                        NumericMatrix points,
                        int k,
                        int nlist,
                        int nprobe,
                        bool exclude_self,
                        int n_threads);

// [[Rcpp::export]]
bool faiss_available_cpp() {
  return faiss_is_available_impl();
}

// [[Rcpp::export]]
std::string faiss_info_json_cpp() {
  return faiss_info_json_impl();
}

// [[Rcpp::export]]
List nn_faiss_flat_cpp(NumericMatrix data,
                       NumericMatrix points,
                       int k,
                       bool exclude_self,
                       int n_threads) {
  return faiss_flat_knn_impl(data, points, k, exclude_self, n_threads);
}

// [[Rcpp::export]]
List nn_faiss_ivf_cpp(NumericMatrix data,
                      NumericMatrix points,
                      int k,
                      int nlist,
                      int nprobe,
                      bool exclude_self,
                      int n_threads) {
  return faiss_ivf_knn_impl(
    data, points, k, nlist, nprobe, exclude_self, n_threads
  );
}
