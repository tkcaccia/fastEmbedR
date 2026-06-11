#include <Rcpp.h>

#include <string>

using Rcpp::List;
using Rcpp::NumericMatrix;

bool cuvs_is_available_impl();
std::string cuvs_info_json_impl();
List cuvs_bruteforce_knn_impl(NumericMatrix data,
                              NumericMatrix points,
                              int k,
                              bool exclude_self);
List cuvs_cagra_knn_impl(NumericMatrix data,
                         NumericMatrix points,
                         int k,
                         bool exclude_self,
                         int graph_degree,
                         int intermediate_graph_degree,
                         int search_width,
                         int itopk_size);
List cuvs_nndescent_self_knn_impl(NumericMatrix data,
                                  int k,
                                  int graph_degree,
                                  int intermediate_graph_degree,
                                  int max_iterations);

// [[Rcpp::export]]
bool cuvs_available_cpp() {
  return cuvs_is_available_impl();
}

// [[Rcpp::export]]
std::string cuvs_info_json_cpp() {
  return cuvs_info_json_impl();
}

// [[Rcpp::export]]
List nn_cuvs_bruteforce_cpp(NumericMatrix data,
                            NumericMatrix points,
                            int k,
                            bool exclude_self) {
  return cuvs_bruteforce_knn_impl(data, points, k, exclude_self);
}

// [[Rcpp::export]]
List nn_cuvs_cagra_cpp(NumericMatrix data,
                       NumericMatrix points,
                       int k,
                       bool exclude_self,
                       int graph_degree,
                       int intermediate_graph_degree,
                       int search_width,
                       int itopk_size) {
  return cuvs_cagra_knn_impl(
    data,
    points,
    k,
    exclude_self,
    graph_degree,
    intermediate_graph_degree,
    search_width,
    itopk_size
  );
}

// [[Rcpp::export]]
List nn_cuvs_nndescent_self_cpp(NumericMatrix data,
                                int k,
                                int graph_degree,
                                int intermediate_graph_degree,
                                int max_iterations) {
  return cuvs_nndescent_self_knn_impl(
    data,
    k,
    graph_degree,
    intermediate_graph_degree,
    max_iterations
  );
}
