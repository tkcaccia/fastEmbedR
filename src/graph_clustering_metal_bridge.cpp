#include <Rcpp.h>

#include "graph_clustering_accel_common.h"
#include "graph_clustering_metal_api.h"

// [[Rcpp::export]]
bool graph_clustering_metal_available_cpp() {
  return graph_clustering_metal_available_impl();
}

// [[Rcpp::export]]
std::string graph_clustering_metal_error_cpp() {
  return graph_clustering_metal_error_impl();
}

// [[Rcpp::export]]
Rcpp::List fastembedr_graph_cluster_metal_cpp(
    Rcpp::IntegerVector from,
    Rcpp::IntegerVector to,
    Rcpp::NumericVector weight,
    int n_vertices,
    std::string method = "leiden",
    double resolution = 1.0,
    int n_iterations = 10,
    int n_runs = 1,
    double seed = 1,
    bool force_wide = false) {
  if (!graph_clustering_metal_available_impl()) {
    Rcpp::stop(
      "Metal graph clustering is unavailable: " +
      graph_clustering_metal_error_impl()
    );
  }
  auto result = fastembedr_graph_accel::cluster(
    from, to, weight, n_vertices, method, resolution, n_iterations, n_runs,
    seed, "metal", graph_local_move_metal_impl, force_wide
  );
  result["metal_graph_input"] =
    Rcpp::as<std::string>(result["csr_offset_dtype"]) == "uint64" ?
    "bounded_resident_csr_tiles" : "resident_csr_full";
  return result;
}
