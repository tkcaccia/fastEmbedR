#include <Rcpp.h>

#include <climits>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef FASTEMBEDR_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include "graph_clustering_accel_common.h"
#include "graph_clustering_cuda_api.h"

// [[Rcpp::export]]
bool graph_clustering_cuda_available_cpp() {
  return graph_clustering_cuda_available_impl();
}

// [[Rcpp::export]]
Rcpp::List fastembedr_graph_cluster_cuda_cpp(
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
  if (!graph_clustering_cuda_available_impl()) {
    Rcpp::stop(
      "CUDA graph clustering is unavailable. Reinstall fastEmbedR with "
      "FASTEMBEDR_USE_CUDA=1 on a machine with a CUDA device."
    );
  }
  Rcpp::List result = fastembedr_graph_accel::cluster(
    from, to, weight, n_vertices, method, resolution, n_iterations, n_runs,
    seed, "cuda", graph_local_move_cuda_impl, force_wide
  );
  result["repeatability"] = "not_bitwise_deterministic";
  return result;
}

// [[Rcpp::export]]
Rcpp::List fastembedr_graph_cluster_cuda_devices_cpp(
    Rcpp::IntegerVector from,
    Rcpp::IntegerVector to,
    Rcpp::NumericVector weight,
    int n_vertices,
    std::string method,
    double resolution,
    int n_iterations,
    int n_runs,
    double seed,
    Rcpp::IntegerVector devices,
    bool force_wide = false) {
#ifdef FASTEMBEDR_HAS_CUDA
  namespace accel = fastembedr_graph_accel;
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1 ||
      devices.size() < 1 || devices.size() > static_cast<std::size_t>(count) ||
      n_vertices < devices.size() ||
      n_runs < 1 || n_iterations < 1 ||
      (method != "louvain" && method != "leiden") ||
      !std::isfinite(resolution) || resolution <= 0.0 ||
      !std::isfinite(seed) || seed != std::floor(seed) ||
      std::abs(seed) > INT_MAX) {
    Rcpp::stop("Invalid resident CUDA device clustering controls.");
  }
  std::vector<int> selected(devices.begin(), devices.end());
  std::vector<bool> seen(static_cast<std::size_t>(count), false);
  for (int device : selected) {
    if (device < 0 || device >= count || seen[device]) {
      Rcpp::stop("CUDA clustering devices must be distinct and visible.");
    }
    seen[device] = true;
  }
  std::vector<double> first_level_edges;
  double sync_bytes = 0.0;
  int color_phases = 0;
  const auto move = [&](const accel::Graph& graph,
      const std::vector<int>& initial,
      const std::vector<int>* parent, int passes, float gamma,
      std::uint64_t run_seed, bool refinement) {
    if (first_level_edges.empty()) {
      for (std::size_t i = 0; i < selected.size(); ++i) {
        const int begin = static_cast<int>(
          static_cast<std::int64_t>(graph.n) * i / selected.size());
        const int end = static_cast<int>(
          static_cast<std::int64_t>(graph.n) * (i + 1) /
          selected.size());
        first_level_edges.push_back(static_cast<double>(
          graph.offset(end) - graph.offset(begin)));
      }
    }
    auto moved = graph_local_move_cuda_devices_impl(graph, initial, parent,
      passes, gamma, run_seed, refinement, selected);
    const int phases = moved.passes * 8;
    color_phases += phases;
    sync_bytes += static_cast<double>(phases) *
      (12.0 * graph.n * selected.size() + 4.0 * graph.n);
    return moved;
  };
  Rcpp::List result = accel::cluster(from, to, weight, n_vertices,
    method, resolution, n_iterations, n_runs, seed, "cuda", move,
    force_wide);
  result["implementation"] = "resident_cuda_coordinated_host_sync";
  const std::string dtype = Rcpp::as<std::string>(
    result["csr_offset_dtype"]);
  result["graph_storage"] = dtype == "uint64" ?
    "partitioned_float32_csr_uint64" :
    "partitioned_float32_csr";
  result["coarsening_backend"] = "native_cpp";
  result["repeatability"] = "not_bitwise_deterministic";
  result["effective_gpu_workers"] =
    static_cast<int>(selected.size());
  result["gpu_devices"] = devices;
  result["synchronization"] = "host_labels_each_color";
  result["quality_safeguard"] = "host_modularity_each_pass";
  Rcpp::IntegerVector vertex_rows(selected.size());
  for (std::size_t i = 0; i < selected.size(); ++i) {
    const auto end = static_cast<std::int64_t>(n_vertices) *
      (i + 1) / selected.size();
    const auto begin = static_cast<std::int64_t>(n_vertices) *
      i / selected.size();
    vertex_rows[i] = static_cast<int>(end - begin);
  }
  result["vertex_rows_per_device"] = vertex_rows;
  result["csr_entries_per_device"] = first_level_edges;
  result["color_sync_phases"] = color_phases;
  result["estimated_host_sync_bytes"] = sync_bytes;
  return result;
#else
  Rcpp::stop("Resident CUDA device clustering requires CUDA.");
#endif
}
