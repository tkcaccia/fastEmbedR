#include <Rcpp.h>
#include <faissR_api_v1.h>
#include <algorithm>
#include <cmath>

namespace {

void validate_faissr_batch(Rcpp::List result, int rows, int k,
                           const std::string& backend,
                           Rcpp::IntegerVector devices) {
  const std::string expected = backend == "cpu" ?
    "faiss_cpu_stream_exact" : backend == "metal" ?
    "faissr_native_metal_exact_file_batch" :
    devices.size() > 1 ? "faiss_gpu_flat_file_shards" :
    "faiss_gpu_flat_file_batch";
  Rcpp::IntegerMatrix ids(result["indices"]);
  Rcpp::NumericMatrix distances(result["distances"]);
  Rcpp::IntegerVector used(result["device_ids"]);
  const bool implicit_cuda = backend == "cuda" &&
    devices.size() == 1 && devices[0] == -1;
  const bool devices_match = implicit_cuda ?
    used.size() == 1 && used[0] >= 0 :
    used.size() == devices.size() &&
      std::equal(used.begin(), used.end(), devices.begin());
  const double build = Rcpp::as<double>(result["index_build_seconds"]);
  const double search = Rcpp::as<double>(result["search_seconds"]);
  const double total = Rcpp::as<double>(result["total_seconds"]);
  const double ram = Rcpp::as<double>(result["peak_ram_bytes_estimate"]);
  const double vram = Rcpp::as<double>(result["peak_device_bytes"]);
  const std::string residency = Rcpp::as<std::string>(
    result["residency"]);
  if (Rcpp::as<std::string>(result["provider"]) != "faissR" ||
      Rcpp::as<std::string>(result["backend_used"]) != expected ||
      Rcpp::as<std::string>(result["method"]) != "stream_exact" ||
      !Rcpp::as<bool>(result["exact"]) ||
      (residency != "resident" && residency != "mixed" &&
        residency != "file-backed") ||
      ids.nrow() != rows || ids.ncol() != k ||
      distances.nrow() != rows || distances.ncol() != k ||
      !devices_match ||
      !std::isfinite(build) || build < 0 ||
      !std::isfinite(search) || search < 0 ||
      !std::isfinite(total) || total < 0 ||
      !std::isfinite(ram) || ram < 0 ||
      !std::isfinite(vram) || vram < 0) {
    Rcpp::stop("faissR KNN batch changed shape, device, or metadata");
  }
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List faissr_matrix_knn_batch_cpp(SEXP query, SEXP reference,
    Rcpp::NumericVector row_ids, int k, std::string backend,
    std::string method, int reference_chunk, double memory_limit,
    int n_threads, bool exclude_self) {
  if (backend != "cpu" && backend != "metal" && backend != "cuda") {
    Rcpp::stop("Unsupported faissR KNN batch backend");
  }
  Rcpp::Function load_namespace =
    Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  Rcpp::List result = faissR_get_matrix_knn_batch_v1()(
    query, reference, row_ids, Rcpp::wrap(k), Rcpp::wrap(backend),
    Rcpp::wrap(method), Rcpp::wrap(reference_chunk),
    Rcpp::wrap(memory_limit), Rcpp::wrap(n_threads),
    Rcpp::wrap(exclude_self));
  validate_faissr_batch(result, row_ids.size(), k, backend,
    backend == "cpu" ? Rcpp::IntegerVector() :
      Rcpp::IntegerVector::create(backend == "cuda" ? -1 : 0));
  return result;
}

// [[Rcpp::export]]
Rcpp::List faissr_matrix_knn_batch_devices_cpp(SEXP query, SEXP reference,
    Rcpp::NumericVector row_ids, int k, int reference_chunk,
    double memory_limit, bool exclude_self,
    Rcpp::IntegerVector devices) {
  Rcpp::Function load_namespace =
    Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1 || devices.size() < 1) {
    Rcpp::stop("Unsupported faissR CUDA KNN capability");
  }
  Rcpp::List result = faissR_get_matrix_knn_batch_v2()(
    query, reference, row_ids, Rcpp::wrap(k), Rcpp::wrap("cuda"),
    Rcpp::wrap("stream_exact"), Rcpp::wrap(reference_chunk),
    Rcpp::wrap(memory_limit), Rcpp::wrap(1),
    Rcpp::wrap(exclude_self), devices);
  validate_faissr_batch(result, row_ids.size(), k, "cuda", devices);
  return result;
}

// [[Rcpp::export]]
Rcpp::List faissr_posting_search_cpp(SEXP query,
    std::string feature_path, std::string ids_path,
    std::string offsets_path, Rcpp::NumericMatrix centers,
    double first_row, int count, int k, int nprobe,
    int reference_chunk, int workers, bool exclude_self,
    double posting_rows, bool verbose) {
  Rcpp::Function load_namespace =
    Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  Rcpp::List out = faissR_get_posting_search_v1()(
    query, Rcpp::wrap(feature_path), Rcpp::wrap(ids_path),
    Rcpp::wrap(offsets_path), centers, Rcpp::wrap(first_row),
    Rcpp::wrap(count), Rcpp::wrap(k), Rcpp::wrap(nprobe),
    Rcpp::wrap(reference_chunk), Rcpp::wrap(workers),
    Rcpp::wrap(exclude_self), Rcpp::wrap(posting_rows),
    Rcpp::wrap(verbose));
  if (Rcpp::as<std::string>(out["provider"]) != "faissR" ||
      Rcpp::as<std::string>(out["backend_used"]) !=
        "faissr_cpu_posting_scan" ||
      Rcpp::as<std::string>(out["method"]) !=
        "physical_postings") {
    Rcpp::stop("faissR posting search changed provider or method");
  }
  return out;
}

// [[Rcpp::export]]
void massive_merge_knn_shard_cpp(std::string indices_path,
                                 std::string distances_path,
                                 Rcpp::IntegerMatrix indices,
                                 Rcpp::NumericMatrix distances,
                                 double query_first,
                                 double shard_first,
                                 double shard_rows,
                                 double n_vertices,
                                 int k, bool first_shard,
                                 double n_queries = -1.0,
                                 bool exclude_self = true) {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  Rcpp::List args = Rcpp::List::create(
      Rcpp::Named("indices_path") = indices_path,
      Rcpp::Named("distances_path") = distances_path,
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances,
      Rcpp::Named("query_first") = query_first,
      Rcpp::Named("shard_first") = shard_first,
      Rcpp::Named("shard_rows") = shard_rows,
      Rcpp::Named("n_vertices") = n_vertices,
      Rcpp::Named("k") = k,
      Rcpp::Named("first_shard") = first_shard,
      Rcpp::Named("n_queries") = n_queries,
      Rcpp::Named("exclude_self") = exclude_self);
  Rcpp::List out = faissR_get_merge_knn_shard_v1()(args);
  if (Rcpp::as<std::string>(out["provider"]) != "faissR" ||
      Rcpp::as<std::string>(out["method"]) != "shard_merge") {
    Rcpp::stop("faissR KNN shard merge changed provider or method");
  }
}

// [[Rcpp::export]]
Rcpp::IntegerVector massive_posting_distant_rows_cpp(
    Rcpp::List spec, Rcpp::NumericMatrix centers_r, int sample,
    int chunk_rows, int workers, int candidate_rows = 0) {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  Rcpp::List args = Rcpp::List::create(
      Rcpp::Named("spec") = spec,
      Rcpp::Named("centers") = centers_r,
      Rcpp::Named("sample") = sample,
      Rcpp::Named("chunk_rows") = chunk_rows,
      Rcpp::Named("workers") = workers,
      Rcpp::Named("candidate_rows") = candidate_rows);
  Rcpp::List out = faissR_get_posting_distant_rows_v1()(args);
  Rcpp::IntegerVector selected(out["indices"]);
  if (Rcpp::as<std::string>(out["provider"]) != "faissR" ||
      Rcpp::as<std::string>(out["backend_used"]) !=
        "faissr_cpu_posting_audit" ||
      Rcpp::as<std::string>(out["method"]) !=
        "distant_center_rows" ||
      selected.size() != sample) {
    Rcpp::stop("faissR posting audit changed provider or method");
  }
  return selected;
}

// [[Rcpp::export]]
Rcpp::List massive_coarse_postings_cpp(
    Rcpp::List spec, Rcpp::NumericMatrix centers_r,
    std::string prefix, int chunk_rows, int workers,
    SEXP resume_state, SEXP checkpoint_callback,
    int checkpoint_every) {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  Rcpp::List args = Rcpp::List::create(
      Rcpp::Named("spec") = spec,
      Rcpp::Named("centers") = centers_r,
      Rcpp::Named("prefix") = prefix,
      Rcpp::Named("chunk_rows") = chunk_rows,
      Rcpp::Named("workers") = workers,
      Rcpp::Named("resume_state") = resume_state,
      Rcpp::Named("checkpoint_callback") = checkpoint_callback,
      Rcpp::Named("checkpoint_every") = checkpoint_every);
  Rcpp::List out = faissR_get_posting_build_v1()(args);
  if (Rcpp::as<std::string>(out["provider"]) != "faissR" ||
      Rcpp::as<std::string>(out["backend_used"]) !=
        "faissr_cpu_posting_build" ||
      Rcpp::as<std::string>(out["method"]) !=
        "nearest_center_physical_postings" ||
      Rcpp::as<Rcpp::NumericVector>(out["counts"]).size() !=
        centers_r.nrow() ||
      Rcpp::as<Rcpp::NumericVector>(out["offsets"]).size() !=
        centers_r.nrow() + 1) {
    Rcpp::stop("faissR posting build changed provider or method");
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::List massive_read_knn_rows_cpp(
    std::string indices_path, std::string distances_path,
    double n_rows, int k, double n_reference,
    double first_row, int count, double max_bytes) {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  Rcpp::List args = Rcpp::List::create(
      Rcpp::Named("indices_path") = indices_path,
      Rcpp::Named("distances_path") = distances_path,
      Rcpp::Named("n_rows") = n_rows,
      Rcpp::Named("k") = k,
      Rcpp::Named("n_reference") = n_reference,
      Rcpp::Named("first_row") = first_row,
      Rcpp::Named("count") = count,
      Rcpp::Named("max_bytes") = max_bytes);
  Rcpp::List out = faissR_get_knn_rows_read_v1()(args);
  Rcpp::IntegerMatrix indices(out["indices"]);
  Rcpp::NumericMatrix distances(out["distances"]);
  if (Rcpp::as<std::string>(out["provider"]) != "faissR" ||
      Rcpp::as<std::string>(out["backend_used"]) !=
        "faissr_cpu_knn_rows_read" ||
      Rcpp::as<std::string>(out["method"]) !=
        "bounded_file_rows" ||
      indices.nrow() != count || indices.ncol() != k ||
      distances.nrow() != count || distances.ncol() != k) {
    Rcpp::stop("faissR KNN row reader changed provider or shape");
  }
  return Rcpp::List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances);
}

// [[Rcpp::export]]
void massive_reorder_posting_graph_cpp(
    std::string posting_ids_path,
    std::string grouped_indices_path,
    std::string grouped_distances_path,
    std::string output_indices_path,
    std::string output_distances_path,
    double n_vertices, int k, int bucket_rows,
    int read_rows, bool restart_work) {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  Rcpp::List args = Rcpp::List::create(
      Rcpp::Named("posting_ids_path") = posting_ids_path,
      Rcpp::Named("grouped_indices_path") = grouped_indices_path,
      Rcpp::Named("grouped_distances_path") =
        grouped_distances_path,
      Rcpp::Named("output_indices_path") =
        output_indices_path,
      Rcpp::Named("output_distances_path") =
        output_distances_path,
      Rcpp::Named("n_vertices") = n_vertices,
      Rcpp::Named("k") = k,
      Rcpp::Named("bucket_rows") = bucket_rows,
      Rcpp::Named("read_rows") = read_rows,
      Rcpp::Named("restart_work") = restart_work);
  Rcpp::List out = faissR_get_reorder_posting_graph_v1()(args);
  if (Rcpp::as<std::string>(out["provider"]) != "faissR" ||
      Rcpp::as<std::string>(out["backend_used"]) !=
        "faissr_cpu_knn_reorder" ||
      Rcpp::as<std::string>(out["method"]) !=
        "posting_graph_reorder" ||
      Rcpp::as<double>(out["n_vertices"]) != n_vertices ||
      Rcpp::as<int>(out["k"]) != k) {
    Rcpp::stop("faissR posting graph reorder changed provider");
  }
}

// [[Rcpp::export]]
SEXP faissr_hnsw_search_cpp(SEXP data, SEXP query, SEXP n, SEXP p,
                            SEXP k, SEXP target_recall, SEXP n_threads,
                            SEXP distance_storage) {
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  faissR_hnsw_search_v2_fun search = faissR_get_hnsw_search_v2();
  return search(data, query, n, p, k, target_recall, n_threads,
                distance_storage);
}

// [[Rcpp::export]]
SEXP faissr_gpu_multi_knn_cpp(SEXP data, int k,
                              std::string method,
                              Rcpp::IntegerVector devices,
                              int nlist, int nprobe,
                              bool shard, std::string output) {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  return faissR_get_gpu_multi_knn_v1()(
      data, Rcpp::wrap(k), Rcpp::wrap(method), devices,
      Rcpp::wrap(nlist), Rcpp::wrap(nprobe), Rcpp::wrap(shard),
      Rcpp::wrap(output));
}

// [[Rcpp::export]]
SEXP faissr_gpu_metric_ivf_knn_cpp(SEXP data, int k,
                                   std::string metric,
                                   Rcpp::IntegerVector devices,
                                   int nlist, int nprobe) {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  return faissR_get_gpu_metric_ivf_knn_v1()(
      data, Rcpp::wrap(k), Rcpp::wrap(metric), devices,
      Rcpp::wrap(nlist), Rcpp::wrap(nprobe));
}

// [[Rcpp::export]]
SEXP faissr_gpu_knn_sample_cpp(SEXP result, Rcpp::IntegerVector rows) {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  return faissR_get_gpu_knn_sample_v1()(result, rows);
}

// [[Rcpp::export]]
SEXP faissr_gpu_knn_retain_data_cpp(SEXP result, SEXP data) {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  return faissR_get_gpu_knn_retain_data_v1()(result, data);
}

// [[Rcpp::export]]
SEXP faissr_cuda_resident_knn_cpp(SEXP data, int k,
                                  std::string method,
                                  std::string metric,
                                  double target_recall) {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  return faissR_get_nn_cuda_tuned_gpu()(
      data, Rcpp::wrap(k), Rcpp::wrap(method), Rcpp::wrap(metric),
      Rcpp::wrap(false), Rcpp::wrap(target_recall));
}

// [[Rcpp::export]]
Rcpp::List faissr_gpu_index_build_cpp(SEXP data, int k,
                                       std::string method,
                                       Rcpp::IntegerVector devices,
                                       int nlist, int nprobe,
                                       bool shard) {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  return faissR_get_gpu_index_build_v1()(
      data, Rcpp::wrap(k), Rcpp::wrap(method), devices,
      Rcpp::wrap(nlist), Rcpp::wrap(nprobe), Rcpp::wrap(shard));
}

// [[Rcpp::export]]
Rcpp::List faissr_gpu_index_search_cpp(SEXP index, SEXP query, int k,
                                      std::string output = "double") {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  return faissR_get_gpu_index_search_v1()(
      index, query, Rcpp::wrap(k), Rcpp::wrap(output));
}

// [[Rcpp::export]]
int faissr_gpu_index_set_nprobe_cpp(SEXP index, int nprobe) {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  return Rcpp::as<int>(faissR_get_gpu_index_set_nprobe_v1()(
      index, Rcpp::wrap(nprobe)));
}

// [[Rcpp::export]]
SEXP faissr_metal_knn_cpp(SEXP data, int k,
                          std::string method,
                          std::string metric,
                          double target_recall) {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  return faissR_get_metal_knn_v1()(
      data, Rcpp::wrap(k), Rcpp::wrap(method),
      Rcpp::wrap(metric), Rcpp::wrap(target_recall));
}

// [[Rcpp::export]]
SEXP faissr_metal_query_knn_cpp(SEXP data, SEXP query, int k,
                                std::string method,
                                std::string metric,
                                double target_recall) {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  return faissR_get_metal_query_knn_v1()(
      data, query, Rcpp::wrap(k), Rcpp::wrap(method),
      Rcpp::wrap(metric), Rcpp::wrap(target_recall));
}

// [[Rcpp::export]]
Rcpp::List faissr_hnsw_index_build_cpp(SEXP data, int n, int p,
                                        int k, int n_threads,
                                        int m = 0,
                                        int ef_construction = 0,
                                        double target_recall = 0.99) {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  Rcpp::List tuning = faissR_get_hnsw_tune_v1()(
      Rcpp::wrap(n), Rcpp::wrap(p), Rcpp::wrap(k),
      Rcpp::wrap("euclidean"), Rcpp::wrap(target_recall));
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
  Rcpp::RObject fitted(pointer);
  m = Rcpp::as<int>(fitted.attr("m"));
  ef_construction = Rcpp::as<int>(fitted.attr("ef_construction"));
  return Rcpp::List::create(
      Rcpp::Named("pointer") = pointer,
      Rcpp::Named("method") = "faissr_hnsw_persistent",
      Rcpp::Named("M") = m,
      Rcpp::Named("efConstruction") = ef_construction,
      Rcpp::Named("efSearch") = ef_search,
      Rcpp::Named("tuning_rule") = tuning["rule"],
      Rcpp::Named("target_recall") = target_recall,
      Rcpp::Named("tuning_benchmark_source") =
          tuning["tuning_benchmark_source"]);
}

// [[Rcpp::export]]
Rcpp::List faissr_hnsw_file_index_build_cpp(SEXP source, int n, int p,
    int k, int n_threads, int m = 0, int ef_construction = 0,
    double target_recall = 0.99, int chunk_rows = 100000) {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  Rcpp::List tuning = faissR_get_hnsw_tune_v1()(
      Rcpp::wrap(n), Rcpp::wrap(p), Rcpp::wrap(k),
      Rcpp::wrap("euclidean"), Rcpp::wrap(target_recall));
  if ((m == 0) != (ef_construction == 0) || m < 0 ||
      ef_construction < 0 ||
      (m > 0 && (m < 2 || ef_construction < m))) {
    Rcpp::stop("Invalid explicit HNSW build settings");
  }
  if (m == 0) {
    m = Rcpp::as<int>(tuning["m"]);
    ef_construction = Rcpp::as<int>(tuning["ef_construction"]);
  } else {
    tuning["rule"] = "explicit";
  }
  const int ef_search = Rcpp::as<int>(tuning["ef_search"]);
  SEXP pointer = faissR_get_hnsw_file_index_build_v1()(
      source, Rcpp::wrap(m), Rcpp::wrap(ef_construction),
      Rcpp::wrap(n_threads), Rcpp::wrap(chunk_rows));
  Rcpp::RObject fitted(pointer);
  if (Rcpp::as<std::string>(fitted.attr("backend_used")) !=
      "faiss_cpu_hnsw_file_index") {
    Rcpp::stop("faissR changed the file-backed HNSW backend");
  }
  return Rcpp::List::create(
      Rcpp::Named("pointer") = pointer,
      Rcpp::Named("method") = "faissr_hnsw_persistent",
      Rcpp::Named("M") = fitted.attr("m"),
      Rcpp::Named("efConstruction") =
          fitted.attr("ef_construction"),
      Rcpp::Named("efSearch") = ef_search,
      Rcpp::Named("tuning_rule") = tuning["rule"],
      Rcpp::Named("target_recall") = target_recall,
      Rcpp::Named("tuning_benchmark_source") =
          tuning["tuning_benchmark_source"],
      Rcpp::Named("backend_used") =
          fitted.attr("backend_used"));
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

// [[Rcpp::export]]
void faissr_hnsw_index_save_cpp(SEXP index, std::string path) {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  faissR_get_hnsw_index_save_v1()(index, Rcpp::wrap(path));
}

// [[Rcpp::export]]
SEXP faissr_hnsw_index_load_cpp(std::string path, int n_threads) {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  return faissR_get_hnsw_index_load_v1()(
      Rcpp::wrap(path), Rcpp::wrap(n_threads));
}

// [[Rcpp::export]]
Rcpp::List faissr_hnsw_index_calibrate_cpp(SEXP index, SEXP query,
    Rcpp::IntegerVector self_ids, int k, double target_recall,
    int initial_ef, int max_ef, int n_threads) {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  return faissR_get_hnsw_index_calibrate_v2()(
      index, query, self_ids, Rcpp::wrap(k),
      Rcpp::wrap(target_recall), Rcpp::wrap(initial_ef),
      Rcpp::wrap(max_ef), Rcpp::wrap(n_threads));
}

// [[Rcpp::export]]
SEXP faissr_flat_index_build_cpp(SEXP data, int n_threads) {
  Rcpp::Function load_namespace =
      Rcpp::Environment::base_env()["loadNamespace"];
  load_namespace("faissR");
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  return faissR_get_flat_index_build_v1()(data, Rcpp::wrap(n_threads));
}

// [[Rcpp::export]]
Rcpp::List faissr_flat_index_search_cpp(SEXP index, SEXP query,
                                         int k, int n_threads) {
  if (faissR_c_api_version() != 1) {
    Rcpp::stop("Unsupported faissR C API version");
  }
  Rcpp::List result = faissR_get_flat_index_search_v1()(
      index, query, Rcpp::wrap(k), Rcpp::wrap(n_threads));
  result["backend"] = "cpu";
  result["method"] = "faissr_flat_query_reused";
  return result;
}
