/*
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT
 *
 * Native CUDA KNN provider adapted from faissR commit
 * f37ea97c5774200025b1480770b8ecbf1d2d7919 (MIT), principally
 * src/nn_cuvs_impl.cpp, src/nn_cuda_impl.cpp, and src/nn_cuda_kernels.cpp.
 * Exact and IVF-Flat search call the installed RAPIDS cuVS C API. No cuVS or
 * RAFT source or binary is redistributed by fastEmbedR. See inst/NOTICE and
 * inst/LICENSES/.
 */

#include <Rcpp.h>

#include "native_knn_common.h"

#include <cuda_runtime.h>
#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/ivf_flat.h>
#include <dlpack/dlpack.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

extern "C" int fastembedr_cuda_finalize_cuvs_knn(
  const int64_t* input_indices,
  const float* input_distances,
  int* output_indices,
  float* output_distances,
  int n,
  int search_k,
  int output_k,
  int exclude_self,
  int distance_mode,
  int row_offset,
  int total_n
);
extern "C" int fastembedr_cuda_finalize_cuvs_query_knn(
  const int64_t* input_indices,
  const float* input_distances,
  int* output_indices,
  float* output_distances,
  int batch_n,
  int k,
  int distance_mode,
  int output_row_offset,
  int output_n,
  int reference_n
);
extern "C" const char* fastembedr_cuda_embedding_last_error();

Rcpp::List native_cuda_knn_to_host_impl(SEXP knn);

namespace {

constexpr int kMaxNativeCudaK = 256;
constexpr int kExactRowThreshold = 100000;

void cuda_check(cudaError_t status, const char* context) {
  if (status == cudaSuccess) return;
  Rcpp::stop("%s failed: %s", context, cudaGetErrorString(status));
}

void cuvs_check(cuvsError_t status, const char* context) {
  if (status == CUVS_SUCCESS) return;
  const char* detail = cuvsGetLastErrorText();
  if (detail != nullptr && detail[0] != '\0') {
    Rcpp::stop("%s failed: %s", context, detail);
  }
  Rcpp::stop("%s failed.", context);
}

DLManagedTensor make_tensor(void* data,
                            int64_t* shape,
                            int ndim,
                            DLDeviceType device_type,
                            uint8_t code,
                            uint8_t bits) {
  DLManagedTensor tensor{};
  tensor.dl_tensor.data = data;
  tensor.dl_tensor.device.device_type = device_type;
  tensor.dl_tensor.device.device_id = 0;
  tensor.dl_tensor.ndim = ndim;
  tensor.dl_tensor.dtype.code = code;
  tensor.dl_tensor.dtype.bits = bits;
  tensor.dl_tensor.dtype.lanes = 1;
  tensor.dl_tensor.shape = shape;
  tensor.dl_tensor.strides = nullptr;
  tensor.dl_tensor.byte_offset = 0;
  tensor.manager_ctx = nullptr;
  tensor.deleter = nullptr;
  return tensor;
}

cuvsFilter no_filter() {
  cuvsFilter filter;
  filter.type = NO_FILTER;
  filter.addr = static_cast<uintptr_t>(0);
  return filter;
}

class CuvsResources {
 public:
  CuvsResources() {
    cuvs_check(cuvsResourcesCreate(&resource_), "cuvsResourcesCreate");
  }
  ~CuvsResources() {
    if (resource_ != 0) cuvsResourcesDestroy(resource_);
  }
  cuvsResources_t get() const { return resource_; }
  CuvsResources(const CuvsResources&) = delete;
  CuvsResources& operator=(const CuvsResources&) = delete;

 private:
  cuvsResources_t resource_ = 0;
};

class CudaBuffer {
 public:
  CudaBuffer() = default;
  explicit CudaBuffer(std::size_t bytes) { reset(bytes); }
  ~CudaBuffer() { reset(0); }
  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;

  void reset(std::size_t bytes) {
    if (pointer_ != nullptr) cudaFree(pointer_);
    pointer_ = nullptr;
    bytes_ = 0;
    if (bytes > 0) {
      cuda_check(cudaMalloc(&pointer_, bytes), "cudaMalloc(native cuVS KNN)");
      bytes_ = bytes;
    }
  }
  void* get() const { return pointer_; }
  std::size_t bytes() const { return bytes_; }
  void* release() {
    void* out = pointer_;
    pointer_ = nullptr;
    bytes_ = 0;
    return out;
  }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0;
};

class BruteForceIndex {
 public:
  BruteForceIndex() {
    cuvs_check(cuvsBruteForceIndexCreate(&index_), "cuvsBruteForceIndexCreate");
  }
  ~BruteForceIndex() {
    if (index_ != nullptr) cuvsBruteForceIndexDestroy(index_);
  }
  cuvsBruteForceIndex_t get() const { return index_; }
  BruteForceIndex(const BruteForceIndex&) = delete;
  BruteForceIndex& operator=(const BruteForceIndex&) = delete;

 private:
  cuvsBruteForceIndex_t index_ = nullptr;
};

class IvfFlatIndexParams {
 public:
  IvfFlatIndexParams() {
    cuvs_check(cuvsIvfFlatIndexParamsCreate(&params_), "cuvsIvfFlatIndexParamsCreate");
  }
  ~IvfFlatIndexParams() {
    if (params_ != nullptr) cuvsIvfFlatIndexParamsDestroy(params_);
  }
  cuvsIvfFlatIndexParams_t get() const { return params_; }
  IvfFlatIndexParams(const IvfFlatIndexParams&) = delete;
  IvfFlatIndexParams& operator=(const IvfFlatIndexParams&) = delete;

 private:
  cuvsIvfFlatIndexParams_t params_ = nullptr;
};

class IvfFlatSearchParams {
 public:
  IvfFlatSearchParams() {
    cuvs_check(cuvsIvfFlatSearchParamsCreate(&params_), "cuvsIvfFlatSearchParamsCreate");
  }
  ~IvfFlatSearchParams() {
    if (params_ != nullptr) cuvsIvfFlatSearchParamsDestroy(params_);
  }
  cuvsIvfFlatSearchParams_t get() const { return params_; }
  IvfFlatSearchParams(const IvfFlatSearchParams&) = delete;
  IvfFlatSearchParams& operator=(const IvfFlatSearchParams&) = delete;

 private:
  cuvsIvfFlatSearchParams_t params_ = nullptr;
};

class IvfFlatIndex {
 public:
  IvfFlatIndex() {
    cuvs_check(cuvsIvfFlatIndexCreate(&index_), "cuvsIvfFlatIndexCreate");
  }
  ~IvfFlatIndex() {
    if (index_ != nullptr) cuvsIvfFlatIndexDestroy(index_);
  }
  cuvsIvfFlatIndex_t get() const { return index_; }
  IvfFlatIndex(const IvfFlatIndex&) = delete;
  IvfFlatIndex& operator=(const IvfFlatIndex&) = delete;

 private:
  cuvsIvfFlatIndex_t index_ = nullptr;
};

struct IvfTuning {
  int nlist;
  int nprobe;
  std::string rule;
};

int k_bucket(int k) {
  if (k <= 15) return 15;
  if (k <= 30) return 30;
  if (k <= 50) return 50;
  return 100;
}

int target_code(double target_recall) {
  if (target_recall >= 0.985) return 99;
  if (target_recall >= 0.925) return 95;
  return 90;
}

IvfTuning tune_ivf(int n, int p, int k, double target_recall) {
  const int root = std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n)))));
  const int half_root = std::max(16, root / 2);
  const int bucket = k_bucket(k);
  const int recall = target_code(target_recall);
  int nlist = root;
  int nprobe = recall >= 99 ? 16 : (recall >= 95 ? 8 : 4);
  std::string rule = "general";

  if (n >= 500000 && p >= 256) {
    nlist = recall >= 99 ? (bucket >= 100 ? 4096 : 2048) : 1024;
    nprobe = recall >= 99 ? (bucket >= 100 ? 192 : 69) :
      (bucket >= 100 ? 17 : 16);
    rule = "large_high_dim";
  } else if (n >= 500000 && p <= 64) {
    nlist = n >= 1000000 ? 1024 : 512;
    nprobe = recall >= 99 ? (bucket <= 15 ? 16 : 32) :
      (recall >= 95 ? 9 : 6);
    rule = "large_low_dim";
  } else if (n >= 20000 && n < 200000 && p >= 256) {
    nlist = bucket <= 15 ? half_root :
      (bucket <= 50 ? half_root * 2 : root);
    nprobe = recall >= 99 ?
      (bucket <= 15 ? 8 : (bucket <= 30 ? 17 : (bucket <= 50 ? 36 : 17))) :
      (recall >= 95 ? (bucket <= 15 ? 4 : 8) : 4);
    rule = "medium_high_dim";
  } else if (n >= 20000 && n < 200000 && p <= 128) {
    nlist = bucket <= 15 ? root : (bucket <= 30 ? root * 4 : half_root);
    nprobe = recall >= 99 ? (bucket <= 15 ? 9 : (bucket <= 30 ? 99 : 17)) :
      (recall >= 95 ? 8 : 4);
    rule = "medium_low_dim";
  } else if (p >= 128) {
    nlist = std::max(32, root);
    nprobe = recall >= 99 ? std::max(16, root / 8) :
      (recall >= 95 ? std::max(8, root / 16) : std::max(4, root / 32));
    rule = "high_dim";
  }

  nlist = std::max(1, std::min(nlist, n));
  nprobe = std::max(1, std::min(nprobe, nlist));
  return {nlist, nprobe, rule};
}

double pilot_recall(const std::vector<int64_t>& reference,
                    const std::vector<int64_t>& observed,
                    const std::vector<int>& query_rows,
                    int search_k,
                    int output_k) {
  double matches = 0.0;
  double total = 0.0;
  std::vector<int64_t> expected;
  std::vector<int64_t> candidate;
  expected.reserve(output_k);
  candidate.reserve(output_k);
  for (std::size_t row = 0; row < query_rows.size(); ++row) {
    expected.clear();
    candidate.clear();
    const std::size_t offset = row * search_k;
    for (int column = 0; column < search_k &&
         static_cast<int>(expected.size()) < output_k; ++column) {
      const int64_t index = reference[offset + column];
      if (index >= 0 && index != query_rows[row]) expected.push_back(index);
    }
    for (int column = 0; column < search_k &&
         static_cast<int>(candidate.size()) < output_k; ++column) {
      const int64_t index = observed[offset + column];
      if (index >= 0 && index != query_rows[row]) candidate.push_back(index);
    }
    for (int64_t index : candidate) {
      if (std::find(expected.begin(), expected.end(), index) != expected.end()) {
        matches += 1.0;
      }
    }
    total += static_cast<double>(output_k);
  }
  return total > 0.0 ? matches / total : 0.0;
}

double query_pilot_recall(const std::vector<int64_t>& reference,
                          const std::vector<int64_t>& observed,
                          int rows,
                          int k) {
  double matches = 0.0;
  for (int row = 0; row < rows; ++row) {
    const std::size_t offset = static_cast<std::size_t>(row) * k;
    for (int candidate = 0; candidate < k; ++candidate) {
      const int64_t index = observed[offset + candidate];
      if (index < 0) continue;
      for (int expected = 0; expected < k; ++expected) {
        if (index == reference[offset + expected]) {
          matches += 1.0;
          break;
        }
      }
    }
  }
  const double total = static_cast<double>(rows) * k;
  return total > 0.0 ? matches / total : 0.0;
}

int distance_mode(fastembedr::KnnMetric metric) {
  if (metric == fastembedr::KnnMetric::Euclidean) return 0;
  return 1;
}

struct NativeCudaKnnHandle {
  int* indices = nullptr;
  float* distances = nullptr;
  int n = 0;
  int k = 0;
  int device = 0;
};

void native_cuda_knn_finalizer(SEXP pointer) {
  if (TYPEOF(pointer) != EXTPTRSXP) return;
  auto* handle = static_cast<NativeCudaKnnHandle*>(R_ExternalPtrAddr(pointer));
  if (handle == nullptr) return;
  cudaSetDevice(handle->device);
  if (handle->indices != nullptr) cudaFree(handle->indices);
  if (handle->distances != nullptr) cudaFree(handle->distances);
  delete handle;
  R_ClearExternalPtr(pointer);
}

NativeCudaKnnHandle* native_cuda_handle(SEXP object) {
  if (!Rf_isNewList(object)) {
    Rcpp::stop("Expected a native fastEmbedR CUDA KNN object.");
  }
  Rcpp::List result(object);
  if (!result.containsElementNamed("handle")) {
    Rcpp::stop("Native CUDA KNN object is missing its owning handle.");
  }
  SEXP pointer = result["handle"];
  if (TYPEOF(pointer) != EXTPTRSXP) {
    Rcpp::stop("Native CUDA KNN handle is invalid.");
  }
  auto* handle = static_cast<NativeCudaKnnHandle*>(R_ExternalPtrAddr(pointer));
  if (handle == nullptr || handle->indices == nullptr || handle->distances == nullptr) {
    Rcpp::stop("Native CUDA KNN storage has already been released.");
  }
  return handle;
}

Rcpp::List make_gpu_result(NativeCudaKnnHandle* handle,
                           const std::string& method,
                           const std::string& metric,
                           double target_recall,
                           bool exact,
                           const IvfTuning& tuning,
                           int search_batch_size,
                           bool exclude_self = true,
                           int n_reference = -1) {
  SEXP owner = PROTECT(R_MakeExternalPtr(handle, R_NilValue, R_NilValue));
  R_RegisterCFinalizerEx(owner, native_cuda_knn_finalizer, TRUE);
  SEXP indices_ptr = PROTECT(R_MakeExternalPtr(handle->indices, R_NilValue, owner));
  SEXP distances_ptr = PROTECT(R_MakeExternalPtr(handle->distances, R_NilValue, owner));
  const std::string backend_used = exact ?
    "native_cuda_cuvs_exact" : "native_cuda_cuvs_ivf_flat";
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("handle") = owner,
    Rcpp::Named("indices_ptr") = indices_ptr,
    Rcpp::Named("distances_ptr") = distances_ptr,
    Rcpp::Named("n_query") = handle->n,
    Rcpp::Named("n") = handle->n,
    Rcpp::Named("k") = handle->k,
    Rcpp::Named("index_base") = 1,
    Rcpp::Named("indices_type") = "int32",
    Rcpp::Named("distance_type") = "float32",
    Rcpp::Named("indices_residency") = "cuda_device",
    Rcpp::Named("distance_residency") = "cuda_device",
    Rcpp::Named("result_residency") = "cuda",
    Rcpp::Named("layout") = "column_major_query_by_k",
    Rcpp::Named("metric") = metric,
    Rcpp::Named("backend_used") = backend_used,
    Rcpp::Named("method") = method,
    Rcpp::Named("accelerator") = "cuda",
    Rcpp::Named("gpu_provider") = "fastEmbedR_native_cuvs",
    Rcpp::Named("device") = handle->device,
    Rcpp::Named("exact") = exact,
    Rcpp::Named("exclude_self") = exclude_self,
    Rcpp::Named("n_reference") =
      n_reference > 0 ? n_reference : handle->n,
    Rcpp::Named("target_recall") = target_recall,
    Rcpp::Named("tuning") = exact ? "exact" : "deterministic_recall_pilot",
    Rcpp::Named("nlist") = exact ? NA_INTEGER : tuning.nlist,
    Rcpp::Named("nprobe") = exact ? NA_INTEGER : tuning.nprobe,
    Rcpp::Named("tuning_rule") = exact ?
      "cuvs_exact_below_100k" : tuning.rule,
    Rcpp::Named("search_batch_size") = search_batch_size,
    Rcpp::Named("input_type") = "float32",
    Rcpp::Named("device_to_host_result_copies") = 0,
    Rcpp::Named("cpu_fallback") = false
  );
  out.attr("class") = Rcpp::CharacterVector::create(
    "fastEmbedR_gpu_knn", "list"
  );
  out.attr("metric") = metric;
  out.attr("backend_used") = backend_used;
  out.attr("result_residency") = "cuda";
  out.attr("exclude_self") = exclude_self;
  UNPROTECT(3);
  return out;
}

}  // namespace

bool native_cuda_knn_available_impl() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

Rcpp::List native_cuda_knn_impl(SEXP data,
                                int k,
                                const std::string& method,
                                const std::string& metric,
                                double target_recall,
                                bool keep_gpu) {
  if (!native_cuda_knn_available_impl()) {
    Rcpp::stop("No CUDA device is available for native cuVS KNN.");
  }
  const fastembedr::KnnMetric parsed_metric = fastembedr::parse_knn_metric(metric);
  fastembedr::FloatMatrix matrix =
    fastembedr::matrix_to_row_major_float(data, parsed_metric);
  if (matrix.nrow < 2) {
    Rcpp::stop("`data` must contain at least two rows and one column.");
  }
  if (k < 1 || k >= input_nrow || k > kMaxNativeCudaK) {
    Rcpp::stop("Native CUDA KNN requires k in [1, min(n - 1, %d)].", kMaxNativeCudaK);
  }
  if (!std::isfinite(target_recall) || target_recall < 0.8 || target_recall > 1.0) {
    Rcpp::stop("`target_recall` must be between 0.8 and 1.");
  }

  std::string resolved = method;
  std::transform(resolved.begin(), resolved.end(), resolved.begin(),
                 [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
  if (resolved == "auto") {
    resolved = matrix.nrow < kExactRowThreshold ? "exact" : "ivf";
  }
  if (resolved == "flat" || resolved == "bruteforce") resolved = "exact";
  if (resolved != "exact" && resolved != "ivf") {
    Rcpp::stop("Native CUDA KNN supports only internal `exact`, `ivf`, or `auto` routes.");
  }
  const bool exact = resolved == "exact";
  if (matrix.nrow < 2 || matrix.ncol < 1) {
    Rcpp::stop("`data` must contain at least two rows and one column.");
  }
  const int search_k = k + 1;
  const std::size_t data_items =
    static_cast<std::size_t>(matrix.nrow) * matrix.ncol;
  const std::size_t final_items =
    static_cast<std::size_t>(matrix.nrow) * k;

  CuvsResources resources;
  CudaBuffer dataset(data_items * sizeof(float));
  CudaBuffer output_indices(final_items * sizeof(int));
  CudaBuffer output_distances(final_items * sizeof(float));
  const int pilot_n = exact ? 0 : std::min(matrix.nrow, 256);
  std::vector<int> pilot_rows;
  std::vector<float> pilot_values;
  if (!exact) {
    pilot_rows.resize(pilot_n);
    pilot_values.resize(static_cast<std::size_t>(pilot_n) * matrix.ncol);
    for (int row = 0; row < pilot_n; ++row) {
      const int source_row = pilot_n == 1 ? 0 : static_cast<int>(
        (static_cast<int64_t>(row) * (matrix.nrow - 1)) / (pilot_n - 1)
      );
      pilot_rows[row] = source_row;
      std::copy_n(
        matrix.values.data() + static_cast<std::size_t>(source_row) * matrix.ncol,
        matrix.ncol,
        pilot_values.data() + static_cast<std::size_t>(row) * matrix.ncol
      );
    }
  }
  cuda_check(
    cudaMemcpy(dataset.get(),
               matrix.values.data(),
               data_items * sizeof(float),
               cudaMemcpyHostToDevice),
    "cudaMemcpy(native CUDA KNN dataset H2D)"
  );
  matrix.values.clear();
  matrix.values.shrink_to_fit();

  int64_t dataset_shape[2] = {matrix.nrow, matrix.ncol};
  DLManagedTensor dataset_tensor = make_tensor(
    dataset.get(), dataset_shape, 2, kDLCUDA, kDLFloat, 32
  );
  const auto distance = L2Expanded;
  IvfTuning tuning = tune_ivf(matrix.nrow, matrix.ncol, k, target_recall);
  int search_batch_size = matrix.nrow;
  double measured_pilot_recall = exact ? 1.0 : 0.0;
  int tuning_attempts = 0;
  auto finalize_batch = [&](const CudaBuffer& raw_indices,
                            const CudaBuffer& raw_distances,
                            int batch_n,
                            int row_offset) {
    const int status = fastembedr_cuda_finalize_cuvs_knn(
      static_cast<const int64_t*>(raw_indices.get()),
      static_cast<const float*>(raw_distances.get()),
      static_cast<int*>(output_indices.get()),
      static_cast<float*>(output_distances.get()),
      batch_n,
      search_k,
      k,
      1,
      distance_mode(parsed_metric),
      row_offset,
      matrix.nrow
    );
    if (status != 0) {
      const char* detail = fastembedr_cuda_embedding_last_error();
      Rcpp::stop(
        "Native CUDA KNN postprocessing failed: %s",
        detail == nullptr ? "unknown CUDA error" : detail
      );
    }
  };

  if (exact) {
    const std::size_t result_items =
      static_cast<std::size_t>(matrix.nrow) * search_k;
    CudaBuffer raw_indices(result_items * sizeof(int64_t));
    CudaBuffer raw_distances(result_items * sizeof(float));
    int64_t output_shape[2] = {matrix.nrow, search_k};
    DLManagedTensor neighbors_tensor = make_tensor(
      raw_indices.get(), output_shape, 2, kDLCUDA, kDLInt, 64
    );
    DLManagedTensor distances_tensor = make_tensor(
      raw_distances.get(), output_shape, 2, kDLCUDA, kDLFloat, 32
    );
    BruteForceIndex index;
    cuvs_check(
      cuvsBruteForceBuild(
        resources.get(), &dataset_tensor, distance, 0.0f, index.get()
      ),
      "cuvsBruteForceBuild"
    );
    cuvs_check(
      cuvsBruteForceSearch(
        resources.get(), index.get(), &dataset_tensor,
        &neighbors_tensor, &distances_tensor, no_filter()
      ),
      "cuvsBruteForceSearch"
    );
    cuvs_check(
      cuvsStreamSync(resources.get()),
      "cuvsStreamSync(brute-force search)"
    );
    finalize_batch(raw_indices, raw_distances, matrix.nrow, 0);
  } else {
    IvfFlatIndexParams index_params;
    index_params.get()->metric = distance;
    index_params.get()->add_data_on_build = true;
    index_params.get()->n_lists = static_cast<uint32_t>(tuning.nlist);
    index_params.get()->kmeans_n_iters = 20;
    index_params.get()->kmeans_trainset_fraction = 1.0;
    index_params.get()->adaptive_centers = false;
    index_params.get()->conservative_memory_allocation = true;
    IvfFlatIndex index;
    cuvs_check(
      cuvsIvfFlatBuild(
        resources.get(), index_params.get(), &dataset_tensor, index.get()
      ),
      "cuvsIvfFlatBuild"
    );
    CudaBuffer pilot_dataset(pilot_values.size() * sizeof(float));
    cuda_check(
      cudaMemcpy(pilot_dataset.get(), pilot_values.data(),
                 pilot_values.size() * sizeof(float), cudaMemcpyHostToDevice),
      "cudaMemcpy(native cuVS IVF pilot H2D)"
    );
    std::vector<float>().swap(pilot_values);
    const std::size_t pilot_items = static_cast<std::size_t>(pilot_n) * search_k;
    CudaBuffer pilot_exact_indices(pilot_items * sizeof(int64_t));
    CudaBuffer pilot_exact_distances(pilot_items * sizeof(float));
    CudaBuffer pilot_ivf_indices(pilot_items * sizeof(int64_t));
    CudaBuffer pilot_ivf_distances(pilot_items * sizeof(float));
    int64_t pilot_query_shape[2] = {pilot_n, matrix.ncol};
    int64_t pilot_output_shape[2] = {pilot_n, search_k};
    DLManagedTensor pilot_query_tensor = make_tensor(
      pilot_dataset.get(), pilot_query_shape, 2, kDLCUDA, kDLFloat, 32
    );
    DLManagedTensor pilot_exact_indices_tensor = make_tensor(
      pilot_exact_indices.get(), pilot_output_shape, 2, kDLCUDA, kDLInt, 64
    );
    DLManagedTensor pilot_exact_distances_tensor = make_tensor(
      pilot_exact_distances.get(), pilot_output_shape, 2, kDLCUDA, kDLFloat, 32
    );
    DLManagedTensor pilot_ivf_indices_tensor = make_tensor(
      pilot_ivf_indices.get(), pilot_output_shape, 2, kDLCUDA, kDLInt, 64
    );
    DLManagedTensor pilot_ivf_distances_tensor = make_tensor(
      pilot_ivf_distances.get(), pilot_output_shape, 2, kDLCUDA, kDLFloat, 32
    );
    BruteForceIndex pilot_reference_index;
    cuvs_check(
      cuvsBruteForceBuild(
        resources.get(), &dataset_tensor, distance, 0.0f,
        pilot_reference_index.get()
      ),
      "cuvsBruteForceBuild(IVF pilot oracle)"
    );
    cuvs_check(
      cuvsBruteForceSearch(
        resources.get(), pilot_reference_index.get(), &pilot_query_tensor,
        &pilot_exact_indices_tensor, &pilot_exact_distances_tensor, no_filter()
      ),
      "cuvsBruteForceSearch(IVF pilot oracle)"
    );
    cuvs_check(
      cuvsStreamSync(resources.get()),
      "cuvsStreamSync(IVF pilot oracle)"
    );
    std::vector<int64_t> pilot_reference(pilot_items);
    std::vector<int64_t> pilot_observed(pilot_items);
    cuda_check(
      cudaMemcpy(pilot_reference.data(), pilot_exact_indices.get(),
                 pilot_items * sizeof(int64_t), cudaMemcpyDeviceToHost),
      "cudaMemcpy(IVF pilot oracle D2H)"
    );

    IvfFlatSearchParams search_params;
    const double pilot_target = std::min(
      1.0, target_recall + (target_recall >= 0.985 ? 0.005 : 0.002)
    );
    int probe = std::max(1, tuning.nprobe);
    while (true) {
      ++tuning_attempts;
      search_params.get()->n_probes = static_cast<uint32_t>(probe);
      cuvs_check(
        cuvsIvfFlatSearch(
          resources.get(), search_params.get(), index.get(),
          &pilot_query_tensor, &pilot_ivf_indices_tensor,
          &pilot_ivf_distances_tensor, no_filter()
        ),
        "cuvsIvfFlatSearch(recall pilot)"
      );
      cuvs_check(
        cuvsStreamSync(resources.get()),
        "cuvsStreamSync(IVF recall pilot)"
      );
      cuda_check(
        cudaMemcpy(pilot_observed.data(), pilot_ivf_indices.get(),
                   pilot_items * sizeof(int64_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy(IVF pilot result D2H)"
      );
      measured_pilot_recall = pilot_recall(
        pilot_reference, pilot_observed, pilot_rows, search_k, k
      );
      tuning.nprobe = probe;
      if (measured_pilot_recall >= pilot_target || probe >= tuning.nlist) break;
      probe = std::min(tuning.nlist, std::max(probe + 1, probe * 2));
    }
    tuning.rule += "_recall_pilot";
    search_params.get()->n_probes = static_cast<uint32_t>(tuning.nprobe);
    search_batch_size = std::min(matrix.nrow, 32768);
    const std::size_t batch_items =
      static_cast<std::size_t>(search_batch_size) * search_k;
    CudaBuffer raw_indices(batch_items * sizeof(int64_t));
    CudaBuffer raw_distances(batch_items * sizeof(float));
    for (int offset = 0; offset < matrix.nrow; offset += search_batch_size) {
      const int current = std::min(search_batch_size, matrix.nrow - offset);
      int64_t query_shape[2] = {current, matrix.ncol};
      int64_t output_shape[2] = {current, search_k};
      auto* query = static_cast<float*>(dataset.get()) +
        static_cast<std::size_t>(offset) * matrix.ncol;
      DLManagedTensor query_tensor = make_tensor(
        query, query_shape, 2, kDLCUDA, kDLFloat, 32
      );
      DLManagedTensor neighbors_tensor = make_tensor(
        raw_indices.get(), output_shape, 2, kDLCUDA, kDLInt, 64
      );
      DLManagedTensor distances_tensor = make_tensor(
        raw_distances.get(), output_shape, 2, kDLCUDA, kDLFloat, 32
      );
      cuvs_check(
        cuvsIvfFlatSearch(
          resources.get(), search_params.get(), index.get(),
          &query_tensor, &neighbors_tensor, &distances_tensor, no_filter()
        ),
        "cuvsIvfFlatSearch"
      );
      cuvs_check(
        cuvsStreamSync(resources.get()),
        "cuvsStreamSync(IVF-Flat search)"
      );
      finalize_batch(raw_indices, raw_distances, current, offset);
    }
  }
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(native CUDA KNN output)");

  auto* handle = new NativeCudaKnnHandle();
  handle->indices = static_cast<int*>(output_indices.release());
  handle->distances = static_cast<float*>(output_distances.release());
  handle->n = matrix.nrow;
  handle->k = k;
  cuda_check(cudaGetDevice(&handle->device), "cudaGetDevice(native CUDA KNN)");
  Rcpp::List result = make_gpu_result(
    handle, resolved, metric, target_recall, exact, tuning, search_batch_size
  );
  result["resident_result_bytes"] = static_cast<double>(final_items) *
    static_cast<double>(sizeof(int) + sizeof(float));
  const double raw_batch_bytes =
    static_cast<double>(search_batch_size) * static_cast<double>(search_k) *
    static_cast<double>(sizeof(int64_t) + sizeof(float));
  const double pilot_bytes = static_cast<double>(pilot_n) *
    (static_cast<double>(matrix.ncol) * sizeof(float) +
     static_cast<double>(search_k) *
       2.0 * static_cast<double>(sizeof(int64_t) + sizeof(float)));
  result["peak_temporary_search_bytes"] = std::max(raw_batch_bytes, pilot_bytes);
  result["input_was_float32"] = matrix.input_float32;
  result["pilot_rows"] = pilot_n;
  result["pilot_recall"] = measured_pilot_recall;
  result["tuning_attempts"] = tuning_attempts;
  result["target_met"] = exact || measured_pilot_recall >= target_recall;
  result.attr("class") = Rcpp::CharacterVector::create(
    "fastEmbedR_gpu_knn", "list"
  );
  if (keep_gpu) return result;
  return native_cuda_knn_to_host_impl(result);
}

Rcpp::List native_cuda_query_knn_impl(SEXP data,
                                      SEXP query,
                                      int k,
                                      const std::string& method,
                                      const std::string& metric,
                                      double target_recall,
                                      bool keep_gpu) {
  if (!native_cuda_knn_available_impl()) {
    Rcpp::stop("No CUDA device is available for native query KNN.");
  }
  const fastembedr::KnnMetric parsed_metric =
    fastembedr::parse_knn_metric(metric);
  fastembedr::FloatMatrix reference =
    fastembedr::matrix_to_row_major_float(data, parsed_metric);
  fastembedr::FloatMatrix queries =
    fastembedr::matrix_to_row_major_float(query, parsed_metric);
  if (reference.nrow < 1 || queries.nrow < 1 ||
      reference.ncol < 1 || reference.ncol != queries.ncol) {
    Rcpp::stop(
      "`data` and `query` must have at least one row and matching columns."
    );
  }
  if (k < 1 || k > reference.nrow || k > kMaxNativeCudaK) {
    Rcpp::stop(
      "Native CUDA query KNN requires k in [1, min(nrow(data), %d)].",
      kMaxNativeCudaK
    );
  }
  if (!std::isfinite(target_recall) ||
      target_recall < 0.8 || target_recall > 1.0) {
    Rcpp::stop("`target_recall` must be between 0.8 and 1.");
  }

  std::string resolved = method;
  std::transform(
    resolved.begin(), resolved.end(), resolved.begin(),
    [](unsigned char value) {
      return static_cast<char>(std::tolower(value));
    }
  );
  if (resolved == "auto") {
    resolved = reference.nrow < kExactRowThreshold ? "exact" : "ivf";
  }
  if (resolved == "flat" || resolved == "bruteforce") resolved = "exact";
  if (resolved != "exact" && resolved != "ivf") {
    Rcpp::stop(
      "Native CUDA query KNN supports only `exact`, `ivf`, or `auto`."
    );
  }
  const bool exact = resolved == "exact";
  const int pilot_n = exact ? 0 : std::min(queries.nrow, 256);
  std::vector<float> pilot_values(
    static_cast<std::size_t>(pilot_n) * queries.ncol
  );
  for (int row = 0; row < pilot_n; ++row) {
    const int source = pilot_n == 1 ? 0 : static_cast<int>(
      (static_cast<int64_t>(row) * (queries.nrow - 1)) / (pilot_n - 1)
    );
    std::copy_n(
      queries.values.data() + static_cast<std::size_t>(source) * queries.ncol,
      queries.ncol,
      pilot_values.data() + static_cast<std::size_t>(row) * queries.ncol
    );
  }
  CuvsResources resources;

  const std::size_t reference_items =
    static_cast<std::size_t>(reference.nrow) * reference.ncol;
  const std::size_t query_items =
    static_cast<std::size_t>(queries.nrow) * queries.ncol;
  const std::size_t final_items =
    static_cast<std::size_t>(queries.nrow) * k;
  CudaBuffer reference_device(reference_items * sizeof(float));
  CudaBuffer query_device(query_items * sizeof(float));
  CudaBuffer output_indices(final_items * sizeof(int));
  CudaBuffer output_distances(final_items * sizeof(float));
  cuda_check(
    cudaMemcpy(
      reference_device.get(), reference.values.data(),
      reference_items * sizeof(float), cudaMemcpyHostToDevice
    ),
    "cudaMemcpy(native CUDA query KNN reference H2D)"
  );
  cuda_check(
    cudaMemcpy(
      query_device.get(), queries.values.data(),
      query_items * sizeof(float), cudaMemcpyHostToDevice
    ),
    "cudaMemcpy(native CUDA query KNN query H2D)"
  );
  std::vector<float>().swap(reference.values);
  std::vector<float>().swap(queries.values);

  int64_t reference_shape[2] = {reference.nrow, reference.ncol};
  int64_t query_shape[2] = {queries.nrow, queries.ncol};
  DLManagedTensor reference_tensor = make_tensor(
    reference_device.get(), reference_shape, 2, kDLCUDA, kDLFloat, 32
  );
  DLManagedTensor query_tensor = make_tensor(
    query_device.get(), query_shape, 2, kDLCUDA, kDLFloat, 32
  );
  const auto distance = L2Expanded;
  const int distance_conversion = distance_mode(parsed_metric);
  IvfTuning tuning = tune_ivf(
    reference.nrow, reference.ncol, k, target_recall
  );
  int search_batch_size = queries.nrow;
  int tuning_attempts = 0;
  double measured_pilot_recall = exact ? 1.0 : 0.0;

  auto finalize_batch = [&](const CudaBuffer& raw_indices,
                            const CudaBuffer& raw_distances,
                            int batch_n,
                            int row_offset) {
    const int status = fastembedr_cuda_finalize_cuvs_query_knn(
      static_cast<const int64_t*>(raw_indices.get()),
      static_cast<const float*>(raw_distances.get()),
      static_cast<int*>(output_indices.get()),
      static_cast<float*>(output_distances.get()),
      batch_n, k, distance_conversion, row_offset, queries.nrow,
      reference.nrow
    );
    if (status != 0) {
      const char* detail = fastembedr_cuda_embedding_last_error();
      Rcpp::stop(
        "Native CUDA query KNN postprocessing failed: %s",
        detail == nullptr ? "unknown CUDA error" : detail
      );
    }
  };

  if (exact) {
    CudaBuffer raw_indices(final_items * sizeof(int64_t));
    CudaBuffer raw_distances(final_items * sizeof(float));
    int64_t output_shape[2] = {queries.nrow, k};
    DLManagedTensor neighbors_tensor = make_tensor(
      raw_indices.get(), output_shape, 2, kDLCUDA, kDLInt, 64
    );
    DLManagedTensor distances_tensor = make_tensor(
      raw_distances.get(), output_shape, 2, kDLCUDA, kDLFloat, 32
    );
    BruteForceIndex index;
    cuvs_check(
      cuvsBruteForceBuild(
        resources.get(), &reference_tensor, distance, 0.0f, index.get()
      ),
      "cuvsBruteForceBuild(query reference)"
    );
    cuvs_check(
      cuvsBruteForceSearch(
        resources.get(), index.get(), &query_tensor,
        &neighbors_tensor, &distances_tensor, no_filter()
      ),
      "cuvsBruteForceSearch(query)"
    );
    cuvs_check(
      cuvsStreamSync(resources.get()),
      "cuvsStreamSync(brute-force query search)"
    );
    finalize_batch(raw_indices, raw_distances, queries.nrow, 0);
  } else {
    IvfFlatIndexParams index_params;
    index_params.get()->metric = distance;
    index_params.get()->add_data_on_build = true;
    index_params.get()->n_lists = static_cast<uint32_t>(tuning.nlist);
    index_params.get()->kmeans_n_iters = 20;
    index_params.get()->kmeans_trainset_fraction = 1.0;
    index_params.get()->adaptive_centers = false;
    index_params.get()->conservative_memory_allocation = true;
    IvfFlatIndex index;
    cuvs_check(
      cuvsIvfFlatBuild(
        resources.get(), index_params.get(), &reference_tensor, index.get()
      ),
      "cuvsIvfFlatBuild(query reference)"
    );

    CudaBuffer pilot_device(pilot_values.size() * sizeof(float));
    cuda_check(
      cudaMemcpy(
        pilot_device.get(), pilot_values.data(),
        pilot_values.size() * sizeof(float), cudaMemcpyHostToDevice
      ),
      "cudaMemcpy(query KNN pilot H2D)"
    );
    std::vector<float>().swap(pilot_values);
    const std::size_t pilot_items =
      static_cast<std::size_t>(pilot_n) * k;
    CudaBuffer pilot_exact_indices(pilot_items * sizeof(int64_t));
    CudaBuffer pilot_exact_distances(pilot_items * sizeof(float));
    CudaBuffer pilot_ivf_indices(pilot_items * sizeof(int64_t));
    CudaBuffer pilot_ivf_distances(pilot_items * sizeof(float));
    int64_t pilot_query_shape[2] = {pilot_n, queries.ncol};
    int64_t pilot_output_shape[2] = {pilot_n, k};
    DLManagedTensor pilot_query_tensor = make_tensor(
      pilot_device.get(), pilot_query_shape, 2, kDLCUDA, kDLFloat, 32
    );
    DLManagedTensor pilot_exact_indices_tensor = make_tensor(
      pilot_exact_indices.get(), pilot_output_shape, 2, kDLCUDA, kDLInt, 64
    );
    DLManagedTensor pilot_exact_distances_tensor = make_tensor(
      pilot_exact_distances.get(), pilot_output_shape, 2, kDLCUDA, kDLFloat, 32
    );
    DLManagedTensor pilot_ivf_indices_tensor = make_tensor(
      pilot_ivf_indices.get(), pilot_output_shape, 2, kDLCUDA, kDLInt, 64
    );
    DLManagedTensor pilot_ivf_distances_tensor = make_tensor(
      pilot_ivf_distances.get(), pilot_output_shape, 2, kDLCUDA, kDLFloat, 32
    );
    BruteForceIndex pilot_reference_index;
    cuvs_check(
      cuvsBruteForceBuild(
        resources.get(), &reference_tensor, distance, 0.0f,
        pilot_reference_index.get()
      ),
      "cuvsBruteForceBuild(query IVF pilot oracle)"
    );
    cuvs_check(
      cuvsBruteForceSearch(
        resources.get(), pilot_reference_index.get(), &pilot_query_tensor,
        &pilot_exact_indices_tensor, &pilot_exact_distances_tensor, no_filter()
      ),
      "cuvsBruteForceSearch(query IVF pilot oracle)"
    );
    cuvs_check(
      cuvsStreamSync(resources.get()),
      "cuvsStreamSync(query IVF pilot oracle)"
    );
    std::vector<int64_t> pilot_reference(pilot_items);
    std::vector<int64_t> pilot_observed(pilot_items);
    cuda_check(
      cudaMemcpy(
        pilot_reference.data(), pilot_exact_indices.get(),
        pilot_items * sizeof(int64_t), cudaMemcpyDeviceToHost
      ),
      "cudaMemcpy(query IVF pilot oracle D2H)"
    );

    IvfFlatSearchParams search_params;
    const double pilot_target = std::min(
      1.0, target_recall + (target_recall >= 0.985 ? 0.005 : 0.002)
    );
    int probe = std::max(1, tuning.nprobe);
    while (true) {
      ++tuning_attempts;
      search_params.get()->n_probes = static_cast<uint32_t>(probe);
      cuvs_check(
        cuvsIvfFlatSearch(
          resources.get(), search_params.get(), index.get(),
          &pilot_query_tensor, &pilot_ivf_indices_tensor,
          &pilot_ivf_distances_tensor, no_filter()
        ),
        "cuvsIvfFlatSearch(query recall pilot)"
      );
      cuvs_check(
        cuvsStreamSync(resources.get()),
        "cuvsStreamSync(query recall pilot)"
      );
      cuda_check(
        cudaMemcpy(
          pilot_observed.data(), pilot_ivf_indices.get(),
          pilot_items * sizeof(int64_t), cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy(query IVF pilot result D2H)"
      );
      measured_pilot_recall = query_pilot_recall(
        pilot_reference, pilot_observed, pilot_n, k
      );
      tuning.nprobe = probe;
      if (measured_pilot_recall >= pilot_target || probe >= tuning.nlist) {
        break;
      }
      probe = std::min(
        tuning.nlist, std::max(probe + 1, probe * 2)
      );
    }
    tuning.rule += "_query_recall_pilot";
    search_params.get()->n_probes = static_cast<uint32_t>(tuning.nprobe);
    search_batch_size = std::min(queries.nrow, 32768);
    const std::size_t batch_items =
      static_cast<std::size_t>(search_batch_size) * k;
    CudaBuffer raw_indices(batch_items * sizeof(int64_t));
    CudaBuffer raw_distances(batch_items * sizeof(float));
    for (int offset = 0; offset < queries.nrow; offset += search_batch_size) {
      const int current = std::min(search_batch_size, queries.nrow - offset);
      int64_t batch_query_shape[2] = {current, queries.ncol};
      int64_t output_shape[2] = {current, k};
      auto* query_pointer = static_cast<float*>(query_device.get()) +
        static_cast<std::size_t>(offset) * queries.ncol;
      DLManagedTensor batch_query_tensor = make_tensor(
        query_pointer, batch_query_shape, 2, kDLCUDA, kDLFloat, 32
      );
      DLManagedTensor neighbors_tensor = make_tensor(
        raw_indices.get(), output_shape, 2, kDLCUDA, kDLInt, 64
      );
      DLManagedTensor distances_tensor = make_tensor(
        raw_distances.get(), output_shape, 2, kDLCUDA, kDLFloat, 32
      );
      cuvs_check(
        cuvsIvfFlatSearch(
          resources.get(), search_params.get(), index.get(),
          &batch_query_tensor, &neighbors_tensor,
          &distances_tensor, no_filter()
        ),
        "cuvsIvfFlatSearch(query)"
      );
      cuvs_check(
        cuvsStreamSync(resources.get()),
        "cuvsStreamSync(IVF query search)"
      );
      finalize_batch(raw_indices, raw_distances, current, offset);
    }
  }
  cuda_check(
    cudaDeviceSynchronize(),
    "cudaDeviceSynchronize(native CUDA query KNN output)"
  );

  auto* handle = new NativeCudaKnnHandle();
  handle->indices = static_cast<int*>(output_indices.release());
  handle->distances = static_cast<float*>(output_distances.release());
  handle->n = queries.nrow;
  handle->k = k;
  cuda_check(cudaGetDevice(&handle->device), "cudaGetDevice(query KNN)");
  Rcpp::List result = make_gpu_result(
    handle, resolved, metric, target_recall, exact, tuning,
    search_batch_size, false, reference.nrow
  );
  result["resident_result_bytes"] = static_cast<double>(final_items) *
    static_cast<double>(sizeof(int) + sizeof(float));
  result["peak_temporary_search_bytes"] =
    static_cast<double>(search_batch_size) * k *
    static_cast<double>(sizeof(int64_t) + sizeof(float));
  result["input_was_float32"] =
    reference.input_float32 && queries.input_float32;
  result["pilot_rows"] = pilot_n;
  result["pilot_recall"] = measured_pilot_recall;
  result["tuning_attempts"] = tuning_attempts;
  result["target_met"] =
    exact || measured_pilot_recall >= target_recall;
  result.attr("class") = Rcpp::CharacterVector::create(
    "fastEmbedR_gpu_knn", "list"
  );
  if (keep_gpu) return result;
  return native_cuda_knn_to_host_impl(result);
}

Rcpp::List native_cuda_knn_to_host_impl(SEXP knn) {
  NativeCudaKnnHandle* handle = native_cuda_handle(knn);
  cuda_check(cudaSetDevice(handle->device), "cudaSetDevice(native CUDA KNN)");
  Rcpp::IntegerMatrix indices(handle->n, handle->k);
  Rcpp::NumericMatrix distances(handle->n, handle->k);
  const std::size_t items = static_cast<std::size_t>(handle->n) * handle->k;
  cuda_check(
    cudaMemcpy(indices.begin(), handle->indices, items * sizeof(int),
               cudaMemcpyDeviceToHost),
    "cudaMemcpy(native CUDA KNN indices D2H)"
  );
  std::vector<float> host_distances(items);
  cuda_check(
    cudaMemcpy(host_distances.data(), handle->distances, items * sizeof(float),
               cudaMemcpyDeviceToHost),
    "cudaMemcpy(native CUDA KNN distances D2H)"
  );
  for (std::size_t i = 0; i < items; ++i) {
    distances.begin()[i] = static_cast<double>(host_distances[i]);
  }

  Rcpp::List source(knn);
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("backend") = source["backend_used"],
    Rcpp::Named("method") = source["method"],
    Rcpp::Named("metric") = source["metric"],
    Rcpp::Named("exact") = source["exact"],
    Rcpp::Named("exclude_self") = source["exclude_self"]
  );
  out.attr("backend") = source["backend_used"];
  out.attr("method") = source["method"];
  out.attr("metric") = source["metric"];
  out.attr("exclude_self") = source["exclude_self"];
  out.attr("gpu_resident_source") = true;
  return out;
}
