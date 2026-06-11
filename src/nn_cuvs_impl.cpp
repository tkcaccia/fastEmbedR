#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuvs/core/c_api.h>
#include <cuvs/distance/distance.h>
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/neighbors/nn_descent.h>
#include <dlpack/dlpack.h>

using Rcpp::IntegerMatrix;
using Rcpp::List;
using Rcpp::NumericMatrix;

namespace {

void cuvs_check(const cuvsError_t status, const char* context) {
  if (status == CUVS_SUCCESS) return;
  const char* detail = cuvsGetLastErrorText();
  if (detail != nullptr && detail[0] != '\0') {
    Rcpp::stop("%s failed: %s", context, detail);
  }
  Rcpp::stop("%s failed.", context);
}

void cuda_check(const cudaError_t status, const char* context) {
  if (status == cudaSuccess) return;
  Rcpp::stop("%s failed: %s", context, cudaGetErrorString(status));
}

void validate_inputs(const NumericMatrix& data,
                     const NumericMatrix& points,
                     const int k,
                     const bool exclude_self) {
  if (data.nrow() < 1 || points.nrow() < 1) {
    Rcpp::stop("data and points must have at least one row");
  }
  if (data.ncol() != points.ncol()) {
    Rcpp::stop("data and points must have the same number of columns");
  }
  if (data.ncol() < 1) {
    Rcpp::stop("data and points must have at least one column");
  }
  if (k < 1 || k > data.nrow()) {
    Rcpp::stop("k must be in [1, nrow(data)]");
  }
  if (exclude_self && data.nrow() != points.nrow()) {
    Rcpp::stop("self-neighbor exclusion requires points to be data");
  }
  if (data.nrow() > std::numeric_limits<int>::max() ||
      points.nrow() > std::numeric_limits<int>::max() ||
      data.ncol() > std::numeric_limits<int>::max()) {
    Rcpp::stop("cuVS backend currently supports dimensions that fit in int");
  }
}

void copy_row_major_float(const NumericMatrix& src, std::vector<float>& dest) {
  const int nrow = src.nrow();
  const int ncol = src.ncol();
  dest.assign(static_cast<std::size_t>(nrow) * ncol, 0.0f);
  for (int c = 0; c < ncol; ++c) {
    for (int r = 0; r < nrow; ++r) {
      const double value = src(r, c);
      if (!std::isfinite(value)) {
        Rcpp::stop("cuVS backend requires finite numeric input");
      }
      dest[static_cast<std::size_t>(r) * ncol + c] =
        static_cast<float>(value);
    }
  }
}

bool same_matrix_storage(const NumericMatrix& data,
                         const NumericMatrix& points) {
  return data.nrow() == points.nrow() &&
    data.ncol() == points.ncol() &&
    data.begin() == points.begin();
}

DLManagedTensor make_tensor(void* data,
                            int64_t* shape,
                            const int ndim,
                            const DLDeviceType device_type,
                            const uint8_t code,
                            const uint8_t bits) {
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

class CuvsResources {
 public:
  CuvsResources() {
    cuvs_check(cuvsResourcesCreate(&res_), "cuvsResourcesCreate");
  }

  ~CuvsResources() {
    if (res_ != 0) {
      cuvsResourcesDestroy(res_);
    }
  }

  cuvsResources_t get() const { return res_; }

  CuvsResources(const CuvsResources&) = delete;
  CuvsResources& operator=(const CuvsResources&) = delete;

 private:
  cuvsResources_t res_ = 0;
};

class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  DeviceBuffer(cuvsResources_t res, const std::size_t bytes) {
    reset(res, bytes);
  }

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cuvsRMMFree(res_, ptr_, bytes_);
    }
  }

  void reset(cuvsResources_t res, const std::size_t bytes) {
    if (ptr_ != nullptr) {
      cuvsRMMFree(res_, ptr_, bytes_);
      ptr_ = nullptr;
    }
    res_ = res;
    bytes_ = bytes;
    if (bytes_ > 0) {
      cuvs_check(cuvsRMMAlloc(res_, &ptr_, bytes_), "cuvsRMMAlloc");
    }
  }

  void* get() const { return ptr_; }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

 private:
  cuvsResources_t res_ = 0;
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
};

class BruteForceIndex {
 public:
  BruteForceIndex() {
    cuvs_check(cuvsBruteForceIndexCreate(&index_), "cuvsBruteForceIndexCreate");
  }
  ~BruteForceIndex() {
    if (index_ != nullptr) {
      cuvsBruteForceIndexDestroy(index_);
    }
  }
  cuvsBruteForceIndex_t get() const { return index_; }
  BruteForceIndex(const BruteForceIndex&) = delete;
  BruteForceIndex& operator=(const BruteForceIndex&) = delete;

 private:
  cuvsBruteForceIndex_t index_ = nullptr;
};

class CagraIndexParams {
 public:
  CagraIndexParams() {
    cuvs_check(cuvsCagraIndexParamsCreate(&params_), "cuvsCagraIndexParamsCreate");
  }
  ~CagraIndexParams() {
    if (params_ != nullptr) {
      cuvsCagraIndexParamsDestroy(params_);
    }
  }
  cuvsCagraIndexParams_t get() const { return params_; }
  CagraIndexParams(const CagraIndexParams&) = delete;
  CagraIndexParams& operator=(const CagraIndexParams&) = delete;

 private:
  cuvsCagraIndexParams_t params_ = nullptr;
};

class CagraSearchParams {
 public:
  CagraSearchParams() {
    cuvs_check(cuvsCagraSearchParamsCreate(&params_), "cuvsCagraSearchParamsCreate");
  }
  ~CagraSearchParams() {
    if (params_ != nullptr) {
      cuvsCagraSearchParamsDestroy(params_);
    }
  }
  cuvsCagraSearchParams_t get() const { return params_; }
  CagraSearchParams(const CagraSearchParams&) = delete;
  CagraSearchParams& operator=(const CagraSearchParams&) = delete;

 private:
  cuvsCagraSearchParams_t params_ = nullptr;
};

class CagraIndex {
 public:
  CagraIndex() {
    cuvs_check(cuvsCagraIndexCreate(&index_), "cuvsCagraIndexCreate");
  }
  ~CagraIndex() {
    if (index_ != nullptr) {
      cuvsCagraIndexDestroy(index_);
    }
  }
  cuvsCagraIndex_t get() const { return index_; }
  CagraIndex(const CagraIndex&) = delete;
  CagraIndex& operator=(const CagraIndex&) = delete;

 private:
  cuvsCagraIndex_t index_ = nullptr;
};

class NNDescentParams {
 public:
  NNDescentParams() {
    cuvs_check(
      cuvsNNDescentIndexParamsCreate(&params_),
      "cuvsNNDescentIndexParamsCreate"
    );
  }
  ~NNDescentParams() {
    if (params_ != nullptr) {
      cuvsNNDescentIndexParamsDestroy(params_);
    }
  }
  cuvsNNDescentIndexParams_t get() const { return params_; }
  NNDescentParams(const NNDescentParams&) = delete;
  NNDescentParams& operator=(const NNDescentParams&) = delete;

 private:
  cuvsNNDescentIndexParams_t params_ = nullptr;
};

class NNDescentIndex {
 public:
  NNDescentIndex() {
    cuvs_check(cuvsNNDescentIndexCreate(&index_), "cuvsNNDescentIndexCreate");
  }
  ~NNDescentIndex() {
    if (index_ != nullptr) {
      cuvsNNDescentIndexDestroy(index_);
    }
  }
  cuvsNNDescentIndex_t get() const { return index_; }
  NNDescentIndex(const NNDescentIndex&) = delete;
  NNDescentIndex& operator=(const NNDescentIndex&) = delete;

 private:
  cuvsNNDescentIndex_t index_ = nullptr;
};

cuvsFilter no_filter() {
  cuvsFilter filter;
  filter.type = NO_FILTER;
  filter.addr = static_cast<uintptr_t>(0);
  return filter;
}

List format_uint32_result(const std::vector<uint32_t>& labels,
                          const std::vector<float>& distances,
                          const int n_points,
                          const int search_k,
                          const int out_k,
                          const bool self_query,
                          const bool exclude_self,
                          const std::string& index_type,
                          const bool exact,
                          const bool already_sqrt = false) {
  IntegerMatrix indices(n_points, out_k);
  NumericMatrix dists(n_points, out_k);
  int* indices_ptr = indices.begin();
  double* dists_ptr = dists.begin();

  for (int i = 0; i < n_points; ++i) {
    int written = 0;
    for (int j = 0; j < search_k && written < out_k; ++j) {
      const uint32_t label = labels[static_cast<std::size_t>(i) * search_k + j];
      if (exclude_self && self_query && label == static_cast<uint32_t>(i)) {
        continue;
      }
      indices_ptr[static_cast<std::size_t>(written) * n_points + i] =
        static_cast<int>(label) + 1;
      const float raw = distances[static_cast<std::size_t>(i) * search_k + j];
      dists_ptr[static_cast<std::size_t>(written) * n_points + i] =
        already_sqrt ? static_cast<double>(raw) :
        std::sqrt(std::max(static_cast<double>(raw), 0.0));
      ++written;
    }
    if (written < out_k) {
      Rcpp::stop("cuVS returned fewer neighbors than requested");
    }
  }

  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = dists,
    Rcpp::Named("index_type") = index_type,
    Rcpp::Named("exact") = exact
  );
}

std::string json_escape(const std::string& text) {
  std::ostringstream out;
  for (char ch : text) {
    switch (ch) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << ch; break;
    }
  }
  return out.str();
}

} // namespace

bool cuvs_is_available_impl() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string cuvs_info_json_impl() {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    return std::string("{\"available\":false,\"library\":\"cuvs\",\"reason\":\"") +
      json_escape(cudaGetErrorString(status)) + "\"}";
  }
  if (count < 1) {
    return "{\"available\":false,\"library\":\"cuvs\",\"reason\":\"no_cuda_device\"}";
  }

  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);
  std::ostringstream out;
  out << "{\"available\":true,\"library\":\"cuvs\",\"interface\":\"c_api\","
      << "\"device\":\"" << json_escape(prop.name) << "\","
      << "\"device_count\":" << count << ","
      << "\"compute_capability\":\"" << prop.major << "." << prop.minor << "\","
      << "\"total_memory\":" << static_cast<unsigned long long>(prop.totalGlobalMem)
      << "}";
  return out.str();
}

List cuvs_bruteforce_knn_impl(NumericMatrix data,
                              NumericMatrix points,
                              int k,
                              bool exclude_self) {
  validate_inputs(data, points, k, exclude_self);
  const bool self_query = exclude_self || same_matrix_storage(data, points);
  const int n_data = data.nrow();
  const int n_points = points.nrow();
  const int n_features = data.ncol();
  const int search_k = exclude_self ? std::min(n_data, k + 1) : k;

  std::vector<float> xb;
  std::vector<float> xq;
  copy_row_major_float(data, xb);
  if (!same_matrix_storage(data, points)) {
    copy_row_major_float(points, xq);
  }

  CuvsResources res;
  const std::size_t data_bytes = xb.size() * sizeof(float);
  DeviceBuffer dataset_d(res.get(), data_bytes);
  cuda_check(
    cudaMemcpy(dataset_d.get(), xb.data(), data_bytes, cudaMemcpyHostToDevice),
    "cudaMemcpy(dataset)"
  );

  DeviceBuffer query_d;
  void* query_ptr = dataset_d.get();
  if (!same_matrix_storage(data, points)) {
    const std::size_t query_bytes = xq.size() * sizeof(float);
    query_d.reset(res.get(), query_bytes);
    cuda_check(
      cudaMemcpy(query_d.get(), xq.data(), query_bytes, cudaMemcpyHostToDevice),
      "cudaMemcpy(queries)"
    );
    query_ptr = query_d.get();
  }

  DeviceBuffer neighbors_d(
    res.get(),
    static_cast<std::size_t>(n_points) * search_k * sizeof(uint32_t)
  );
  DeviceBuffer distances_d(
    res.get(),
    static_cast<std::size_t>(n_points) * search_k * sizeof(float)
  );

  int64_t dataset_shape[2] = {n_data, n_features};
  int64_t query_shape[2] = {n_points, n_features};
  int64_t output_shape[2] = {n_points, search_k};
  DLManagedTensor dataset_tensor = make_tensor(
    dataset_d.get(), dataset_shape, 2, kDLCUDA, kDLFloat, 32
  );
  DLManagedTensor query_tensor = make_tensor(
    query_ptr, query_shape, 2, kDLCUDA, kDLFloat, 32
  );
  DLManagedTensor neighbors_tensor = make_tensor(
    neighbors_d.get(), output_shape, 2, kDLCUDA, kDLUInt, 32
  );
  DLManagedTensor distances_tensor = make_tensor(
    distances_d.get(), output_shape, 2, kDLCUDA, kDLFloat, 32
  );

  BruteForceIndex index;
  cuvs_check(
    cuvsBruteForceBuild(res.get(), &dataset_tensor, L2Expanded, 0.0f, index.get()),
    "cuvsBruteForceBuild"
  );
  cuvs_check(
    cuvsBruteForceSearch(
      res.get(),
      index.get(),
      &query_tensor,
      &neighbors_tensor,
      &distances_tensor,
      no_filter()
    ),
    "cuvsBruteForceSearch"
  );
  cuvs_check(cuvsStreamSync(res.get()), "cuvsStreamSync");

  std::vector<uint32_t> labels(static_cast<std::size_t>(n_points) * search_k);
  std::vector<float> distances(static_cast<std::size_t>(n_points) * search_k);
  cuda_check(
    cudaMemcpy(labels.data(), neighbors_d.get(), labels.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost),
    "cudaMemcpy(neighbors)"
  );
  cuda_check(
    cudaMemcpy(distances.data(), distances_d.get(), distances.size() * sizeof(float), cudaMemcpyDeviceToHost),
    "cudaMemcpy(distances)"
  );

  return format_uint32_result(
    labels,
    distances,
    n_points,
    search_k,
    k,
    self_query,
    exclude_self,
    "cuVS_BruteForce",
    true
  );
}

List cuvs_cagra_knn_impl(NumericMatrix data,
                         NumericMatrix points,
                         int k,
                         bool exclude_self,
                         int graph_degree,
                         int intermediate_graph_degree,
                         int search_width,
                         int itopk_size) {
  validate_inputs(data, points, k, exclude_self);
  const bool self_query = exclude_self || same_matrix_storage(data, points);
  const int n_data = data.nrow();
  const int n_points = points.nrow();
  const int n_features = data.ncol();
  const int search_k = exclude_self ? std::min(n_data, k + 1) : k;

  graph_degree = std::max(search_k, graph_degree);
  graph_degree = std::min(graph_degree, std::max(1, n_data - 1));
  intermediate_graph_degree = std::max(
    intermediate_graph_degree,
    std::max(graph_degree, graph_degree * 2)
  );
  intermediate_graph_degree = std::min(
    intermediate_graph_degree,
    std::max(1, n_data - 1)
  );

  std::vector<float> xb;
  std::vector<float> xq;
  copy_row_major_float(data, xb);
  if (!same_matrix_storage(data, points)) {
    copy_row_major_float(points, xq);
  }

  CuvsResources res;
  const std::size_t data_bytes = xb.size() * sizeof(float);
  DeviceBuffer dataset_d(res.get(), data_bytes);
  cuda_check(
    cudaMemcpy(dataset_d.get(), xb.data(), data_bytes, cudaMemcpyHostToDevice),
    "cudaMemcpy(dataset)"
  );

  DeviceBuffer query_d;
  void* query_ptr = dataset_d.get();
  if (!same_matrix_storage(data, points)) {
    const std::size_t query_bytes = xq.size() * sizeof(float);
    query_d.reset(res.get(), query_bytes);
    cuda_check(
      cudaMemcpy(query_d.get(), xq.data(), query_bytes, cudaMemcpyHostToDevice),
      "cudaMemcpy(queries)"
    );
    query_ptr = query_d.get();
  }

  int64_t dataset_shape[2] = {n_data, n_features};
  int64_t query_shape[2] = {n_points, n_features};
  int64_t output_shape[2] = {n_points, search_k};
  DLManagedTensor dataset_tensor = make_tensor(
    dataset_d.get(), dataset_shape, 2, kDLCUDA, kDLFloat, 32
  );

  CagraIndexParams index_params;
  index_params.get()->metric = L2Expanded;
  index_params.get()->graph_degree = static_cast<std::size_t>(graph_degree);
  index_params.get()->intermediate_graph_degree =
    static_cast<std::size_t>(intermediate_graph_degree);

  CagraIndex index;
  cuvs_check(
    cuvsCagraBuild(res.get(), index_params.get(), &dataset_tensor, index.get()),
    "cuvsCagraBuild"
  );

  DeviceBuffer neighbors_d(
    res.get(),
    static_cast<std::size_t>(n_points) * search_k * sizeof(uint32_t)
  );
  DeviceBuffer distances_d(
    res.get(),
    static_cast<std::size_t>(n_points) * search_k * sizeof(float)
  );
  DLManagedTensor query_tensor = make_tensor(
    query_ptr, query_shape, 2, kDLCUDA, kDLFloat, 32
  );
  DLManagedTensor neighbors_tensor = make_tensor(
    neighbors_d.get(), output_shape, 2, kDLCUDA, kDLUInt, 32
  );
  DLManagedTensor distances_tensor = make_tensor(
    distances_d.get(), output_shape, 2, kDLCUDA, kDLFloat, 32
  );

  CagraSearchParams search_params;
  if (itopk_size > 0) {
    search_params.get()->itopk_size = static_cast<std::size_t>(itopk_size);
  }
  if (search_width > 0) {
    search_params.get()->search_width = static_cast<std::size_t>(search_width);
  }

  cuvs_check(
    cuvsCagraSearch(
      res.get(),
      search_params.get(),
      index.get(),
      &query_tensor,
      &neighbors_tensor,
      &distances_tensor,
      no_filter()
    ),
    "cuvsCagraSearch"
  );
  cuvs_check(cuvsStreamSync(res.get()), "cuvsStreamSync");

  std::vector<uint32_t> labels(static_cast<std::size_t>(n_points) * search_k);
  std::vector<float> distances(static_cast<std::size_t>(n_points) * search_k);
  cuda_check(
    cudaMemcpy(labels.data(), neighbors_d.get(), labels.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost),
    "cudaMemcpy(neighbors)"
  );
  cuda_check(
    cudaMemcpy(distances.data(), distances_d.get(), distances.size() * sizeof(float), cudaMemcpyDeviceToHost),
    "cudaMemcpy(distances)"
  );

  List out = format_uint32_result(
    labels,
    distances,
    n_points,
    search_k,
    k,
    self_query,
    exclude_self,
    "cuVS_CAGRA",
    false
  );
  out["graph_degree"] = graph_degree;
  out["intermediate_graph_degree"] = intermediate_graph_degree;
  out["search_width"] = search_width;
  out["itopk_size"] = itopk_size;
  return out;
}

List cuvs_nndescent_self_knn_impl(NumericMatrix data,
                                  int k,
                                  int graph_degree,
                                  int intermediate_graph_degree,
                                  int max_iterations) {
  if (data.nrow() < 2) Rcpp::stop("data must have at least two rows");
  if (data.ncol() < 1) Rcpp::stop("data must have at least one column");
  if (k < 1 || k >= data.nrow()) {
    Rcpp::stop("k must be in [1, nrow(data) - 1]");
  }

  const int n_data = data.nrow();
  const int n_features = data.ncol();
  graph_degree = std::max(k + 1, graph_degree);
  graph_degree = std::min(graph_degree, n_data - 1);
  intermediate_graph_degree = std::max(
    intermediate_graph_degree,
    std::max(graph_degree, graph_degree * 2)
  );
  intermediate_graph_degree = std::min(intermediate_graph_degree, n_data - 1);
  max_iterations = std::max(1, max_iterations);

  std::vector<float> xb;
  copy_row_major_float(data, xb);
  std::vector<uint32_t> graph(
    static_cast<std::size_t>(n_data) * graph_degree,
    0
  );
  std::vector<float> distances(
    static_cast<std::size_t>(n_data) * graph_degree,
    0.0f
  );

  CuvsResources res;
  int64_t dataset_shape[2] = {n_data, n_features};
  int64_t graph_shape[2] = {n_data, graph_degree};
  DLManagedTensor dataset_tensor = make_tensor(
    xb.data(), dataset_shape, 2, kDLCPU, kDLFloat, 32
  );
  DLManagedTensor graph_tensor = make_tensor(
    graph.data(), graph_shape, 2, kDLCPU, kDLUInt, 32
  );
  DLManagedTensor distances_tensor = make_tensor(
    distances.data(), graph_shape, 2, kDLCPU, kDLFloat, 32
  );

  NNDescentParams params;
  params.get()->metric = L2Expanded;
  params.get()->graph_degree = static_cast<std::size_t>(graph_degree);
  params.get()->intermediate_graph_degree =
    static_cast<std::size_t>(intermediate_graph_degree);
  params.get()->max_iterations = static_cast<std::size_t>(max_iterations);
  params.get()->return_distances = true;

  NNDescentIndex index;
  cuvs_check(
    cuvsNNDescentBuild(
      res.get(),
      params.get(),
      &dataset_tensor,
      &graph_tensor,
      index.get()
    ),
    "cuvsNNDescentBuild"
  );
  cuvs_check(
    cuvsNNDescentIndexGetGraph(res.get(), index.get(), &graph_tensor),
    "cuvsNNDescentIndexGetGraph"
  );
  cuvs_check(
    cuvsNNDescentIndexGetDistances(res.get(), index.get(), &distances_tensor),
    "cuvsNNDescentIndexGetDistances"
  );
  cuvs_check(cuvsStreamSync(res.get()), "cuvsStreamSync");

  List out = format_uint32_result(
    graph,
    distances,
    n_data,
    graph_degree,
    k,
    true,
    true,
    "cuVS_NNDescent",
    false
  );
  out["graph_degree"] = graph_degree;
  out["intermediate_graph_degree"] = intermediate_graph_degree;
  out["max_iterations"] = max_iterations;
  return out;
}
