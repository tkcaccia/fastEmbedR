#include <Rcpp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "massive_pca_cuda.h"
#include "massive_graph_attraction.h"
#include "massive_umap_cuda.h"
#include "massive_umap_metal.h"
#include "massive_tsne_cuda.h"
#include "louvain_objective.h"
#include "native_knn_common.h"
#include "tsne_affinity_common.h"
#include "umap_membership.h"
#include "umap_optimizer_common.h"

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

int massive_cuda_device_count_cpp();
int massive_cuda_select_cpp(int device);
List massive_cuda_memory_cpp();

namespace {

std::uint64_t whole_number(double value, const char* name) {
  if (!std::isfinite(value) || value < 0 || std::floor(value) != value ||
      value > 9007199254740991.0) {
    Rcpp::stop(std::string(name) + " must be an exact non-negative integer.");
  }
  return static_cast<std::uint64_t>(value);
}

std::uint64_t checked_product(std::uint64_t left,
                              std::uint64_t right) {
  if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() /
      right) {
    Rcpp::stop("Massive matrix byte count overflows uint64.");
  }
  return left * right;
}

void require_little_endian() {
  const std::uint16_t value = 1;
  if (*reinterpret_cast<const unsigned char*>(&value) != 1) {
    Rcpp::stop("Experimental float32 files require a little-endian host.");
  }
}

std::uint32_t read_u32(const unsigned char* bytes) {
  return static_cast<std::uint32_t>(bytes[0]) |
    (static_cast<std::uint32_t>(bytes[1]) << 8) |
    (static_cast<std::uint32_t>(bytes[2]) << 16) |
    (static_cast<std::uint32_t>(bytes[3]) << 24);
}

std::uint64_t read_u64(const unsigned char* bytes) {
  return static_cast<std::uint64_t>(read_u32(bytes)) |
    (static_cast<std::uint64_t>(read_u32(bytes + 4)) << 32);
}

std::uint64_t graph_row_offset(std::uint64_t row, int k) {
  return checked_product(checked_product(row - 1, k), 4);
}

struct FileInfo {
  std::uint64_t rows;
  std::uint64_t columns;
  std::uint64_t bytes;
  std::uint64_t offset;
};

std::string file_stamp(const std::string& path) {
  std::error_code error;
  const auto modified = std::filesystem::last_write_time(path, error);
  if (error) Rcpp::stop("Cannot inspect massive input modification time.");
  return std::to_string(static_cast<long long>(
    modified.time_since_epoch().count()));
}

FileInfo inspect_file(const std::string& path,
                      const std::string& format,
                      double requested_rows,
                      double requested_columns) {
  require_little_endian();
  if (format != "f32" && format != "fbin") {
    Rcpp::stop("Massive file format must be 'f32' or 'fbin'.");
  }
  std::error_code error;
  const auto size = std::filesystem::file_size(path, error);
  if (error) Rcpp::stop("Cannot inspect massive input: " + error.message());
  FileInfo info{0, 0, static_cast<std::uint64_t>(size), 0};
  if (format == "fbin") {
    if (size < 8) Rcpp::stop("The .fbin header is incomplete.");
    std::ifstream stream(path, std::ios::binary);
    unsigned char header[8]{};
    stream.read(reinterpret_cast<char*>(header), 8);
    if (!stream) Rcpp::stop("Cannot read the .fbin header.");
    info.rows = read_u32(header);
    info.columns = read_u32(header + 4);
    info.offset = 8;
  } else {
    info.rows = whole_number(requested_rows, "nrow");
    info.columns = whole_number(requested_columns, "ncol");
  }
  if (info.rows < 1 || info.columns < 1) {
    Rcpp::stop("Massive matrix dimensions must be positive.");
  }
  const auto data_bytes = checked_product(
    checked_product(info.rows, info.columns), 4
  );
  if (data_bytes > std::numeric_limits<std::uint64_t>::max() -
      info.offset || info.bytes != data_bytes + info.offset) {
    Rcpp::stop("Massive input file size does not match its dimensions.");
  }
  return info;
}

class MatrixSource {
 public:
  virtual ~MatrixSource() = default;
  virtual std::uint64_t nrow() const = 0;
  virtual std::uint64_t ncol() const = 0;
  virtual void read_rows(std::uint64_t start,
                         std::uint64_t count,
                         float* output) = 0;
};

class MemoryMatrixSource final : public MatrixSource {
 public:
  explicit MemoryMatrixSource(NumericMatrix matrix) : matrix_(matrix) {}
  std::uint64_t nrow() const override { return matrix_.nrow(); }
  std::uint64_t ncol() const override { return matrix_.ncol(); }

  void read_rows(std::uint64_t start,
                 std::uint64_t count,
                 float* output) override {
    for (std::uint64_t row = 0; row < count; ++row) {
      for (std::uint64_t col = 0; col < ncol(); ++col) {
        const double value = matrix_(start + row, col);
        if (!std::isfinite(value) ||
            std::abs(value) > std::numeric_limits<float>::max()) {
          Rcpp::stop("Massive input has non-finite or non-float32 values.");
        }
        output[row * ncol() + col] = static_cast<float>(value);
      }
    }
  }

 private:
  NumericMatrix matrix_;
};

class SyntheticMatrixSource final : public MatrixSource {
 public:
  SyntheticMatrixSource(double rows, double columns)
      : rows_(whole_number(rows, "nrow")),
        columns_(whole_number(columns, "ncol")) {
    if (rows_ < 1 || columns_ < 1) {
      Rcpp::stop("Synthetic dimensions must be positive.");
    }
  }

  std::uint64_t nrow() const override { return rows_; }
  std::uint64_t ncol() const override { return columns_; }

  void read_rows(std::uint64_t start,
                 std::uint64_t count,
                 float* output) override {
    if (start > rows_ || count > rows_ - start) {
      Rcpp::stop("Synthetic read exceeds the input row count.");
    }
    for (std::uint64_t row = 0; row < count; ++row) {
      for (std::uint64_t col = 0; col < columns_; ++col) {
        output[row * columns_ + col] =
          static_cast<float>((start + row) % 10007) / 10007.0f +
          static_cast<float>(col) * 0.01f;
      }
    }
  }

 private:
  std::uint64_t rows_;
  std::uint64_t columns_;
};

class FileMatrixSource final : public MatrixSource {
 public:
  FileMatrixSource(const std::string& path,
                   const std::string& format,
                   const std::string& access,
                   double rows,
                   double columns,
                   std::string stamp = "")
      : path_(path), stamp_(stamp.empty() ? file_stamp(path) : stamp),
        info_(inspect_file(path, format, rows, columns)),
        stream_(), mapped_(nullptr), mapped_size_(0) {
    check_identity();
    if (access == "stream") {
      stream_.open(path, std::ios::binary);
      if (!stream_) Rcpp::stop("Cannot open massive input file.");
    } else if (access == "mmap") {
#if defined(__unix__) || defined(__APPLE__)
      if (info_.bytes > std::numeric_limits<std::size_t>::max()) {
        Rcpp::stop("Massive input exceeds the mmap address range.");
      }
      const int descriptor = open(path.c_str(), O_RDONLY);
      if (descriptor < 0) Rcpp::stop("Cannot open massive input for mmap.");
      mapped_size_ = static_cast<std::size_t>(info_.bytes);
      void* address = mmap(nullptr, mapped_size_, PROT_READ,
                           MAP_PRIVATE, descriptor, 0);
      close(descriptor);
      if (address == MAP_FAILED) Rcpp::stop("Cannot mmap massive input.");
      mapped_ = static_cast<const unsigned char*>(address);
#elif defined(_WIN32)
      if (info_.bytes > std::numeric_limits<std::size_t>::max()) {
        Rcpp::stop("Massive input exceeds the mmap address range.");
      }
      const auto wide_path = std::filesystem::u8path(path).wstring();
      HANDLE file = CreateFileW(wide_path.c_str(), GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
      if (file == INVALID_HANDLE_VALUE) {
        Rcpp::stop("Cannot open massive input for mmap.");
      }
      HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY,
        0, 0, nullptr);
      CloseHandle(file);
      if (mapping == nullptr) Rcpp::stop("Cannot map massive input.");
      void* address = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
      CloseHandle(mapping);
      if (address == nullptr) Rcpp::stop("Cannot mmap massive input.");
      mapped_size_ = static_cast<std::size_t>(info_.bytes);
      mapped_ = static_cast<const unsigned char*>(address);
#else
      Rcpp::stop("Experimental mmap input is unavailable on this system.");
#endif
    } else {
      Rcpp::stop("Massive file access must be 'stream' or 'mmap'.");
    }
  }

  ~FileMatrixSource() override {
#if defined(__unix__) || defined(__APPLE__)
    if (mapped_ != nullptr) munmap(
      const_cast<unsigned char*>(mapped_), mapped_size_
    );
#elif defined(_WIN32)
    if (mapped_ != nullptr) UnmapViewOfFile(mapped_);
#endif
  }

  std::uint64_t nrow() const override { return info_.rows; }
  std::uint64_t ncol() const override { return info_.columns; }

  void read_rows(std::uint64_t start,
                 std::uint64_t count,
                 float* output) override {
    check_identity();
    if (start > nrow() || count > nrow() - start) {
      Rcpp::stop("Massive read exceeds the input row count.");
    }
    const auto offset = info_.offset + checked_product(
      checked_product(start, ncol()), 4
    );
    const auto bytes = checked_product(checked_product(count, ncol()), 4);
    if (bytes > std::numeric_limits<std::size_t>::max()) {
      Rcpp::stop("Massive read chunk exceeds addressable memory.");
    }
    if (mapped_ != nullptr) {
      std::memcpy(output, mapped_ + offset, static_cast<std::size_t>(bytes));
    } else {
      if (offset > static_cast<std::uint64_t>(
          std::numeric_limits<std::streamoff>::max()) ||
          bytes > static_cast<std::uint64_t>(
          std::numeric_limits<std::streamsize>::max())) {
        Rcpp::stop("Massive file offset exceeds stream limits.");
      }
      stream_.seekg(static_cast<std::streamoff>(offset));
      stream_.read(reinterpret_cast<char*>(output),
                   static_cast<std::streamsize>(bytes));
      if (!stream_) Rcpp::stop("Massive input read was incomplete.");
    }
    for (std::uint64_t i = 0; i < count * ncol(); ++i) {
      if (!std::isfinite(output[i])) {
        Rcpp::stop("Massive input contains non-finite values.");
      }
    }
    check_identity();
  }

 private:
  void check_identity() const {
    std::error_code error;
    const auto bytes = std::filesystem::file_size(path_, error);
    if (error || bytes != info_.bytes || file_stamp(path_) != stamp_) {
      Rcpp::stop("Massive input changed since descriptor creation.");
    }
  }

  std::string path_;
  std::string stamp_;
  FileInfo info_;
  std::ifstream stream_;
  const unsigned char* mapped_;
  std::size_t mapped_size_;
};

class RowViewSource final : public MatrixSource {
 public:
  RowViewSource(std::unique_ptr<MatrixSource> source,
                double first, double rows)
      : source_(std::move(source)),
        offset_(whole_number(first, "first") - 1),
        rows_(whole_number(rows, "nrow")) {
    if (first < 1 || rows_ < 1 || offset_ >= source_->nrow() ||
        rows_ > source_->nrow() - offset_) {
      Rcpp::stop("Massive row view exceeds its source range.");
    }
  }

  std::uint64_t nrow() const override { return rows_; }
  std::uint64_t ncol() const override { return source_->ncol(); }

  void read_rows(std::uint64_t start, std::uint64_t count,
                 float* output) override {
    if (start > rows_ || count > rows_ - start) {
      Rcpp::stop("Massive row view read exceeds its range.");
    }
    source_->read_rows(offset_ + start, count, output);
  }

 private:
  std::unique_ptr<MatrixSource> source_;
  std::uint64_t offset_;
  std::uint64_t rows_;
};

std::unique_ptr<MatrixSource> make_source(List spec) {
  const std::string format = Rcpp::as<std::string>(spec["format"]);
  if (format == "view") {
    return std::unique_ptr<MatrixSource>(new RowViewSource(
      make_source(Rcpp::as<List>(spec["source"])),
      Rcpp::as<double>(spec["first"]),
      Rcpp::as<double>(spec["nrow"])
    ));
  }
  if (format == "memory") {
    return std::unique_ptr<MatrixSource>(
      new MemoryMatrixSource(Rcpp::as<NumericMatrix>(spec["data"]))
    );
  }
  if (format == "synthetic") {
    return std::unique_ptr<MatrixSource>(new SyntheticMatrixSource(
      Rcpp::as<double>(spec["nrow"]), Rcpp::as<double>(spec["ncol"])
    ));
  }
  return std::unique_ptr<MatrixSource>(new FileMatrixSource(
    Rcpp::as<std::string>(spec["path"]), format,
    Rcpp::as<std::string>(spec["access"]),
    Rcpp::as<double>(spec["nrow"]), Rcpp::as<double>(spec["ncol"]),
    Rcpp::as<std::string>(spec["stamp"])
  ));
}

template <typename Function>
void parallel_rows(std::uint64_t rows, int requested, Function work) {
  const int threads = static_cast<int>(std::min<std::uint64_t>(
    rows, std::max(1, requested)
  ));
  if (threads <= 1) {
    work(0, rows, 0);
    return;
  }
  std::vector<std::thread> workers;
  workers.reserve(threads);
  for (int worker = 0; worker < threads; ++worker) {
    const auto begin = rows * worker / threads;
    const auto end = rows * (worker + 1) / threads;
    workers.emplace_back(work, begin, end, worker);
  }
  for (auto& worker : workers) worker.join();
}

class Float32Sink {
 public:
  explicit Float32Sink(const std::string& path,
                       std::uint64_t committed_bytes = 0,
                       std::uint64_t expected_bytes = 0,
                       bool resume = false)
      : path_(path), partial_(path + ".part") {
    if (std::filesystem::exists(path_) ||
        (!resume && std::filesystem::exists(partial_))) {
      Rcpp::stop("Massive output or its .part file already exists.");
    }
    if (resume) {
      std::error_code error;
      const auto size = std::filesystem::file_size(partial_, error);
      if (error || size < committed_bytes || size > expected_bytes) {
        Rcpp::stop("Massive partial output does not match checkpoint.");
      }
      stream_.open(partial_, std::ios::binary |
                     std::ios::in | std::ios::out);
      stream_.seekp(static_cast<std::streamoff>(committed_bytes));
    } else {
      stream_.open(partial_, std::ios::binary);
    }
    if (!stream_) Rcpp::stop("Cannot open massive output for writing.");
  }

  void append(const float* values, std::uint64_t count) {
    const auto bytes = checked_product(count, 4);
    if (bytes > static_cast<std::uint64_t>(
        std::numeric_limits<std::streamsize>::max())) {
      Rcpp::stop("Massive output chunk exceeds stream limits.");
    }
    stream_.write(reinterpret_cast<const char*>(values),
                  static_cast<std::streamsize>(bytes));
    if (!stream_) Rcpp::stop("Massive output write failed; .part retained.");
  }

  void finish(std::uint64_t expected_bytes) {
    flush();
    stream_.close();
    std::error_code error;
    const auto actual = std::filesystem::file_size(partial_, error);
    if (error || actual != expected_bytes) {
      Rcpp::stop("Massive output size mismatch; .part retained.");
    }
    if (std::filesystem::exists(path_)) {
      Rcpp::stop("Massive output appeared during writing; .part retained.");
    }
    std::filesystem::rename(partial_, path_, error);
    if (error) Rcpp::stop("Massive output rename failed: " + error.message());
  }

  void flush() {
    stream_.flush();
    if (!stream_) Rcpp::stop("Massive output flush failed; .part retained.");
  }

 private:
  std::string path_;
  std::string partial_;
  std::ofstream stream_;
};

class GraphFile {
 public:
  GraphFile(const std::string& path, std::uint64_t expected,
            const std::string& access)
      : mapped_(nullptr), size_(expected) {
    std::error_code error;
    const auto actual = std::filesystem::file_size(path, error);
    if (error || actual != expected) {
      Rcpp::stop("Massive graph file size does not match dimensions.");
    }
    if (access == "stream") {
      stream_.open(path, std::ios::binary);
      if (!stream_) Rcpp::stop("Cannot open massive graph file.");
    } else if (access == "mmap") {
#if defined(__unix__) || defined(__APPLE__)
      if (expected > std::numeric_limits<std::size_t>::max()) {
        Rcpp::stop("Massive graph exceeds the mmap address range.");
      }
      const int fd = open(path.c_str(), O_RDONLY);
      if (fd < 0) Rcpp::stop("Cannot open massive graph for mmap.");
      void* address = mmap(nullptr, static_cast<std::size_t>(expected),
                           PROT_READ, MAP_PRIVATE, fd, 0);
      ::close(fd);
      if (address == MAP_FAILED) Rcpp::stop("Cannot mmap massive graph.");
      mapped_ = static_cast<const unsigned char*>(address);
#elif defined(_WIN32)
      if (expected > std::numeric_limits<std::size_t>::max()) {
        Rcpp::stop("Massive graph exceeds the mmap address range.");
      }
      const auto wide_path = std::filesystem::u8path(path).wstring();
      HANDLE file = CreateFileW(wide_path.c_str(), GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
      if (file == INVALID_HANDLE_VALUE) {
        Rcpp::stop("Cannot open massive graph for mmap.");
      }
      HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY,
        0, 0, nullptr);
      CloseHandle(file);
      if (mapping == nullptr) Rcpp::stop("Cannot map massive graph.");
      void* address = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
      CloseHandle(mapping);
      if (address == nullptr) Rcpp::stop("Cannot mmap massive graph.");
      mapped_ = static_cast<const unsigned char*>(address);
#else
      Rcpp::stop("Massive graph mmap is unavailable on this system.");
#endif
    } else {
      Rcpp::stop("Graph access must be 'stream' or 'mmap'.");
    }
  }

  ~GraphFile() {
    close();
  }

  void close() {
#if defined(__unix__) || defined(__APPLE__)
    if (mapped_) munmap(const_cast<unsigned char*>(mapped_),
                       static_cast<std::size_t>(size_));
#elif defined(_WIN32)
    if (mapped_) UnmapViewOfFile(mapped_);
#endif
    mapped_ = nullptr;
    if (stream_.is_open()) stream_.close();
  }

  const unsigned char* read(std::uint64_t offset,
                            std::uint64_t bytes) {
    if (offset > size_ || bytes > size_ - offset) {
      Rcpp::stop("Massive graph read exceeds file size.");
    }
    if (bytes == 0) {
      static const unsigned char empty = 0;
      return &empty;
    }
    if (mapped_) return mapped_ + offset;
    if (offset > static_cast<std::uint64_t>(
        std::numeric_limits<std::streamoff>::max()) ||
        bytes > static_cast<std::uint64_t>(
        std::numeric_limits<std::streamsize>::max())) {
      Rcpp::stop("Massive graph offset exceeds stream limits.");
    }
    buffer_.resize(static_cast<std::size_t>(bytes));
    stream_.clear();
    stream_.seekg(static_cast<std::streamoff>(offset));
    stream_.read(reinterpret_cast<char*>(buffer_.data()),
                 static_cast<std::streamsize>(bytes));
    if (!stream_) Rcpp::stop("Massive graph read was incomplete.");
    return buffer_.data();
  }

 private:
  std::ifstream stream_;
  std::vector<unsigned char> buffer_;
  const unsigned char* mapped_;
  std::uint64_t size_;
};

class KnnGraphSource {
 public:
  KnnGraphSource(const std::string& indices_path,
                 const std::string& distances_path,
                 double vertices, int k, const std::string& access,
                 const std::string& kind)
      : vertices_(whole_number(vertices, "n_vertices")), k_(k),
        weighted_(kind == "weight"),
        indices_(indices_path, graph_bytes(vertices_, k), access),
        distances_(distances_path, graph_bytes(vertices_, k), access) {
    require_little_endian();
    if (kind != "distance" && kind != "weight") {
      Rcpp::stop("Massive graph value kind must be distance or weight.");
    }
    if (vertices_ < 2 ||
        vertices_ > static_cast<std::uint64_t>(
          std::numeric_limits<std::int32_t>::max()) ||
        k < 1 || k > 65536 ||
        static_cast<std::uint64_t>(k) >= vertices_) {
      Rcpp::stop("Invalid massive graph vertex count or k.");
    }
  }

  std::uint64_t vertices() const { return vertices_; }
  std::uint64_t edges() const { return checked_product(vertices_, k_); }
  int k() const { return k_; }
  void close() { indices_.close(); distances_.close(); }

  template <typename Function>
  void visit(std::uint64_t first, std::uint64_t count,
             Function consume) {
    if (first < 1 || first > vertices_ ||
        count < 1 || count > vertices_ - first + 1) {
      Rcpp::stop("Invalid massive graph row range.");
    }
    const auto items = checked_product(count, k_);
    const auto offset = graph_row_offset(first, k_);
    const auto bytes = checked_product(items, 4);
    const auto* ids = indices_.read(offset, bytes);
    const auto* values = distances_.read(offset, bytes);
    for (std::uint64_t row = 0; row < count; ++row) {
      const auto source = first + row;
      for (int col = 0; col < k_; ++col) {
        const auto pos = row * k_ + col;
        const auto target = read_u32(ids + 4 * pos);
        float value = 0.0f;
        std::memcpy(&value, values + 4 * pos, 4);
        if (target < 1 || target > vertices_ || target == source) {
          Rcpp::stop("Massive graph has invalid or self edges.");
        }
        if (!std::isfinite(value) || value < 0 ||
            (weighted_ && value > 1)) {
          Rcpp::stop("Massive graph has invalid edge values.");
        }
        consume(source, target, value, row, col);
      }
    }
  }

 private:
  static std::uint64_t graph_bytes(std::uint64_t vertices, int k) {
    if (k < 1) Rcpp::stop("Massive graph k must be positive.");
    return checked_product(checked_product(vertices, k), 4);
  }

  std::uint64_t vertices_;
  int k_;
  bool weighted_;
  GraphFile indices_;
  GraphFile distances_;
};

std::uint64_t csr_edge_bytes(const std::string& path) {
  std::error_code error;
  const auto bytes = std::filesystem::file_size(path, error);
  if (error || bytes < 4 || bytes % 4 != 0) {
    Rcpp::stop("CSR index file must contain whole uint32 edges.");
  }
  return static_cast<std::uint64_t>(bytes);
}

class CsrGraphSource {
 public:
  CsrGraphSource(const std::string& offsets_path,
                 const std::string& indices_path,
                 const std::string& weights_path,
                 double vertices, const std::string& access,
                 bool contracted = false)
      : vertices_(whole_number(vertices, "n_vertices")),
        edge_bytes_(csr_edge_bytes(indices_path)),
        contracted_(contracted),
        offsets_(offsets_path, checked_product(vertices_ + 1, 8),
                 access),
        indices_(indices_path, edge_bytes_, access),
        weights_(weights_path, checked_product(edges(),
                 contracted_ ? 8 : 4), access) {
    require_little_endian();
    if (vertices_ < 2 ||
        vertices_ > static_cast<std::uint64_t>(
          std::numeric_limits<std::int32_t>::max())) {
      Rcpp::stop("Invalid CSR graph vertex count.");
    }
    if (offset(0) != 0 || offset(vertices_) != edges()) {
      Rcpp::stop("CSR offsets must span the complete edge files.");
    }
  }

  std::uint64_t vertices() const { return vertices_; }
  std::uint64_t edges() const { return edge_bytes_ / 4; }

  std::uint64_t offset(std::uint64_t row) {
    return read_u64(offsets_.read(checked_product(row, 8), 8));
  }

  std::uint64_t bounded_rows(std::uint64_t first,
                             std::uint64_t limit_rows,
                             std::uint64_t limit_edges) {
    const auto rows = std::min(limit_rows, vertices_ - first + 1);
    const auto* offsets = offsets_.read(
      checked_product(first - 1, 8), checked_product(rows + 1, 8));
    const auto begin = read_u64(offsets);
    if (begin > edges()) Rcpp::stop("CSR graph has invalid offsets.");
    std::uint64_t previous = begin;
    std::uint64_t count = 0;
    for (std::uint64_t row = 1; row <= rows; ++row) {
      const auto current = read_u64(offsets + row * 8);
      if (current < previous || current > edges()) {
        Rcpp::stop("CSR graph has invalid offsets.");
      }
      if (current - begin > limit_edges) break;
      previous = current;
      count = row;
    }
    if (count == 0) Rcpp::stop("CSR graph row exceeds edge budget.");
    return count;
  }

  template <typename Function>
  std::uint64_t visit(std::uint64_t first, std::uint64_t count,
                      Function consume, bool interrupt = true) {
    if (first < 1 || first > vertices_ || count < 1 ||
        count > vertices_ - first + 1) {
      Rcpp::stop("Invalid CSR graph row range.");
    }
    const auto* offsets = offsets_.read(
      checked_product(first - 1, 8), checked_product(count + 1, 8));
    const auto begin = read_u64(offsets);
    const auto end = read_u64(offsets + count * 8);
    if (end < begin || end > edges()) {
      Rcpp::stop("CSR graph has invalid offsets.");
    }
    const auto value_bytes = contracted_ ? 8 : 4;
    std::uint64_t max_degree = 0;
    for (std::uint64_t row = 0; row < count; ++row) {
      const auto row_begin = read_u64(offsets + row * 8);
      const auto row_end = read_u64(offsets + (row + 1) * 8);
      if (row_end < row_begin || row_end > end) {
        Rcpp::stop("CSR graph has invalid offsets.");
      }
      max_degree = std::max(max_degree, row_end - row_begin);
    }
    std::uint64_t row = 0;
    std::uint64_t row_end = read_u64(offsets + 8);
    std::uint32_t previous = 0;
    for (auto block = begin; block < end;) {
      const auto items = std::min<std::uint64_t>(65536, end - block);
      const auto* ids = indices_.read(checked_product(block, 4),
                                      checked_product(items, 4));
      const auto* values = weights_.read(
        checked_product(block, value_bytes),
        checked_product(items, value_bytes));
      for (std::uint64_t local = 0; local < items; ++local) {
        const auto edge = block + local;
        while (edge >= row_end) {
          ++row;
          previous = 0;
          row_end = read_u64(offsets + (row + 1) * 8);
        }
        const auto target = read_u32(ids + local * 4);
        double weight = 0.0;
        if (contracted_) {
          std::memcpy(&weight, values + local * 8, 8);
        } else {
          float value = 0.0f;
          std::memcpy(&value, values + local * 4, 4);
          weight = value;
        }
        if (target <= previous || target > vertices_ ||
            (!contracted_ && target == first + row) ||
            !std::isfinite(weight) || weight < 0.0 ||
            (contracted_ && weight == 0.0) ||
            (!contracted_ && weight > 1.0)) {
          Rcpp::stop("CSR graph has invalid targets or weights.");
        }
        previous = target;
        consume(first + row, target, weight, edge - begin);
      }
      block += items;
      if (block < end && interrupt) Rcpp::checkUserInterrupt();
    }
    return max_degree;
  }

 private:
  std::uint64_t vertices_;
  std::uint64_t edge_bytes_;
  bool contracted_;
  GraphFile offsets_;
  GraphFile indices_;
  GraphFile weights_;
};

#if defined(__unix__) || defined(__APPLE__)
template <typename T>
class MappedLouvainVector {
 public:
  MappedLouvainVector(const std::string& path, std::uint64_t count)
      : fd_(-1), data_(nullptr), bytes_(checked_product(count, sizeof(T))) {
    if (bytes_ > std::numeric_limits<std::size_t>::max() ||
        bytes_ > static_cast<std::uint64_t>(
          std::numeric_limits<off_t>::max())) {
      Rcpp::stop("Louvain state exceeds the mmap address range.");
    }
    fd_ = open(path.c_str(), O_CREAT | O_EXCL | O_RDWR, 0666);
    if (fd_ < 0) Rcpp::stop("Cannot create Louvain state file: " + path);
    if (ftruncate(fd_, static_cast<off_t>(bytes_)) != 0) {
      close(fd_);
      fd_ = -1;
      Rcpp::stop("Cannot size Louvain state file: " + path);
    }
    void* address = mmap(nullptr, static_cast<std::size_t>(bytes_),
                         PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (address == MAP_FAILED) {
      close(fd_);
      fd_ = -1;
      Rcpp::stop("Cannot mmap Louvain state file: " + path);
    }
    data_ = static_cast<T*>(address);
  }

  ~MappedLouvainVector() {
    if (data_) munmap(data_, static_cast<std::size_t>(bytes_));
    if (fd_ >= 0) close(fd_);
  }

  T& operator[](std::uint64_t index) { return data_[index]; }

  void sync() {
    if (msync(data_, static_cast<std::size_t>(bytes_), MS_SYNC) != 0) {
      Rcpp::stop("Cannot flush Louvain membership file.");
    }
  }

 private:
  int fd_;
  T* data_;
  std::uint64_t bytes_;
};

class MassiveLouvainLevel {
 public:
  MassiveLouvainLevel(CsrGraphSource& graph, const std::string& output)
      : graph_(graph), n_(graph.vertices()),
        labels_(output + ".part", n_),
        counts_(output + ".counts.part", n_),
        volumes_(output + ".volumes.part", n_) {}

  void initialize(int chunk_rows, std::uint64_t edge_budget,
                  double resolution, const std::string& initial_path,
                  int initial_count) {
    std::unique_ptr<GraphFile> initial;
    if (!initial_path.empty()) {
      if (initial_count < 1 || initial_count > n_) {
        Rcpp::stop("Invalid initial Louvain community count.");
      }
      initial = std::make_unique<GraphFile>(initial_path,
        checked_product(n_, 4), "mmap");
    }
    for (std::uint64_t row = 0; row < n_; ++row) {
      counts_[row] = 0;
      volumes_[row] = 0.0;
    }
    for (std::uint64_t row = 0; row < n_; ++row) {
      const auto label = initial ? read_u32(initial->read(
        checked_product(row, 4), 4)) :
        static_cast<std::uint32_t>(row + 1);
      if (label < 1 || label > static_cast<std::uint32_t>(
          initial ? initial_count : n_)) {
        Rcpp::stop("Initial Louvain label is out of range.");
      }
      labels_[row] = label;
      ++counts_[label - 1];
    }
    double internal = 0.0;
    scan_graph(chunk_rows, edge_budget,
      [&](std::uint64_t source, std::uint32_t target, double weight) {
        const auto label = labels_[source - 1];
        volumes_[label - 1] += weight;
        graph_volume_ += weight;
        if (source <= target && label == labels_[target - 1]) {
          internal += source == target ? weight / 2.0 : weight;
        }
      });
    if (!(graph_volume_ > 0.0)) {
      Rcpp::stop("Louvain requires positive graph edge weight.");
    }
    double penalty = 0.0;
    for (std::uint64_t row = 0; row < n_; ++row) {
      if (counts_[row] == 0) continue;
      const double fraction = volumes_[row] / graph_volume_;
      penalty += fraction * fraction;
    }
    initial_modularity_ = internal / (graph_volume_ / 2.0) -
      resolution * penalty;
  }

  void optimize(int max_passes, int chunk_rows,
                std::uint64_t edge_budget, double resolution,
                std::uint64_t seed) {
    std::mt19937_64 generator(seed);
    const double tolerance = louvain_move_tolerance(
      graph_volume_ / 2.0);
    for (int pass = 0; pass < max_passes; ++pass) {
      const auto moved = scan_pass(chunk_rows, edge_budget,
        resolution, tolerance, generator);
      moves_ += moved;
      ++passes_;
      Rcpp::Rcout << "  Louvain pass " << passes_
                  << ": " << moved << " moves\n";
      if (moved == 0) break;
    }
  }

  double modularity(int chunk_rows, std::uint64_t edge_budget,
                    double resolution) {
    double penalty = 0.0;
    for (std::uint64_t row = 0; row < n_; ++row) {
      if (counts_[row] == 0) continue;
      const double fraction = volumes_[row] / graph_volume_;
      penalty += fraction * fraction;
    }
    double internal = 0.0;
    scan_graph(chunk_rows, edge_budget,
      [&](std::uint64_t source, std::uint32_t target, double weight) {
        if (source <= target && labels_[source - 1] ==
            labels_[target - 1]) {
          internal += source == target ? weight / 2.0 : weight;
        }
      });
    return internal / (graph_volume_ / 2.0) - resolution * penalty;
  }

  std::uint32_t compact() {
    std::uint32_t communities = 0;
    for (std::uint64_t row = 0; row < n_; ++row) {
      counts_[row] = counts_[row] > 0 ? ++communities : 0;
    }
    for (std::uint64_t row = 0; row < n_; ++row) {
      labels_[row] = counts_[labels_[row] - 1];
      if (labels_[row] == 0) Rcpp::stop("Louvain label compaction failed.");
    }
    labels_.sync();
    return communities;
  }

  double initial_modularity() const { return initial_modularity_; }
  double graph_volume() const { return graph_volume_; }
  std::uint64_t moves() const { return moves_; }
  int passes() const { return passes_; }

 private:
  template <typename Function>
  void scan_graph(int chunk_rows, std::uint64_t edge_budget,
                  Function consume) {
    for (std::uint64_t first = 1; first <= n_;) {
      const auto rows = graph_.bounded_rows(first, chunk_rows,
                                            edge_budget);
      graph_.visit(first, rows,
        [&](std::uint64_t source, std::uint32_t target, double weight,
            std::uint64_t) { consume(source, target, weight); });
      first += rows;
      Rcpp::checkUserInterrupt();
    }
  }

  std::uint32_t empty_label() {
    for (std::uint64_t checked = 0; checked < n_; ++checked) {
      const auto candidate = free_cursor_++ % n_;
      if (counts_[candidate] == 0) {
        return static_cast<std::uint32_t>(candidate + 1);
      }
    }
    Rcpp::stop("Louvain could not find an empty community label.");
    return 0;
  }

  bool move_row(std::uint64_t source, double degree,
                const std::unordered_map<std::uint32_t, double>& weights,
                double resolution, double tolerance) {
    if (!(degree > 0.0)) return false;
    const auto old = labels_[source - 1];
    const auto old_index = static_cast<std::uint64_t>(old - 1);
    if (counts_[old_index] == 0) {
      Rcpp::stop("Louvain community state is inconsistent.");
    }
    --counts_[old_index];
    volumes_[old_index] -= degree;
    if (counts_[old_index] == 0) volumes_[old_index] = 0.0;
    const auto found = weights.find(old);
    const double old_weight = found == weights.end() ? 0.0 :
      found->second;
    const double old_score = louvain_move_score(old_weight, degree,
      volumes_[old_index], graph_volume_, resolution);
    std::uint32_t best = old;
    double best_score = old_score;
    if (best_score < -tolerance && counts_[old_index] > 0) {
      best = 0;
      best_score = 0.0;
    }
    for (const auto& candidate : weights) {
      const auto index = static_cast<std::uint64_t>(candidate.first - 1);
      if (candidate.first == old || counts_[index] == 0) continue;
      const double score = louvain_move_score(candidate.second,
        degree, volumes_[index], graph_volume_, resolution);
      if (score > best_score + tolerance) {
        best = candidate.first;
        best_score = score;
      }
    }
    if (best_score <= old_score + tolerance) best = old;
    if (best == 0) best = empty_label();
    const auto best_index = static_cast<std::uint64_t>(best - 1);
    ++counts_[best_index];
    volumes_[best_index] += degree;
    labels_[source - 1] = best;
    return best != old;
  }

  std::uint64_t scan_pass(int chunk_rows,
                          std::uint64_t edge_budget,
                          double resolution, double tolerance,
                          std::mt19937_64& generator) {
    std::uint64_t moves = 0;
    const auto blocks = (n_ - 1) / chunk_rows + 1;
    const auto offset = generator() % blocks;
    for (std::uint64_t block = 0; block < blocks; ++block) {
      const auto first = ((offset + block) % blocks) *
        chunk_rows + 1;
      const auto end = std::min(n_, first + chunk_rows - 1);
      moves += scan_segment(first, end, edge_budget,
        resolution, tolerance, generator);
    }
    return moves;
  }

  std::uint64_t scan_segment(std::uint64_t first, std::uint64_t end,
                             std::uint64_t edge_budget,
                             double resolution, double tolerance,
                             std::mt19937_64& generator) {
    std::uint64_t moves = 0;
    std::unordered_map<std::uint32_t, double> weights;
    std::vector<std::pair<std::uint32_t, double>> edges;
    std::vector<std::uint64_t> row_ptr;
    std::vector<std::uint32_t> order;
    while (first <= end) {
      const auto rows = graph_.bounded_rows(first,
        end - first + 1, edge_budget);
      edges.clear();
      row_ptr.assign(rows + 1, 0);
      graph_.visit(first, rows,
        [&](std::uint64_t source, std::uint32_t target, double weight,
            std::uint64_t) {
          edges.emplace_back(target, weight);
          ++row_ptr[source - first + 1];
        });
      std::partial_sum(row_ptr.begin(), row_ptr.end(), row_ptr.begin());
      order.resize(rows);
      std::iota(order.begin(), order.end(), 0);
      std::shuffle(order.begin(), order.end(), generator);
      for (const auto row : order) {
        weights.clear();
        double degree = 0.0;
        for (auto edge = row_ptr[row]; edge < row_ptr[row + 1]; ++edge) {
          const auto& neighbor = edges[edge];
          degree += neighbor.second;
          if (neighbor.first != first + row) {
            weights[labels_[neighbor.first - 1]] += neighbor.second;
          }
        }
        moves += move_row(first + row, degree, weights,
                          resolution, tolerance);
      }
      first += rows;
      Rcpp::checkUserInterrupt();
    }
    return moves;
  }

  CsrGraphSource& graph_;
  std::uint64_t n_;
  MappedLouvainVector<std::uint32_t> labels_;
  MappedLouvainVector<std::uint32_t> counts_;
  MappedLouvainVector<double> volumes_;
  double graph_volume_ = 0.0;
  double initial_modularity_ = 0.0;
  std::uint64_t moves_ = 0;
  std::uint64_t free_cursor_ = 0;
  int passes_ = 0;
};

class MassiveLeidenRefinement {
 public:
  MassiveLeidenRefinement(CsrGraphSource& graph,
                         const std::string& parent_path,
                         int parent_count, const std::string& output)
      : graph_(graph), n_(graph.vertices()),
        parent_count_(parent_count),
        parent_(parent_path, checked_product(n_, 4), "mmap"),
        labels_(output + ".part", n_),
        sizes_(output + ".sizes.part", n_),
        volumes_(output + ".volumes.part", n_),
        cuts_(output + ".cuts.part", n_),
        parent_volumes_(output + ".parent_volumes.part",
                        parent_count) {}

  void initialize(int chunk_rows, std::uint64_t edge_budget) {
    for (std::uint64_t row = 0; row < n_; ++row) {
      parent_at(row);
      labels_[row] = static_cast<std::uint32_t>(row + 1);
      sizes_[row] = 1;
      volumes_[row] = 0.0;
      cuts_[row] = 0.0;
    }
    for (int group = 0; group < parent_count_; ++group) {
      parent_volumes_[group] = 0.0;
    }
    for (std::uint64_t first = 1; first <= n_;) {
      const auto rows = graph_.bounded_rows(first, chunk_rows,
                                            edge_budget);
      graph_.visit(first, rows,
        [&](std::uint64_t source, std::uint32_t target,
            double weight, std::uint64_t) {
        const auto own_parent = parent_at(source - 1);
        volumes_[source - 1] += weight;
        parent_volumes_[own_parent - 1] += weight;
        graph_volume_ += weight;
        if (source != target && own_parent == parent_at(target - 1)) {
          cuts_[source - 1] += weight;
        }
      });
      first += rows;
      Rcpp::checkUserInterrupt();
    }
    if (!(graph_volume_ > 0.0)) {
      Rcpp::stop("Leiden requires positive graph edge weight.");
    }
  }

  void optimize(int chunk_rows, std::uint64_t edge_budget,
                double resolution, std::uint64_t seed) {
    std::mt19937_64 generator(seed);
    const auto blocks = (n_ - 1) / chunk_rows + 1;
    const auto offset = generator() % blocks;
    const double tolerance = 1e-12 *
      std::max(1.0, graph_volume_ / 2.0);
    for (std::uint64_t block = 0; block < blocks; ++block) {
      const auto first = ((offset + block) % blocks) *
        chunk_rows + 1;
      const auto end = std::min(n_, first + chunk_rows - 1);
      refine_segment(first, end, edge_budget, resolution,
                     tolerance, generator);
      Rcpp::checkUserInterrupt();
    }
  }

  std::uint32_t compact(const std::string& mapping_path) {
    std::uint32_t groups = 0;
    for (std::uint64_t row = 0; row < n_; ++row) {
      sizes_[row] = sizes_[row] > 0 ? ++groups : 0;
    }
    MappedLouvainVector<std::uint32_t> mapping(mapping_path, groups);
    for (std::uint32_t group = 0; group < groups; ++group) {
      mapping[group] = 0;
    }
    for (std::uint64_t row = 0; row < n_; ++row) {
      const auto group = sizes_[labels_[row] - 1];
      if (group == 0) Rcpp::stop("Leiden label compaction failed.");
      labels_[row] = group;
      const auto parent_label = parent_at(row);
      if (mapping[group - 1] == 0) {
        mapping[group - 1] = parent_label;
      } else if (mapping[group - 1] != parent_label) {
        Rcpp::stop("Leiden refinement crossed a parent community.");
      }
    }
    labels_.sync();
    mapping.sync();
    return groups;
  }

  std::uint64_t moves() const { return moves_; }

 private:
  std::uint32_t parent_at(std::uint64_t row) {
    const auto label = read_u32(parent_.read(
      checked_product(row, 4), 4));
    if (label < 1 || label > static_cast<std::uint32_t>(parent_count_)) {
      Rcpp::stop("Leiden parent label is out of range.");
    }
    return label;
  }

  void refine_row(std::uint64_t row,
                  const std::vector<std::pair<std::uint32_t, double>>& edges,
                  std::uint64_t begin, std::uint64_t end,
                  double resolution, double tolerance) {
    if (sizes_[row - 1] != 1) return;
    const auto parent_id = parent_at(row - 1);
    const double degree = volumes_[row - 1];
    const double parent_volume = parent_volumes_[parent_id - 1];
    const double threshold = resolution * degree *
      (parent_volume - degree) / graph_volume_;
    if (cuts_[row - 1] + tolerance < threshold) return;
    std::unordered_map<std::uint32_t, double> weights;
    std::vector<std::uint32_t> touched;
    for (auto edge = begin; edge < end; ++edge) {
      const auto& neighbor = edges[edge];
      if (neighbor.first == row ||
          parent_at(neighbor.first - 1) != parent_id) continue;
      const auto candidate = labels_[neighbor.first - 1];
      if (candidate == row) continue;
      if (weights.emplace(candidate, 0.0).second) {
        touched.push_back(candidate);
      }
      weights[candidate] += neighbor.second;
    }
    std::uint32_t best = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (const auto candidate : touched) {
      const double candidate_volume = volumes_[candidate - 1];
      const double candidate_threshold = resolution * candidate_volume *
        (parent_volume - candidate_volume) / graph_volume_;
      if (cuts_[candidate - 1] + tolerance < candidate_threshold) {
        continue;
      }
      const double score = weights[candidate] - resolution *
        degree * candidate_volume / graph_volume_;
      if (score >= -tolerance && score > best_score + tolerance) {
        best = candidate;
        best_score = score;
      }
    }
    if (best == 0) return;
    labels_[row - 1] = best;
    ++sizes_[best - 1];
    sizes_[row - 1] = 0;
    volumes_[best - 1] += degree;
    volumes_[row - 1] = 0.0;
    cuts_[best - 1] += cuts_[row - 1] - 2.0 * weights[best];
    cuts_[row - 1] = 0.0;
    ++moves_;
  }

  void refine_segment(std::uint64_t first, std::uint64_t end,
                      std::uint64_t edge_budget, double resolution,
                      double tolerance, std::mt19937_64& generator) {
    std::vector<std::pair<std::uint32_t, double>> edges;
    std::vector<std::uint64_t> row_ptr;
    std::vector<std::uint32_t> order;
    while (first <= end) {
      const auto rows = graph_.bounded_rows(first,
        end - first + 1, edge_budget);
      edges.clear();
      row_ptr.assign(rows + 1, 0);
      graph_.visit(first, rows,
        [&](std::uint64_t source, std::uint32_t target,
            double weight, std::uint64_t) {
        edges.emplace_back(target, weight);
        ++row_ptr[source - first + 1];
      });
      std::partial_sum(row_ptr.begin(), row_ptr.end(), row_ptr.begin());
      order.resize(rows);
      std::iota(order.begin(), order.end(), 0);
      std::shuffle(order.begin(), order.end(), generator);
      for (const auto offset : order) {
        refine_row(first + offset, edges, row_ptr[offset],
                   row_ptr[offset + 1], resolution, tolerance);
      }
      first += rows;
      Rcpp::checkUserInterrupt();
    }
  }

  CsrGraphSource& graph_;
  std::uint64_t n_;
  int parent_count_;
  GraphFile parent_;
  MappedLouvainVector<std::uint32_t> labels_;
  MappedLouvainVector<std::uint32_t> sizes_;
  MappedLouvainVector<double> volumes_;
  MappedLouvainVector<double> cuts_;
  MappedLouvainVector<double> parent_volumes_;
  double graph_volume_ = 0.0;
  std::uint64_t moves_ = 0;
};
#endif

void checkpoint_output(Float32Sink& sink, SEXP progress,
                       std::uint64_t rows) {
  if (progress == R_NilValue) return;
  sink.flush();
  Rcpp::Function callback(progress);
  callback(static_cast<double>(rows));
}

void report_graph_progress(const char* stage, std::uint64_t done,
                           std::uint64_t total, int& next_percent) {
  if (done * 100 < static_cast<std::uint64_t>(next_percent) * total) {
    return;
  }
  const int percent = static_cast<int>(done * 100 / total);
  Rcpp::Rcout << "  " << stage << ": " << percent << "%\n";
  next_percent = (percent / 10 + 1) * 10;
}

}  // namespace

void massive_csr_attraction(const std::string& offsets_path,
                            const std::string& indices_path,
                            const std::string& weights_path,
                            int n, const std::string& access,
                            const std::vector<float>& y, int dims,
                            float exaggeration, int n_threads,
                            std::vector<float>& grad) {
  const auto size = static_cast<std::size_t>(n) * dims;
  if (n < 2 || (dims != 2 && dims != 3) ||
      y.size() != size || grad.size() != size ||
      !std::isfinite(exaggeration) || exaggeration <= 0.0f ||
      n_threads < 1) {
    Rcpp::stop("Invalid massive t-SNE attraction buffers.");
  }
  const int threads = std::min(n, n_threads);
  std::vector<std::exception_ptr> errors(threads);
  parallel_rows(n, threads,
    [&](std::uint64_t begin, std::uint64_t end, int worker) {
      try {
        CsrGraphSource graph(offsets_path, indices_path, weights_path,
                             n, access);
        for (std::uint64_t first = begin + 1; first <= end;) {
          const auto count = std::min(graph.bounded_rows(
            first, 8192, 1000000), end - first + 1);
          graph.visit(first, count,
            [&](std::uint64_t source, std::uint32_t target,
                double weight, std::uint64_t) {
              const auto from = static_cast<std::size_t>(source - 1) * dims;
              const auto to = static_cast<std::size_t>(target - 1) * dims;
              float difference[3] = {0.0f, 0.0f, 0.0f};
              float squared = 0.0f;
              for (int axis = 0; axis < dims; ++axis) {
                difference[axis] = y[from + axis] - y[to + axis];
                squared += difference[axis] * difference[axis];
              }
              const float factor = exaggeration *
                static_cast<float>(weight) / (1.0f + squared);
              for (int axis = 0; axis < dims; ++axis) {
                grad[from + axis] += factor * difference[axis];
              }
            }, false);
          first += count;
        }
      } catch (...) {
        errors[worker] = std::current_exception();
      }
  });
  for (const auto& error : errors) {
    if (error) std::rethrow_exception(error);
  }
}


// [[Rcpp::export]]
List massive_file_info_cpp(std::string path,
                           std::string format,
                           double rows,
                           double columns) {
  const auto info = inspect_file(path, format, rows, columns);
  return List::create(
    Rcpp::Named("nrow") = static_cast<double>(info.rows),
    Rcpp::Named("ncol") = static_cast<double>(info.columns),
    Rcpp::Named("bytes") = static_cast<double>(info.bytes),
    Rcpp::Named("offset_bytes") = static_cast<double>(info.offset),
    Rcpp::Named("stamp") = file_stamp(path)
  );
}

// [[Rcpp::export]]
double massive_disk_available_cpp(std::string output) {
  std::error_code error;
  const auto parent = std::filesystem::path(output).parent_path();
  const auto space = std::filesystem::space(parent, error);
  if (error) Rcpp::stop("Cannot inspect output disk capacity.");
  return static_cast<double>(space.available);
}

// [[Rcpp::export]]
List massive_graph_modularity_cpp(std::string offsets_path,
                                  std::string indices_path,
                                  std::string weights_path,
                                  std::string membership_path,
                                  double n_vertices,
                                  int n_communities,
                                  double resolution,
                                  int chunk_rows,
                                  double edge_budget,
                                  bool connectivity) {
  if (n_communities < 1 || !std::isfinite(resolution) ||
      resolution <= 0.0 || chunk_rows < 1) {
    Rcpp::stop("Invalid massive modularity controls.");
  }
  const auto maximum = whole_number(edge_budget, "edge_budget");
  if (maximum < 1) Rcpp::stop("Modularity edge budget is empty.");
  CsrGraphSource graph(offsets_path, indices_path, weights_path,
                       n_vertices, "stream");
  if (connectivity && graph.vertices() > INT_MAX) {
    Rcpp::stop("Connectivity audit exceeds the vertex-ID limit.");
  }
  GraphFile labels(membership_path,
                   checked_product(graph.vertices(), 4), "mmap");
  std::vector<std::uint32_t> parent;
  std::vector<std::uint32_t> size;
  if (connectivity) {
    parent.resize(graph.vertices());
    std::iota(parent.begin(), parent.end(), 0u);
    size.assign(graph.vertices(), 1u);
  }
  auto root_of = [&](std::uint32_t row) {
    while (parent[row] != row) {
      parent[row] = parent[parent[row]];
      row = parent[row];
    }
    return row;
  };
  auto label_at = [&](std::uint64_t row) {
    const auto label = read_u32(labels.read(
      checked_product(row - 1, 4), 4));
    if (label < 1 || label > static_cast<std::uint32_t>(n_communities)) {
      Rcpp::stop("Cluster label exceeds `n_communities`.");
    }
    return static_cast<std::size_t>(label - 1);
  };
  for (std::uint64_t row = 1; row <= graph.vertices(); ++row) {
    label_at(row);
    if (row % 65536 == 0) Rcpp::checkUserInterrupt();
  }
  std::vector<double> volume(n_communities, 0.0);
  std::vector<double> internal(n_communities, 0.0);
  double total = 0.0;
  std::uint64_t pairs = 0;
  int next_percent = 10;
  for (std::uint64_t first = 1; first <= graph.vertices();) {
    const auto rows = graph.bounded_rows(first, chunk_rows, maximum);
    graph.visit(first, rows,
      [&](std::uint64_t source, std::uint32_t target,
          float weight, std::uint64_t) {
      if (source >= target) return;
      const auto left = label_at(source);
      const auto right = label_at(target);
      volume[left] += weight;
      volume[right] += weight;
      if (left == right) internal[left] += weight;
      if (connectivity && left == right) {
        auto a = root_of(static_cast<std::uint32_t>(source - 1));
        auto b = root_of(target - 1);
        if (a != b) {
          if (size[a] < size[b]) std::swap(a, b);
          parent[b] = a;
          size[a] += size[b];
        }
      }
      total += weight;
      ++pairs;
    });
    first += rows;
    report_graph_progress("Graph modularity", first - 1,
                          graph.vertices(), next_percent);
    Rcpp::checkUserInterrupt();
  }
  double score = NA_REAL;
  double inside = 0.0;
  if (total > 0.0) {
    score = 0.0;
    for (int community = 0; community < n_communities; ++community) {
      const auto index = static_cast<std::size_t>(community);
      const double fraction = volume[index] / (2.0 * total);
      inside += internal[index];
      score += internal[index] / total - resolution * fraction * fraction;
    }
  }
  int disconnected = NA_INTEGER;
  double component_count = NA_REAL;
  if (connectivity) {
    std::vector<int> per_label(n_communities, 0);
    component_count = 0;
    for (std::uint64_t row = 0; row < graph.vertices(); ++row) {
      const auto root = root_of(static_cast<std::uint32_t>(row));
      if (size[root] != 0) {
        ++per_label[label_at(row + 1)];
        ++component_count;
        size[root] = 0;
      }
      if (row % 65536 == 0) Rcpp::checkUserInterrupt();
    }
    disconnected = static_cast<int>(std::count_if(
      per_label.begin(), per_label.end(),
      [](int count) { return count > 1; }));
  }
  return List::create(
    Rcpp::Named("modularity") = score,
    Rcpp::Named("internal_weight") = inside,
    Rcpp::Named("total_edge_weight") = total,
    Rcpp::Named("n_edge_pairs") = static_cast<double>(pairs),
    Rcpp::Named("n_communities") = n_communities,
    Rcpp::Named("connectivity_audited") = connectivity,
    Rcpp::Named("disconnected_communities") = disconnected,
    Rcpp::Named("n_components") = component_count,
    Rcpp::Named("backend") = "cpu"
  );
}

// [[Rcpp::export]]
List massive_louvain_level_cpp(std::string offsets_path,
                               std::string indices_path,
                               std::string weights_path,
                               double n_vertices,
                               std::string output,
                               int max_passes,
                               double resolution,
                               int chunk_rows,
                               double edge_budget,
                               int seed,
                               bool contracted,
                               std::string initial_path = "",
                               int initial_count = 0) {
#if defined(__unix__) || defined(__APPLE__)
  if (max_passes < 1 || chunk_rows < 1 || seed < 0 ||
      !std::isfinite(resolution) || resolution <= 0) {
    Rcpp::stop("Invalid massive Louvain controls.");
  }
  const auto maximum = whole_number(edge_budget, "edge_budget");
  if (maximum < 1) Rcpp::stop("Louvain edge budget is empty.");
  CsrGraphSource graph(offsets_path, indices_path, weights_path,
                       n_vertices, "stream", contracted);
  std::vector<std::string> inputs = {
    offsets_path, indices_path, weights_path};
  if (!initial_path.empty()) inputs.push_back(initial_path);
  std::vector<std::string> input_stamps;
  std::vector<std::uint64_t> input_sizes;
  for (const auto& path : inputs) {
    input_stamps.push_back(file_stamp(path));
    input_sizes.push_back(std::filesystem::file_size(path));
  }
  const std::string part = output + ".part";
  const std::string counts = output + ".counts.part";
  const std::string volumes = output + ".volumes.part";
  if (std::filesystem::exists(output) ||
      std::filesystem::exists(part) ||
      std::filesystem::exists(counts) ||
      std::filesystem::exists(volumes)) {
    Rcpp::stop("Louvain output or partial state already exists.");
  }
  double initial = 0.0;
  double final = 0.0;
  std::uint64_t moves = 0;
  int passes = 0;
  std::uint32_t communities = 0;
  double total_edge_weight = 0.0;
  {
    MassiveLouvainLevel level(graph, output);
    level.initialize(chunk_rows, maximum, resolution,
                     initial_path, initial_count);
    level.optimize(max_passes, chunk_rows, maximum, resolution,
                   static_cast<std::uint64_t>(seed));
    initial = level.initial_modularity();
    final = level.modularity(chunk_rows, maximum, resolution);
    if (final < initial - 1e-7) {
      Rcpp::stop("Louvain local moves reduced modularity.");
    }
    communities = level.compact();
    moves = level.moves();
    passes = level.passes();
    total_edge_weight = level.graph_volume() / 2.0;
  }
  if (std::filesystem::exists(output)) {
    Rcpp::stop("Louvain output appeared during computation.");
  }
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    std::error_code inspect_error;
    const auto size = std::filesystem::file_size(inputs[index],
                                                inspect_error);
    if (inspect_error || size != input_sizes[index] ||
        file_stamp(inputs[index]) != input_stamps[index]) {
      Rcpp::stop("Massive graph files changed during Louvain.");
    }
  }
  std::error_code error;
  std::filesystem::rename(part, output, error);
  if (error) Rcpp::stop("Cannot finalize Louvain labels: " +
                        error.message());
  std::filesystem::remove(counts, error);
  if (error) Rcpp::warning("Louvain counts file was retained.");
  error.clear();
  std::filesystem::remove(volumes, error);
  if (error) Rcpp::warning("Louvain volumes file was retained.");
  return List::create(
    Rcpp::Named("membership_path") = output,
    Rcpp::Named("n_vertices") = static_cast<double>(graph.vertices()),
    Rcpp::Named("n_communities") = static_cast<int>(communities),
    Rcpp::Named("modularity_initial") = initial,
    Rcpp::Named("modularity_final") = final,
    Rcpp::Named("total_edge_weight") = total_edge_weight,
    Rcpp::Named("moves") = static_cast<double>(moves),
    Rcpp::Named("passes") = passes,
    Rcpp::Named("scan_order") = "cyclic_blocks_shuffled_rows",
    Rcpp::Named("initial_partition") = !initial_path.empty(),
    Rcpp::Named("backend") = "cpu"
  );
#else
  Rcpp::stop("Massive Louvain requires POSIX memory mapping.");
#endif
}

// [[Rcpp::export]]
List massive_leiden_refine_cpp(std::string offsets_path,
                               std::string indices_path,
                               std::string weights_path,
                               std::string parent_path,
                               double n_vertices,
                               int n_parent,
                               std::string output,
                               double resolution,
                               int chunk_rows,
                               double edge_budget,
                               int seed,
                               bool contracted) {
#if defined(__unix__) || defined(__APPLE__)
  if (n_parent < 1 || chunk_rows < 1 || seed < 0 ||
      !std::isfinite(resolution) || resolution <= 0) {
    Rcpp::stop("Invalid massive Leiden refinement controls.");
  }
  const auto maximum = whole_number(edge_budget, "edge_budget");
  if (maximum < 1) Rcpp::stop("Leiden edge budget is empty.");
  CsrGraphSource graph(offsets_path, indices_path, weights_path,
                       n_vertices, "stream", contracted);
  if (static_cast<std::uint64_t>(n_parent) > graph.vertices()) {
    Rcpp::stop("Leiden parent count exceeds graph vertices.");
  }
  const std::vector<std::string> inputs = {
    offsets_path, indices_path, weights_path, parent_path};
  std::vector<std::string> stamps;
  std::vector<std::uint64_t> sizes;
  for (const auto& path : inputs) {
    stamps.push_back(file_stamp(path));
    sizes.push_back(std::filesystem::file_size(path));
  }
  const std::string map = output + ".parent.u32";
  const std::vector<std::string> created = {
    output, output + ".part", map, map + ".part",
    output + ".sizes.part", output + ".volumes.part",
    output + ".cuts.part", output + ".parent_volumes.part"};
  for (const auto& path : created) {
    if (std::filesystem::exists(path)) {
      Rcpp::stop("Leiden output or partial state already exists.");
    }
  }
  std::uint32_t communities = 0;
  std::uint64_t moves = 0;
  {
    MassiveLeidenRefinement refinement(graph, parent_path,
                                       n_parent, output);
    refinement.initialize(chunk_rows, maximum);
    refinement.optimize(chunk_rows, maximum, resolution,
                        static_cast<std::uint64_t>(seed));
    communities = refinement.compact(map + ".part");
    moves = refinement.moves();
  }
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    std::error_code error;
    const auto size = std::filesystem::file_size(inputs[index], error);
    if (error || size != sizes[index] ||
        file_stamp(inputs[index]) != stamps[index]) {
      Rcpp::stop("Massive graph or parent changed during Leiden.");
    }
  }
  std::error_code error;
  std::filesystem::rename(output + ".part", output, error);
  if (error) Rcpp::stop("Cannot finalize Leiden labels: " +
                        error.message());
  std::filesystem::rename(map + ".part", map, error);
  if (error) Rcpp::stop("Cannot finalize Leiden parent map: " +
                        error.message());
  for (const auto& suffix : {".sizes.part", ".volumes.part",
                              ".cuts.part", ".parent_volumes.part"}) {
    error.clear();
    std::filesystem::remove(output + suffix, error);
    if (error) Rcpp::warning("Leiden mapped state was retained.");
  }
  return List::create(
    Rcpp::Named("membership_path") = output,
    Rcpp::Named("parent_mapping_path") = map,
    Rcpp::Named("n_vertices") = static_cast<double>(graph.vertices()),
    Rcpp::Named("n_refined") = static_cast<int>(communities),
    Rcpp::Named("n_parent") = n_parent,
    Rcpp::Named("moves") = static_cast<double>(moves),
    Rcpp::Named("scan_order") = "cyclic_blocks_shuffled_rows",
    Rcpp::Named("backend") = "cpu"
  );
#else
  Rcpp::stop("Massive Leiden requires POSIX memory mapping.");
#endif
}

// [[Rcpp::export]]
SEXP massive_reference_buffer_cpp(List spec) {
  auto source = make_source(spec);
  if (source->nrow() > std::numeric_limits<int>::max() ||
      source->ncol() > std::numeric_limits<int>::max()) {
    Rcpp::stop("Landmark index dimensions exceed native integer limits.");
  }
  const auto items = checked_product(source->nrow(), source->ncol());
  if (items > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    Rcpp::stop("Landmark reference exceeds addressable memory.");
  }
  Rcpp::XPtr<fastembedr::FloatMatrix> buffer(
    new fastembedr::FloatMatrix(), true
  );
  buffer->nrow = static_cast<int>(source->nrow());
  buffer->ncol = static_cast<int>(source->ncol());
  buffer->input_float32 = true;
  buffer->values.resize(static_cast<std::size_t>(items));
  source->read_rows(0, source->nrow(), buffer->values.data());
  R_SetExternalPtrTag(buffer,
    Rf_install("fastEmbedR_massive_float_buffer"));
  return buffer;
}

// [[Rcpp::export]]
SEXP massive_read_rows_cpp(List spec,
                           double first_row,
                           int count,
                           double max_bytes,
                           bool float32 = false) {
  auto source = make_source(spec);
  const auto start = whole_number(first_row, "first_row");
  if (start < 1 || count < 1 || start > source->nrow() ||
      static_cast<std::uint64_t>(count) > source->nrow() - start + 1) {
    Rcpp::stop("Requested massive row range is invalid.");
  }
  const auto items = checked_product(count, source->ncol());
  if (!std::isfinite(max_bytes) || max_bytes <= 0) {
    Rcpp::stop("Massive row read needs a positive byte limit.");
  }
  const int bytes_per_item = float32 ? 4 : 8;
  if (source->ncol() > std::numeric_limits<int>::max() ||
      items > static_cast<std::uint64_t>(
        std::floor(max_bytes / bytes_per_item))) {
    Rcpp::stop("Massive row read exceeds its byte limit.");
  }
  std::vector<float> buffer(static_cast<std::size_t>(items));
  source->read_rows(start - 1, count, buffer.data());
  if (float32) {
    Rcpp::IntegerMatrix payload(count, static_cast<int>(source->ncol()));
    for (int col = 0; col < payload.ncol(); ++col) {
      for (int row = 0; row < count; ++row) {
        const float value = buffer[
          static_cast<std::size_t>(row) * payload.ncol() + col
        ];
        std::memcpy(&payload(row, col), &value, sizeof(float));
      }
    }
    Rcpp::S4 result("float32");
    result.slot("Data") = payload;
    return result;
  }
  NumericMatrix result(count, static_cast<int>(source->ncol()));
  for (int col = 0; col < result.ncol(); ++col) {
    for (int row = 0; row < count; ++row) {
      result(row, col) = buffer[
        static_cast<std::size_t>(row) * result.ncol() + col
      ];
    }
  }
  return result;
}

// [[Rcpp::export]]
List massive_read_knn_rows_cpp(std::string indices_path,
                              std::string distances_path,
                              double n_rows,
                              int k,
                              double n_reference,
                              double first_row,
                              int count,
                              double max_bytes) {
  require_little_endian();
  const auto rows = whole_number(n_rows, "n_rows");
  const auto references = whole_number(n_reference, "n_reference");
  const auto first = whole_number(first_row, "first_row");
  if (k < 1 || count < 1 || first < 1 || first > rows ||
      references < 1 ||
      references > static_cast<std::uint64_t>(
        std::numeric_limits<std::int32_t>::max()) ||
      static_cast<std::uint64_t>(count) > rows - first + 1) {
    Rcpp::stop("Invalid massive KNN row range or dimensions.");
  }
  const auto items = checked_product(count, k);
  if (!std::isfinite(max_bytes) || max_bytes <= 0 ||
      items > static_cast<std::uint64_t>(std::floor(max_bytes / 20))) {
    Rcpp::stop("Massive KNN row read exceeds its byte limit.");
  }
  const auto expected = checked_product(checked_product(rows, k), 4);
  const auto offset = checked_product(
    checked_product(first - 1, k), 4
  );
  if (offset > static_cast<std::uint64_t>(
      std::numeric_limits<std::streamoff>::max())) {
    Rcpp::stop("Massive KNN file offset exceeds stream limits.");
  }
  std::error_code error;
  const auto index_size = std::filesystem::file_size(indices_path, error);
  if (error || index_size != expected) {
    Rcpp::stop("Massive KNN index file size mismatch.");
  }
  const auto distance_size = std::filesystem::file_size(
    distances_path, error
  );
  if (error || distance_size != expected) {
    Rcpp::stop("Massive KNN distance file size mismatch.");
  }
  std::ifstream index_stream(indices_path, std::ios::binary);
  std::ifstream distance_stream(distances_path, std::ios::binary);
  index_stream.seekg(static_cast<std::streamoff>(offset));
  distance_stream.seekg(static_cast<std::streamoff>(offset));
  std::vector<unsigned char> ids(checked_product(items, 4));
  std::vector<unsigned char> values(checked_product(items, 4));
  index_stream.read(reinterpret_cast<char*>(ids.data()), ids.size());
  distance_stream.read(
    reinterpret_cast<char*>(values.data()), values.size()
  );
  if (!index_stream || !distance_stream) {
    Rcpp::stop("Cannot read massive KNN row range.");
  }
  Rcpp::IntegerMatrix indices(count, k);
  Rcpp::NumericMatrix distances(count, k);
  for (int row = 0; row < count; ++row) {
    for (int col = 0; col < k; ++col) {
      const auto pos = static_cast<std::size_t>(row) * k + col;
      const auto id = read_u32(ids.data() + 4 * pos);
      float distance = 0.0f;
      std::memcpy(&distance, values.data() + 4 * pos, 4);
      if (id < 1 || id > references ||
          !std::isfinite(distance) || distance < 0) {
        Rcpp::stop("Massive KNN row contains invalid values.");
      }
      indices(row, col) = static_cast<int>(id);
      distances(row, col) = distance;
    }
  }
  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances
  );
}

// [[Rcpp::export]]
List massive_validate_graph_cpp(std::string indices_path,
                                std::string distances_path,
                                double n_vertices, int k,
                                std::string access,
                                std::string kind) {
  KnnGraphSource graph(indices_path, distances_path,
                       n_vertices, k, access, kind);
  std::vector<std::uint32_t> neighbors(static_cast<std::size_t>(k));
  const auto rows_per_block = std::max(1, 65536 / k);
  for (std::uint64_t first = 1; first <= graph.vertices();) {
    const auto count = std::min<std::uint64_t>(
      rows_per_block, graph.vertices() - first + 1
    );
    graph.visit(first, count, [&](std::uint64_t, std::uint32_t target,
                                  float, std::uint64_t, int col) {
      neighbors[static_cast<std::size_t>(col)] = target;
      if (col == k - 1) {
        std::sort(neighbors.begin(), neighbors.end());
        if (std::adjacent_find(neighbors.begin(), neighbors.end()) !=
            neighbors.end()) {
          Rcpp::stop("Massive graph has duplicate neighbors.");
        }
      }
    });
    first += count;
    Rcpp::checkUserInterrupt();
  }
  return List::create(
    Rcpp::Named("n_vertices") = static_cast<double>(graph.vertices()),
    Rcpp::Named("n_edges") = static_cast<double>(graph.edges()),
    Rcpp::Named("validated") = true
  );
}

// [[Rcpp::export]]
List massive_graph_storage_cpp(double n_vertices, int k) {
  const auto vertices = whole_number(n_vertices, "n_vertices");
  if (vertices < 2 ||
      vertices > static_cast<std::uint64_t>(
        std::numeric_limits<std::int32_t>::max()) ||
      k < 1 || k > 65536 ||
      static_cast<std::uint64_t>(k) >= vertices) {
    Rcpp::stop("Invalid massive graph vertex count or k.");
  }
  const auto edges = checked_product(vertices, k);
  return List::create(
    Rcpp::Named("n_edges") = static_cast<double>(edges),
    Rcpp::Named("file_bytes") = static_cast<double>(
      checked_product(edges, 4)),
    Rcpp::Named("last_row_offset_bytes") = static_cast<double>(
      graph_row_offset(vertices, k))
  );
}

namespace {

std::string umap_mean_sum_text(long double total) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::setprecision(
    std::numeric_limits<long double>::max_digits10) << total;
  return stream.str();
}

long double umap_mean_parse_sum(const std::string& text) {
  std::istringstream stream(text);
  stream.imbue(std::locale::classic());
  long double total = 0.0L;
  stream >> total;
  if (!stream || !stream.eof() || !std::isfinite(total) || total < 0) {
    Rcpp::stop("Invalid UMAP membership mean checkpoint sum.");
  }
  return total;
}

double umap_mean_scan(const std::string& indices_path,
                      const std::string& distances_path,
                      double n_vertices, int k,
                      double completed_rows, const std::string& sum,
                      SEXP progress, int checkpoint_every) {
  KnnGraphSource graph(indices_path, distances_path,
                       n_vertices, k, "stream", "distance");
  const auto rows_per_block = std::max(1, 65536 / k);
  const auto start = whole_number(completed_rows, "completed_rows");
  if (start > graph.vertices() ||
      (start < graph.vertices() && start % rows_per_block != 0) ||
      checkpoint_every < 1) {
    Rcpp::stop("Invalid UMAP membership mean resume state.");
  }
  long double total = umap_mean_parse_sum(sum);
  std::uint64_t values = checked_product(start, k);
  std::uint64_t batch = start / rows_per_block;
  int next_report = 10;
  for (std::uint64_t first = start + 1;
       first <= graph.vertices();) {
    const auto rows = std::min<std::uint64_t>(
      rows_per_block, graph.vertices() - first + 1);
    graph.visit(first, rows,
                [&](std::uint64_t, std::uint32_t, float distance,
                    std::uint64_t, int) {
      total += static_cast<long double>(distance);
      ++values;
    });
    first += rows;
    ++batch;
    if (progress != R_NilValue &&
        (batch % checkpoint_every == 0 || first > graph.vertices())) {
      Rcpp::Function callback(progress);
      callback(static_cast<double>(first - 1),
               umap_mean_sum_text(total));
    }
    report_graph_progress("UMAP membership mean pass", first - 1,
                          graph.vertices(), next_report);
    Rcpp::checkUserInterrupt();
  }
  return values > 0 ? static_cast<double>(total / values) : 1.0;
}

}  // namespace

// [[Rcpp::export]]
double massive_umap_global_mean_cpp(std::string indices_path,
                                    std::string distances_path,
                                    double n_vertices, int k) {
  return umap_mean_scan(indices_path, distances_path, n_vertices, k,
                        0, "0", R_NilValue, 1);
}

// [[Rcpp::export]]
double massive_umap_global_mean_resume_cpp(
    std::string indices_path, std::string distances_path,
    double n_vertices, int k, double completed_rows,
    std::string sum, SEXP progress, int checkpoint_every) {
  if (progress == R_NilValue) {
    Rcpp::stop("UMAP mean resume requires a checkpoint callback.");
  }
  return umap_mean_scan(indices_path, distances_path, n_vertices, k,
                        completed_rows, sum, progress,
                        checkpoint_every);
}

// [[Rcpp::export]]
void massive_umap_memberships_cpp(std::string indices_path,
                                  std::string distances_path,
                                  double n_vertices, int k,
                                  std::string output_path,
                                  double global_mean,
                                  double start_row = 0,
                                  SEXP progress = R_NilValue,
                                  bool resume = false,
                                  int checkpoint_every = 100,
                                  int n_threads = 1) {
  KnnGraphSource graph(indices_path, distances_path,
                       n_vertices, k, "stream", "distance");
  const auto start = whole_number(start_row, "start_row");
  const auto rows_per_block = std::max(1, 65536 / k);
  if (!std::isfinite(global_mean) || global_mean < 0 ||
      start > graph.vertices() ||
      (start < graph.vertices() && start % rows_per_block != 0) ||
      (start > 0 && !resume) || checkpoint_every < 1 ||
      n_threads < 1 || n_threads > 256 ||
      (resume && progress == R_NilValue)) {
    Rcpp::stop("Invalid UMAP membership resume state or controls.");
  }
  const auto expected = checked_product(graph.edges(), 4);
  Float32Sink output(output_path, checked_product(start * k, 4),
                     expected, resume);
  std::vector<float> distances(65536 + static_cast<std::size_t>(k));
  std::vector<float> weights(65536 + static_cast<std::size_t>(k));
  std::uint64_t batch = start / rows_per_block;
  int next_report = 10;
  for (std::uint64_t first = start + 1;
       first <= graph.vertices();) {
    const auto rows = std::min<std::uint64_t>(
      rows_per_block, graph.vertices() - first + 1);
    graph.visit(first, rows,
                [&](std::uint64_t, std::uint32_t, float distance,
                    std::uint64_t row, int col) {
      distances[row * k + col] = distance;
    });
    parallel_rows(rows, n_threads,
      [&](std::uint64_t begin, std::uint64_t end, int) {
        for (auto row = begin; row < end; ++row) {
          const auto* values = distances.data() + row * k;
          const auto scale = umap_membership_scale(
            values, k, global_mean);
          for (int j = 0; j < k; ++j) {
            const float d = values[j];
            weights[row * k + j] = d <= scale.rho ? 1.0f :
              std::exp(-(d - scale.rho) / scale.sigma);
          }
        }
      });
    output.append(weights.data(), checked_product(rows, k));
    first += rows;
    ++batch;
    if (batch % checkpoint_every == 0 || first > graph.vertices()) {
      checkpoint_output(output, progress, first - 1);
    }
    report_graph_progress("UMAP membership write pass", first - 1,
                          graph.vertices(), next_report);
    Rcpp::checkUserInterrupt();
  }
  output.finish(expected);
}

namespace {

struct PairRecord {
  std::uint32_t low;
  std::uint32_t high;
  float weight;
};

static_assert(sizeof(PairRecord) == 12,
              "Graph pair records must occupy 12 bytes.");

struct RowRecord {
  std::uint32_t source;
  std::uint32_t target;
  float weight;
};

struct CoarseRowRecord {
  std::uint32_t source;
  std::uint32_t target;
  double weight;
};

struct CoarseRecord {
  std::uint32_t low;
  std::uint32_t high;
  double weight;
};

static_assert(sizeof(CoarseRecord) == 16,
              "Contracted graph records must occupy 16 bytes.");

struct CoarseLess {
  bool operator()(const CoarseRecord& left,
                  const CoarseRecord& right) const {
    if (left.low != right.low) return left.low < right.low;
    return left.high < right.high;
  }
};

struct PairLess {
  bool operator()(const PairRecord& left,
                  const PairRecord& right) const {
    if (left.low != right.low) return left.low < right.low;
    if (left.high != right.high) return left.high < right.high;
    return std::signbit(left.weight) < std::signbit(right.weight);
  }
};

struct RowLess {
  bool operator()(const RowRecord& left,
                  const RowRecord& right) const {
    if (left.source != right.source) return left.source < right.source;
    return left.target < right.target;
  }
};

struct CoarseRowLess {
  bool operator()(const CoarseRowRecord& left,
                  const CoarseRowRecord& right) const {
    if (left.source != right.source) return left.source < right.source;
    return left.target < right.target;
  }
};

class GraphDiskPeak {
 public:
  void add(const std::string& path) {
    const auto bytes = std::filesystem::file_size(path);
    if (!files_.emplace(path, bytes).second) {
      Rcpp::stop("Graph payload was counted twice.");
    }
    current_ += bytes;
    peak_ = std::max(peak_, current_);
    peak_with_external_ = std::max(peak_with_external_,
                                   current_ + external_);
  }

  void written(const std::string& path, std::uintmax_t bytes) {
    const auto found = files_.find(path);
    if (found == files_.end()) {
      Rcpp::stop("Graph payload was not tracked.");
    }
    found->second += bytes;
    current_ += bytes;
    peak_ = std::max(peak_, current_);
    peak_with_external_ = std::max(peak_with_external_,
                                   current_ + external_);
  }

  void remove(const std::string& path) {
    if (!std::filesystem::remove(path)) return;
    const auto found = files_.find(path);
    if (found == files_.end()) {
      Rcpp::stop("Graph payload was not tracked.");
    }
    current_ -= found->second;
    files_.erase(found);
  }

  double peak() const { return static_cast<double>(peak_); }
  double current() const { return static_cast<double>(current_); }
  double peak_with_external() const {
    return static_cast<double>(peak_with_external_);
  }
  void external(std::uintmax_t bytes) {
    external_ = bytes;
    peak_with_external_ = std::max(peak_with_external_,
                                   current_ + external_);
  }

 private:
  std::unordered_map<std::string, std::uintmax_t> files_;
  std::uintmax_t current_ = 0;
  std::uintmax_t peak_ = 0;
  std::uintmax_t external_ = 0;
  std::uintmax_t peak_with_external_ = 0;
};

template <typename Record>
class RunSink {
 public:
  explicit RunSink(const std::string& path,
                   GraphDiskPeak* disk = nullptr)
      : path_(path), disk_(disk), stream_(path, std::ios::binary) {
    if (!stream_) Rcpp::stop("Cannot create massive graph run.");
    if (disk_) disk_->add(path_);
    buffer_.reserve(16384);
  }

  void push(const Record& value) {
    buffer_.push_back(value);
    if (buffer_.size() == 16384) flush();
  }

  void finish() {
    flush();
    stream_.close();
    if (!stream_) Rcpp::stop("Massive graph run close failed.");
  }

 private:
  void flush() {
    if (buffer_.empty()) return;
    const auto bytes = buffer_.size() * sizeof(Record);
    stream_.write(reinterpret_cast<const char*>(buffer_.data()), bytes);
    if (!stream_) Rcpp::stop("Massive graph run write failed.");
    if (disk_) disk_->written(path_, bytes);
    buffer_.clear();
  }

  std::string path_;
  GraphDiskPeak* disk_;
  std::ofstream stream_;
  std::vector<Record> buffer_;
};

template <typename Record>
class RunReader {
 public:
  explicit RunReader(const std::string& path)
      : stream_(path, std::ios::binary), remaining_(0),
        position_(0), filled_(0), buffer_(4096) {
    std::error_code error;
    const auto bytes = std::filesystem::file_size(path, error);
    if (error || bytes % sizeof(Record) != 0 || !stream_) {
      Rcpp::stop("Massive graph run is invalid.");
    }
    remaining_ = bytes / sizeof(Record);
  }

  bool next(Record& value) {
    if (position_ == filled_) {
      if (remaining_ == 0) return false;
      filled_ = std::min<std::uint64_t>(remaining_, buffer_.size());
      stream_.read(reinterpret_cast<char*>(buffer_.data()),
                   filled_ * sizeof(Record));
      if (!stream_) Rcpp::stop("Massive graph run read failed.");
      remaining_ -= filled_;
      position_ = 0;
    }
    value = buffer_[position_++];
    return true;
  }

 private:
  std::ifstream stream_;
  std::uint64_t remaining_;
  std::size_t position_;
  std::size_t filled_;
  std::vector<Record> buffer_;
};

template <typename Record, typename Less, typename Consume>
void merge_graph_runs(const std::vector<std::string>& paths,
                      Less less, Consume consume) {
  struct Item { Record value; std::size_t run; };
  auto later = [&](const Item& left, const Item& right) {
    return less(right.value, left.value);
  };
  std::priority_queue<Item, std::vector<Item>, decltype(later)> heap(later);
  std::vector<std::unique_ptr<RunReader<Record>>> readers;
  readers.reserve(paths.size());
  for (const auto& path : paths) {
    readers.emplace_back(new RunReader<Record>(path));
    Record value{};
    if (readers.back()->next(value)) {
      heap.push({value, readers.size() - 1});
    }
  }
  std::uint64_t visited = 0;
  while (!heap.empty()) {
    const auto item = heap.top();
    heap.pop();
    consume(item.value);
    Record next{};
    if (readers[item.run]->next(next)) heap.push({next, item.run});
    if (++visited % 65536 == 0) Rcpp::checkUserInterrupt();
  }
}

template <typename Record, typename Less>
class GraphRunSorter {
 public:
  GraphRunSorter(std::string directory, std::string label,
                 std::size_t capacity, Less less,
                 std::vector<std::string> runs = {},
                 GraphDiskPeak* disk = nullptr)
      : directory_(std::move(directory)), label_(std::move(label)),
        capacity_(capacity), less_(less), runs_(std::move(runs)),
        disk_(disk) {
    if (capacity_ < 1) Rcpp::stop("Graph sort budget is too small.");
    buffer_.reserve(capacity_);
  }

  void push(const Record& value) {
    buffer_.push_back(value);
    if (buffer_.size() == capacity_) spill();
  }

  std::vector<std::string> finish() {
    spill();
    std::vector<Record>().swap(buffer_);
    return runs_;
  }

  std::vector<std::string> checkpoint() {
    spill();
    return runs_;
  }

 private:
  void spill() {
    if (buffer_.empty()) return;
    std::sort(buffer_.begin(), buffer_.end(), less_);
    const auto path = directory_ + "/" + label_ + "_run_" +
      std::to_string(runs_.size()) + ".bin";
    RunSink<Record> sink(path, disk_);
    for (const auto& value : buffer_) sink.push(value);
    sink.finish();
    runs_.push_back(path);
    buffer_.clear();
  }

  std::string directory_;
  std::string label_;
  std::size_t capacity_;
  Less less_;
  std::vector<Record> buffer_;
  std::vector<std::string> runs_;
  GraphDiskPeak* disk_;
};

template <typename Record, typename Less>
std::vector<std::string> reduce_graph_runs(
    std::vector<std::string> runs, const std::string& directory,
    const std::string& label, Less less,
    GraphDiskPeak* disk = nullptr,
    std::function<void(const std::vector<std::string>&)> checkpoint = {}) {
  int round = 0;
  const auto prefix = label + "_merge_";
  for (const auto& path : runs) {
    const auto name = std::filesystem::path(path).filename().string();
    if (name.rfind(prefix, 0) == 0) {
      const auto separator = name.find('_', prefix.size());
      if (separator == std::string::npos) {
        Rcpp::stop("Graph merge run name is invalid.");
      }
      round = std::max(round, std::stoi(name.substr(prefix.size(),
        separator - prefix.size())) + 1);
    }
  }
  while (runs.size() > 32) {
    std::unordered_map<std::string, std::uintmax_t> sizes;
    for (const auto& path : runs) {
      sizes.emplace(path, std::filesystem::file_size(path));
    }
    std::sort(runs.begin(), runs.end(),
      [&](const std::string& left, const std::string& right) {
        if (sizes.at(left) != sizes.at(right)) {
          return sizes.at(left) < sizes.at(right);
        }
        return left < right;
      });
    const auto group_size = std::min<std::size_t>(
      32, runs.size() - 31);
    const std::vector<std::string> group(runs.begin(),
                                          runs.begin() + group_size);
    const auto path = directory + "/" + prefix +
      std::to_string(round++) + "_0.bin";
    RunSink<Record> sink(path, disk);
    merge_graph_runs<Record>(group, less,
      [&](const Record& value) { sink.push(value); });
    sink.finish();
    std::vector<std::string> reduced(
      runs.begin() + group_size, runs.end());
    reduced.push_back(path);
    if (checkpoint) checkpoint(reduced);
    for (const auto& old : group) {
      if (disk) disk->remove(old);
      else std::filesystem::remove(old);
    }
    runs = std::move(reduced);
  }
  return runs;
}

double graph_sort_budget(std::uint64_t directed,
                         std::uint64_t pairs,
                         std::uint64_t vertices) {
  const double input = static_cast<double>(directed);
  const double unique = static_cast<double>(pairs);
  const double offsets = 8.0 * (vertices + 1);
  // At most 32 of 63 runs merge at once: the smallest group is <= 32/63
  // of its phase's payload. CSR writing overlaps row runs by 16 bytes/pair.
  return std::max({19.0 * input, 12.0 * input + 24.0 * unique,
                   40.0 * unique}) + offsets;
}

void check_graph_sort_space(const std::string& directory,
                            const GraphDiskPeak& disk,
                            double budget, bool discard_work) {
  std::error_code error;
  const auto available = std::filesystem::space(directory, error);
  if (error) Rcpp::stop("Cannot inspect graph sort disk capacity.");
  if (std::max(0.0, budget - disk.current()) >
      0.8 * static_cast<double>(available.available)) {
    if (discard_work) {
      std::filesystem::remove_all(directory, error);
    }
    Rcpp::stop("Graph remaining phases exceed the free-disk budget.");
  }
}

template <typename Record>
class GraphBucketWriter {
 public:
  GraphBucketWriter(const std::string& directory,
                    const std::string& label, std::size_t count,
                    GraphDiskPeak* disk)
      : disk_(disk), paths_(count), buffers_(count), created_(count),
        drained_(count), streams_(count) {
    for (std::size_t i = 0; i < count; ++i) {
      paths_[i] = directory + "/" + label + "_bucket_" +
        std::to_string(i) + ".bin";
    }
  }

  void push(std::size_t bucket, const Record& value) {
    if (drained_.at(bucket)) {
      Rcpp::stop("Graph bucket was already drained.");
    }
    auto& buffer = buffers_.at(bucket);
    buffer.push_back(value);
    if (buffer.size() == 1024) flush(bucket);
  }

  void flush_all() {
    for (std::size_t i = 0; i < paths_.size(); ++i) {
      flush(i);
      if (streams_[i]) {
        streams_[i]->close();
        if (!*streams_[i]) Rcpp::stop("Graph bucket close failed.");
        streams_[i].reset();
      }
    }
  }

  const std::string& path(std::size_t bucket) const {
    return paths_.at(bucket);
  }

  bool exists(std::size_t bucket) const { return created_.at(bucket); }

  template <typename Consume>
  void drain(std::size_t bucket, Consume consume) {
    flush(bucket);
    if (streams_[bucket]) {
      streams_[bucket]->close();
      if (!*streams_[bucket]) Rcpp::stop("Graph bucket close failed.");
      streams_[bucket].reset();
    }
    drained_[bucket] = true;
    if (!created_[bucket]) return;
    RunReader<Record> reader(paths_[bucket]);
    Record record{};
    while (reader.next(record)) consume(record);
    disk_->remove(paths_[bucket]);
  }

 private:
  void flush(std::size_t bucket) {
    auto& buffer = buffers_[bucket];
    if (buffer.empty()) return;
    const auto& path = paths_[bucket];
    if (!created_[bucket]) {
      streams_[bucket].reset(new std::ofstream(path, std::ios::binary));
      if (!*streams_[bucket]) Rcpp::stop("Cannot create graph bucket.");
      disk_->add(path);
      created_[bucket] = true;
    }
    const auto bytes = buffer.size() * sizeof(Record);
    streams_[bucket]->write(reinterpret_cast<const char*>(buffer.data()),
                            bytes);
    if (!*streams_[bucket]) Rcpp::stop("Graph bucket write failed.");
    disk_->written(path, bytes);
    buffer.clear();
  }

  GraphDiskPeak* disk_;
  std::vector<std::string> paths_;
  std::vector<std::vector<Record>> buffers_;
  std::vector<bool> created_;
  std::vector<bool> drained_;
  std::vector<std::unique_ptr<std::ofstream>> streams_;
};

class GraphBucketWork {
 public:
  explicit GraphBucketWork(const std::string& prefix)
      : prefix_(prefix), directory_(prefix + ".work") {
    if (!std::filesystem::create_directory(directory_)) {
      Rcpp::stop("Fuzzy graph work directory already exists.");
    }
  }

  ~GraphBucketWork() {
    if (finished_) return;
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
    for (const auto& suffix : {".offsets.u64.part",
                               ".indices.u32.part",
                               ".weights.f32.part",
                               ".offsets.u64", ".indices.u32",
                               ".weights.f32"}) {
      std::filesystem::remove(prefix_ + suffix, error);
    }
  }

  const std::string& directory() const { return directory_; }
  void finish() { finished_ = true; }

 private:
  std::string prefix_;
  std::string directory_;
  bool finished_ = false;
};

void normalize_bucketed_affinity(const std::string& path,
                                 std::uint64_t edges,
                                 double mass) {
  std::fstream stream(path, std::ios::in | std::ios::out |
                           std::ios::binary);
  if (!stream) Rcpp::stop("Cannot normalize t-SNE affinities.");
  std::vector<float> values(65536);
  for (std::uint64_t first = 0; first < edges;) {
    const auto count = std::min<std::uint64_t>(
      values.size(), edges - first);
    const auto position = static_cast<std::streamoff>(first * 4);
    stream.seekg(position);
    stream.read(reinterpret_cast<char*>(values.data()), count * 4);
    if (!stream) Rcpp::stop("Cannot read t-SNE affinities.");
    for (std::uint64_t i = 0; i < count; ++i) {
      values[i] = static_cast<float>(values[i] * (0.5 / mass));
    }
    stream.seekp(position);
    stream.write(reinterpret_cast<const char*>(values.data()), count * 4);
    if (!stream) Rcpp::stop("Cannot write t-SNE affinities.");
    first += count;
    Rcpp::checkUserInterrupt();
  }
  stream.close();
  if (!stream) Rcpp::stop("Cannot close t-SNE affinities.");
}

List bucketed_symmetrize_graph(KnnGraphSource& graph,
                               const std::string& output_prefix,
                               const std::string& indices_path,
                               const std::string& values_path,
                               std::size_t capacity, bool tsne,
                               double perplexity, int n_threads,
                               SEXP release_input) {
  GraphBucketWork work(output_prefix);
  const auto& directory = work.directory();
  const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(
    std::min<std::uint64_t>(64, graph.vertices()),
    std::max<std::uint64_t>(2,
      (graph.edges() + capacity - 1) / capacity)));
  const auto width = (graph.vertices() + count - 1) / count;
  const auto bucket = [&](std::uint64_t row) {
    return std::min<std::size_t>((row - 1) / width, count - 1);
  };
  GraphDiskPeak disk;
  const auto input_bytes = checked_product(graph.edges(), 8);
  disk.external(input_bytes);
  std::vector<std::unique_ptr<GraphRunSorter<PairRecord, PairLess>>>
    pair_sorters;
  pair_sorters.reserve(count);
  const auto pair_capacity = std::max<std::size_t>(1, capacity / count);
  for (std::size_t i = 0; i < count; ++i) {
    pair_sorters.emplace_back(new GraphRunSorter<PairRecord, PairLess>(
      directory, "pairs_sort_" + std::to_string(i), pair_capacity,
      PairLess{}, {}, &disk));
  }
  std::vector<std::uint64_t> reverse_edges(count, 0);
  const auto mean = tsne ? 0.0 : umap_mean_scan(indices_path,
    values_path, graph.vertices(), graph.k(), 0, "0", R_NilValue, 1);
  const auto block_rows = std::max(1, 65536 / graph.k());
  std::vector<float> distances(65536 + graph.k());
  std::vector<std::uint32_t> targets(65536 + graph.k());
  std::vector<float> probabilities(tsne ? distances.size() : 0);
  const bool consume = Rf_isFunction(release_input);
  std::uint64_t remaining = graph.vertices();
  for (std::uint64_t first = 1; first <= graph.vertices();) {
    if (consume) first = remaining - std::min<std::uint64_t>(
      block_rows, remaining) + 1;
    const auto rows = std::min<std::uint64_t>(
      block_rows, consume ? remaining : graph.vertices() - first + 1);
    graph.visit(first, rows,
      [&](std::uint64_t, std::uint32_t target, float distance,
          std::uint64_t row, int col) {
        const auto pos = row * graph.k() + col;
        distances[pos] = distance;
        targets[pos] = target;
      });
    if (tsne) {
      parallel_rows(rows, n_threads,
        [&](std::uint64_t begin, std::uint64_t end, int) {
          for (auto row = begin; row < end; ++row) {
            const auto pos = row * graph.k();
            tsne_row_probabilities_float(distances.data() + pos,
              graph.k(), perplexity, probabilities.data() + pos);
          }
        });
    }
    for (std::uint64_t row = 0; row < rows; ++row) {
      const auto source = first + row;
      const auto pos = row * graph.k();
      UmapMembershipScale scale{};
      if (!tsne) scale = umap_membership_scale(
        distances.data() + pos, graph.k(), mean);
      for (int col = 0; col < graph.k(); ++col) {
        const auto target = targets[pos + col];
        const auto distance = distances[pos + col];
        const float weight = tsne ? probabilities[pos + col] :
          (distance <= scale.rho ? 1.0f :
            std::exp(-(distance - scale.rho) / scale.sigma));
        const auto low = static_cast<std::uint32_t>(
          std::min<std::uint64_t>(source, target));
        const auto high = static_cast<std::uint32_t>(
          std::max<std::uint64_t>(source, target));
        if (bucket(high) != bucket(low)) ++reverse_edges[bucket(high)];
        pair_sorters[bucket(low)]->push({low, high, std::copysign(weight,
          source == low ? 1.0f : -1.0f)});
      }
    }
    if (consume) {
      const auto bytes = checked_product(
        checked_product(first - 1, graph.k()), 4);
      std::filesystem::resize_file(indices_path, bytes);
      std::filesystem::resize_file(values_path, bytes);
      disk.external(2 * bytes);
      remaining = first - 1;
      if (!remaining) break;
    } else {
      first += rows;
    }
    Rcpp::checkUserInterrupt();
  }
  std::vector<std::vector<std::string>> pair_runs(count);
  for (std::size_t i = 0; i < count; ++i) {
    pair_runs[i] = pair_sorters[i]->finish();
  }
  pair_sorters.clear();
  double released_input_bytes = 0;
  if (consume) {
    graph.close();
    Rcpp::Function release(release_input);
    released_input_bytes = Rcpp::as<double>(release());
    if (released_input_bytes != input_bytes) {
      Rcpp::stop("Consumed KNN byte count is invalid.");
    }
  }
  double max_pair_bytes = 0;
  double max_reverse_bytes = 0;
  std::vector<double> pair_bucket_bytes(count, 0);
  for (std::size_t i = 0; i < count; ++i) {
    for (const auto& path : pair_runs[i]) {
      pair_bucket_bytes[i] += static_cast<double>(
        std::filesystem::file_size(path));
    }
    max_pair_bytes = std::max(max_pair_bytes, pair_bucket_bytes[i]);
    max_reverse_bytes = std::max(max_reverse_bytes,
      12.0 * static_cast<double>(reverse_edges[i]));
  }
  const double vertices_bytes = 8.0 * (graph.vertices() + 1);
  const double disk_budget = std::min(
    36.0 * graph.edges() + vertices_bytes,
    20.0 * graph.edges() +
      6.0 * (max_pair_bytes + max_reverse_bytes) + vertices_bytes);
  check_graph_sort_space(directory, disk, disk_budget, false);
  GraphBucketWriter<RowRecord> row_buckets(directory, "rows", count,
                                            &disk);
  const auto offsets_path = output_prefix + ".offsets.u64";
  const auto output_indices = output_prefix + ".indices.u32";
  const auto weights_path = output_prefix + ".weights.f32";
  RunSink<std::uint64_t> offsets(offsets_path + ".part", &disk);
  RunSink<std::uint32_t> indices(output_indices + ".part", &disk);
  RunSink<float> weights(weights_path + ".part", &disk);
  offsets.push(0);
  std::uint64_t source = 1;
  std::uint64_t edges = 0;
  std::uint64_t degree = 0;
  std::uint64_t max_degree = 0;
  std::uint32_t last_target = 0;
  double total_mass = 0;
  for (std::size_t i = 0; i < count; ++i) {
    GraphRunSorter<RowRecord, RowLess> rows(directory,
      "rows_sort_" + std::to_string(i), capacity, RowLess{}, {}, &disk);
    if (!pair_runs[i].empty()) {
      check_graph_sort_space(directory, disk,
        disk.current() + pair_bucket_bytes[i],
        false);
      auto runs = reduce_graph_runs<PairRecord>(pair_runs[i], directory,
        "pairs_sort_" + std::to_string(i), PairLess{}, &disk);
      PairRecord current{};
      float forward = 0.0f;
      float backward = 0.0f;
      std::uint32_t seen = 0;
      auto emit = [&]() {
        if (!seen) return;
        const float weight = tsne ? forward + backward :
          std::min(1.0f, forward + backward - forward * backward);
        if (weight <= 0.0f) return;
        rows.push({current.low, current.high, weight});
        if (bucket(current.high) == i) {
          rows.push({current.high, current.low, weight});
        } else {
          row_buckets.push(bucket(current.high),
            {current.high, current.low, weight});
        }
        if (tsne) total_mass += weight;
      };
      merge_graph_runs<PairRecord>(runs, PairLess{},
        [&](const PairRecord& edge) {
          if (seen && (edge.low != current.low ||
                       edge.high != current.high)) {
            emit();
            seen = 0;
            forward = backward = 0.0f;
          }
          current = edge;
          const bool reverse = std::signbit(edge.weight);
          const auto bit = reverse ? 2u : 1u;
          if (seen & bit) {
            Rcpp::stop("Graph has duplicate directed pair edges.");
          }
          seen |= bit;
          if (reverse) backward = std::fabs(edge.weight);
          else forward = edge.weight;
        });
      emit();
      for (const auto& path : runs) disk.remove(path);
    }
    row_buckets.drain(i, [&](const RowRecord& edge) {
      rows.push(edge);
    });
    auto row_runs = rows.finish();
    row_runs = reduce_graph_runs<RowRecord>(row_runs, directory,
      "rows_sort_" + std::to_string(i), RowLess{}, &disk);
    merge_graph_runs<RowRecord>(row_runs, RowLess{},
      [&](const RowRecord& edge) {
        if (edge.source < source || edge.source > graph.vertices() ||
            edge.target < 1 || edge.target > graph.vertices() ||
            edge.source == edge.target) {
          Rcpp::stop("Fuzzy graph row sort produced invalid edges.");
        }
        while (source < edge.source) {
          offsets.push(edges);
          max_degree = std::max(max_degree, degree);
          ++source;
          degree = 0;
          last_target = 0;
        }
        if (edge.target <= last_target) {
          Rcpp::stop("Fuzzy graph row has duplicate edges.");
        }
        indices.push(edge.target);
        weights.push(edge.weight);
        ++edges;
        ++degree;
        last_target = edge.target;
      });
    for (const auto& path : row_runs) disk.remove(path);
    const auto end_row = std::min<std::uint64_t>(
      graph.vertices(), (i + 1) * width);
    while (source <= end_row) {
      offsets.push(edges);
      max_degree = std::max(max_degree, degree);
      ++source;
      degree = 0;
      last_target = 0;
    }
    Rcpp::checkUserInterrupt();
  }
  if (tsne && (!std::isfinite(total_mass) || total_mass <= 0.0)) {
    Rcpp::stop("t-SNE affinity normalization failed.");
  }
  row_buckets.flush_all();
  offsets.finish();
  indices.finish();
  weights.finish();
  if (tsne) normalize_bucketed_affinity(weights_path + ".part",
                                        edges, total_mass);
  if (!std::filesystem::is_empty(directory)) {
    Rcpp::stop("Bucketed graph sort left work files.");
  }
  if (disk.peak() > disk_budget) {
    Rcpp::stop("Bucketed graph sort exceeded its disk budget.");
  }
  for (const auto& path : {offsets_path, output_indices, weights_path}) {
    std::filesystem::rename(path + ".part", path);
  }
  std::filesystem::remove(directory);
  work.finish();
  return List::create(
    Rcpp::Named("n_vertices") = static_cast<double>(graph.vertices()),
    Rcpp::Named("n_edges") = static_cast<double>(edges),
    Rcpp::Named("max_degree") = static_cast<double>(max_degree),
    Rcpp::Named("disk_budget_bytes") = disk_budget,
    Rcpp::Named("peak_sort_payload_bytes") = disk.peak(),
    Rcpp::Named("peak_with_knn_bytes") = disk.peak_with_external(),
    Rcpp::Named("released_input_bytes") = released_input_bytes);
}

}  // namespace

// [[Rcpp::export]]
List massive_symmetrize_graph_cpp(std::string indices_path,
                                 std::string values_path,
                                 double n_vertices, int k,
                                 std::string output_prefix,
                                 double memory_limit_bytes,
                                 std::string method,
                                 double perplexity,
                                 int n_threads,
                                 bool resume,
                                 double completed_rows,
                                 Rcpp::CharacterVector resume_runs,
                                 std::string resume_phase,
                                 double resume_mass,
                                 int checkpoint_every,
                                 SEXP progress,
                                 SEXP release_input = R_NilValue) {
  require_little_endian();
  if (!Rf_isNull(release_input) && !Rf_isFunction(release_input)) {
    Rcpp::stop("KNN release callback must be a function or NULL.");
  }
  const bool tsne = method == "tsne" || method == "tsne_bucketed";
  const bool bucketed = method == "umap_bucketed" ||
    method == "tsne_bucketed";
  if (!tsne && method != "umap" && !bucketed) {
    Rcpp::stop("Unsupported graph sort method.");
  }
  if (tsne && (!std::isfinite(perplexity) || perplexity <= 0.0 ||
               perplexity > k)) {
    Rcpp::stop("t-SNE perplexity must be positive and at most k.");
  }
  if (n_threads < 1 || n_threads > 256 ||
      (!tsne && n_threads != 1)) {
    Rcpp::stop("Invalid graph-sorting worker count.");
  }
  KnnGraphSource graph(indices_path, values_path, n_vertices,
                       k, "stream", tsne || bucketed ?
                       "distance" : "weight");
  if (!std::isfinite(memory_limit_bytes) ||
      memory_limit_bytes < 64.0 * 1024 * 1024) {
    Rcpp::stop("Fuzzy graph sorting needs at least 64 MB RAM budget.");
  }
  const auto capacity = static_cast<std::size_t>(
    std::min(memory_limit_bytes * 0.25, 512.0 * 1024 * 1024) /
    sizeof(PairRecord));
  if (bucketed) {
    if (resume || completed_rows != 0 || resume_runs.size() ||
        resume_phase != "pairs_input" || resume_mass != 0 ||
        Rf_isFunction(progress)) {
      Rcpp::stop("Bucketed graph sort does not support checkpointing.");
    }
    for (const auto& suffix : {".offsets.u64", ".indices.u32",
                               ".weights.f32"}) {
      if (std::filesystem::exists(output_prefix + suffix) ||
          std::filesystem::exists(output_prefix + suffix + ".part")) {
        Rcpp::stop("Fuzzy graph output or partial file already exists.");
      }
    }
    return bucketed_symmetrize_graph(graph, output_prefix,
      indices_path, values_path, capacity, tsne, perplexity, n_threads,
      release_input);
  }
  if (Rf_isFunction(release_input)) {
    Rcpp::stop("KNN consumption requires bucketed graph sorting.");
  }
  const auto directory = output_prefix + ".work";
  const auto offsets_path = output_prefix + ".offsets.u64";
  const auto indices_out = output_prefix + ".indices.u32";
  const auto weights_out = output_prefix + ".weights.f32";
  for (const auto& path : {offsets_path, indices_out, weights_out}) {
    if (std::filesystem::exists(path) ||
        (!resume && std::filesystem::exists(path + ".part"))) {
      Rcpp::stop("Fuzzy graph output or partial file already exists.");
    }
  }
  const auto rows_per_block = std::max(1, 65536 / k);
  const bool resume_reduced = resume_phase == "pairs_reduced";
  const bool resume_reduced_rows = resume_phase == "rows_reduced";
  const bool resume_rows = resume_phase == "rows_generated" ||
    resume_reduced_rows;
  if (!std::isfinite(completed_rows) || completed_rows < 0 ||
      completed_rows > graph.vertices() ||
      completed_rows != std::floor(completed_rows) ||
      (completed_rows != graph.vertices() &&
       static_cast<std::uint64_t>(completed_rows) % rows_per_block != 0) ||
      (resume_phase != "pairs_input" && !resume_reduced &&
       !resume_rows) ||
      ((resume_reduced || resume_rows) && (!resume ||
       completed_rows != graph.vertices() || resume_runs.size() == 0)) ||
      !std::isfinite(resume_mass) || resume_mass < 0 ||
      (!resume_rows && resume_mass != 0) ||
      (resume_rows && tsne && resume_mass <= 0) ||
      (resume && !Rf_isFunction(progress)) ||
      (Rf_isFunction(progress) && checkpoint_every < 1) ||
      (!Rf_isFunction(progress) && checkpoint_every != 0)) {
    Rcpp::stop("Invalid graph sort checkpoint controls.");
  }
  std::vector<std::string> saved_runs;
  for (R_xlen_t i = 0; i < resume_runs.size(); ++i) {
    const auto saved = Rcpp::as<std::string>(resume_runs[i]);
    const auto name = std::filesystem::path(saved).filename().string();
    const auto expected = (resume_reduced || resume_rows) ?
      directory + "/" + name :
      directory + "/pairs_run_" + std::to_string(i) + ".bin";
    if (resume_reduced &&
        ((name.rfind("pairs_merge_", 0) != 0 &&
          name.rfind("pairs_run_", 0) != 0) ||
         name.size() < 5 || name.substr(name.size() - 4) != ".bin")) {
      Rcpp::stop("Reduced graph run path is invalid.");
    }
    if (resume_phase == "rows_generated" &&
        (name.rfind("rows_run_", 0) != 0 ||
         name.size() < 5 || name.substr(name.size() - 4) != ".bin")) {
      Rcpp::stop("Row graph run path is invalid.");
    }
    if (resume_reduced_rows &&
        ((name.rfind("rows_merge_", 0) != 0 &&
          name.rfind("rows_run_", 0) != 0) ||
         name.size() < 5 || name.substr(name.size() - 4) != ".bin")) {
      Rcpp::stop("Reduced row run path is invalid.");
    }
    std::error_code path_error;
    if (!std::filesystem::equivalent(saved, expected, path_error) ||
        path_error) {
      Rcpp::stop("Graph sort checkpoint run path is invalid.");
    }
    saved_runs.push_back(
      std::filesystem::path(expected).generic_string());
  }
  const std::unordered_set<std::string> retained(
    saved_runs.begin(), saved_runs.end());
  if (resume) {
    if (!std::filesystem::is_directory(directory)) {
      if (completed_rows != 0 || !saved_runs.empty() ||
          !std::filesystem::create_directory(directory)) {
        Rcpp::stop("Graph sort work directory is missing.");
      }
    }
    for (const auto& entry :
         std::filesystem::directory_iterator(directory)) {
      const auto path = entry.path().generic_string();
      if (retained.count(path) != 0) continue;
      const auto name = entry.path().filename().string();
      if (!entry.is_regular_file() ||
          (name.rfind("pairs_run_", 0) != 0 &&
           name.rfind("pairs_merge_", 0) != 0 &&
           name.rfind("rows_run_", 0) != 0 &&
           name.rfind("rows_merge_", 0) != 0)) {
        Rcpp::stop("Graph sort work directory contains unknown files.");
      }
      std::filesystem::remove(path);
    }
    for (const auto& path : {offsets_path, indices_out, weights_out}) {
      std::filesystem::remove(path + ".part");
    }
  } else if (!std::filesystem::create_directory(directory)) {
    Rcpp::stop("Fuzzy graph work directory already exists.");
  }
  GraphDiskPeak disk;
  for (const auto& entry :
       std::filesystem::directory_iterator(directory)) {
    disk.add(entry.path().generic_string());
  }
  std::vector<std::string> row_runs = resume_rows ? saved_runs :
    std::vector<std::string>{};
  double total_mass = resume_mass;
  double disk_budget = 0;
  if (!resume_rows) {
  GraphRunSorter<PairRecord, PairLess> pairs(
    directory, "pairs", capacity, PairLess{}, saved_runs, &disk);
  const auto block_items = static_cast<std::size_t>(rows_per_block) * k;
  std::vector<float> distances(tsne ? block_items : 0);
  std::vector<float> probabilities(tsne ? block_items : 0);
  std::vector<std::uint32_t> targets(tsne ? block_items : 0);
  std::uint64_t block = 0;
  for (std::uint64_t first =
       static_cast<std::uint64_t>(completed_rows) + 1;
       first <= graph.vertices();) {
    const auto count = std::min<std::uint64_t>(
      rows_per_block, graph.vertices() - first + 1);
    graph.visit(first, count,
      [&](std::uint64_t source, std::uint32_t target,
          float weight, std::uint64_t row, int column) {
      if (tsne) {
        const auto pos = static_cast<std::size_t>(row) * k + column;
        distances[pos] = weight;
        targets[pos] = target;
        return;
      }
      const auto low = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(source, target));
      const auto high = static_cast<std::uint32_t>(
        std::max<std::uint64_t>(source, target));
      pairs.push({low, high, std::copysign(weight,
        source == low ? 1.0f : -1.0f)});
    });
    if (tsne) {
      parallel_rows(count, n_threads,
        [&](std::uint64_t begin, std::uint64_t end, int) {
        for (auto row = begin; row < end; ++row) {
          const auto pos = static_cast<std::size_t>(row) * k;
          tsne_row_probabilities_float(distances.data() + pos, k,
                                       perplexity, probabilities.data() + pos);
        }
      });
      for (std::uint64_t row = 0; row < count; ++row) {
        const auto source = first + row;
        const auto pos = static_cast<std::size_t>(row) * k;
        for (int j = 0; j < k; ++j) {
          const auto low = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(source, targets[pos + j]));
          const auto high = static_cast<std::uint32_t>(
            std::max<std::uint64_t>(source, targets[pos + j]));
          pairs.push({low, high, std::copysign(
            probabilities[pos + j],
            source == low ? 1.0f : -1.0f)});
        }
      }
    }
    first += count;
    if (Rf_isFunction(progress) &&
        (++block % checkpoint_every == 0 || first > graph.vertices())) {
      Rcpp::Function callback(progress);
      callback(static_cast<double>(first - 1),
               Rcpp::wrap(pairs.checkpoint()));
    }
    Rcpp::checkUserInterrupt();
  }
  const auto original_runs = pairs.finish();
  if (Rf_isFunction(progress) && completed_rows == graph.vertices()) {
    Rcpp::Function callback(progress);
    callback(completed_rows, Rcpp::wrap(original_runs));
  }
  std::function<void(const std::vector<std::string>&)> pair_checkpoint;
  if (Rf_isFunction(progress)) {
    pair_checkpoint = [&](const std::vector<std::string>& runs) {
      Rcpp::Function callback(progress);
      callback(static_cast<double>(graph.vertices()),
               Rcpp::wrap(runs), "pairs_reduced");
    };
  }
  auto pair_runs = reduce_graph_runs<PairRecord>(
    original_runs, directory, "pairs", PairLess{}, &disk,
    pair_checkpoint);
  std::uint64_t unique_pairs = 0;
  PairRecord previous_pair{};
  bool have_pair = false;
  float first_weight = 0.0f;
  float second_weight = 0.0f;
  std::uint32_t directions = 0;
  std::uint64_t counted_edges = 0;
  std::uint64_t next_count_check = 65536;
  int next_count_report = 10;
  auto count_pair = [&]() {
    if (have_pair && first_weight + second_weight > 0.0f) {
      ++unique_pairs;
    }
  };
  merge_graph_runs<PairRecord>(pair_runs, PairLess{},
    [&](const PairRecord& edge) {
      if (!have_pair || edge.low != previous_pair.low ||
          edge.high != previous_pair.high) {
        count_pair();
        first_weight = second_weight = 0.0f;
        directions = 0;
      }
      const bool backward_edge = std::signbit(edge.weight);
      const auto bit = backward_edge ? 2u : 1u;
      if ((directions & bit) != 0) {
        Rcpp::stop("Graph has duplicate directed pair edges.");
      }
      directions |= bit;
      if (backward_edge) second_weight = std::fabs(edge.weight);
      else first_weight = edge.weight;
      previous_pair = edge;
      have_pair = true;
      ++counted_edges;
      if (counted_edges == next_count_check) {
        report_graph_progress("Distinct pair count", counted_edges,
                              graph.edges(), next_count_report);
        next_count_check += 65536;
      }
    });
  count_pair();
  report_graph_progress("Distinct pair count", counted_edges,
                        graph.edges(), next_count_report);
  disk_budget = graph_sort_budget(graph.edges(), unique_pairs,
                                  graph.vertices());
  check_graph_sort_space(directory, disk, std::max({
    12.0 * graph.edges() + 24.0 * unique_pairs,
    40.0 * unique_pairs}) + 8.0 * (graph.vertices() + 1),
    !Rf_isFunction(progress));
  Rcpp::Rcout << "  Graph pair sorting: 100%\n";
  GraphRunSorter<RowRecord, RowLess> rows(
    directory, "rows", capacity, RowLess{}, {}, &disk);
  PairRecord current{};
  float forward = 0.0f;
  float backward = 0.0f;
  std::uint32_t seen = 0;
  auto emit_pair = [&]() {
    if (seen == 0) return;
    const float weight = tsne ? forward + backward :
      std::min(1.0f, forward + backward - forward * backward);
    if (weight > 0.0f) {
      rows.push({current.low, current.high, weight});
      rows.push({current.high, current.low, weight});
      if (tsne) total_mass += weight;
    }
  };
  merge_graph_runs<PairRecord>(pair_runs, PairLess{},
    [&](const PairRecord& edge) {
      if (seen != 0 && (edge.low != current.low ||
                        edge.high != current.high)) {
        emit_pair();
        seen = 0;
        forward = backward = 0.0f;
      }
      current = edge;
      const bool backward_edge = std::signbit(edge.weight);
      const auto bit = backward_edge ? 2u : 1u;
      if ((seen & bit) != 0) {
        Rcpp::stop("Graph has duplicate directed pair edges.");
      }
      seen |= bit;
      if (backward_edge) backward = std::fabs(edge.weight);
      else forward = edge.weight;
    });
  emit_pair();
  if (tsne && (!std::isfinite(total_mass) || total_mass <= 0.0)) {
    Rcpp::stop("t-SNE affinity normalization failed.");
  }
  row_runs = rows.finish();
  if (row_runs.empty()) Rcpp::stop("Graph has no positive edges.");
  if (Rf_isFunction(progress)) {
    Rcpp::Function callback(progress);
    callback(static_cast<double>(graph.vertices()),
             Rcpp::wrap(row_runs), "rows_generated", total_mass);
  }
  for (const auto& path : pair_runs) disk.remove(path);
  for (const auto& path : original_runs) disk.remove(path);
  }
  if (row_runs.empty()) Rcpp::stop("Graph has no positive edges.");
  if (resume_rows) {
    std::uint64_t row_bytes = 0;
    for (const auto& path : row_runs) {
      row_bytes += std::filesystem::file_size(path);
    }
    if (row_bytes % (2 * sizeof(RowRecord)) != 0) {
      Rcpp::stop("Graph row runs have invalid pair count.");
    }
    disk_budget = graph_sort_budget(graph.edges(),
      row_bytes / (2 * sizeof(RowRecord)), graph.vertices());
    const double unique = static_cast<double>(
      row_bytes / (2 * sizeof(RowRecord)));
    check_graph_sort_space(directory, disk,
      40.0 * unique + 8.0 * (graph.vertices() + 1), false);
  }
  std::function<void(const std::vector<std::string>&)> row_checkpoint;
  if (Rf_isFunction(progress)) {
    row_checkpoint = [&](const std::vector<std::string>& runs) {
      Rcpp::Function callback(progress);
      callback(static_cast<double>(graph.vertices()),
               Rcpp::wrap(runs), "rows_reduced", total_mass);
    };
  }
  row_runs = reduce_graph_runs<RowRecord>(
    row_runs, directory, "rows", RowLess{}, &disk,
    row_checkpoint);
  Rcpp::Rcout << "  Graph symmetrization and row sorting: 100%\n";
  RunSink<std::uint64_t> offsets(offsets_path + ".part", &disk);
  RunSink<std::uint32_t> indices(indices_out + ".part", &disk);
  RunSink<float> weights(weights_out + ".part", &disk);
  offsets.push(0);
  std::uint64_t source_row = 1;
  std::uint64_t edge_count = 0;
  std::uint64_t degree = 0;
  std::uint64_t max_degree = 0;
  std::uint32_t last_target = 0;
  merge_graph_runs<RowRecord>(row_runs, RowLess{},
    [&](const RowRecord& edge) {
      if (edge.source < source_row || edge.source > graph.vertices() ||
          edge.target < 1 || edge.target > graph.vertices() ||
          edge.source == edge.target) {
        Rcpp::stop("Fuzzy graph row sort produced invalid edges.");
      }
      while (source_row < edge.source) {
        offsets.push(edge_count);
        max_degree = std::max(max_degree, degree);
        ++source_row;
        degree = 0;
        last_target = 0;
      }
      if (edge.target <= last_target) {
        Rcpp::stop("Fuzzy graph row has duplicate edges.");
      }
      indices.push(edge.target);
      weights.push(tsne ?
        static_cast<float>(edge.weight * (0.5 / total_mass)) :
        edge.weight);
      ++degree;
      ++edge_count;
      last_target = edge.target;
    });
  while (source_row <= graph.vertices()) {
    offsets.push(edge_count);
    max_degree = std::max(max_degree, degree);
    ++source_row;
    degree = 0;
  }
  offsets.finish();
  indices.finish();
  weights.finish();
  for (const auto& path : row_runs) disk.remove(path);
  for (const auto& path : {offsets_path, indices_out, weights_out}) {
    std::filesystem::rename(path + ".part", path);
  }
  for (const auto& entry :
       std::filesystem::directory_iterator(directory)) {
    Rcpp::stop("Graph sort left a work file: " +
               entry.path().filename().string());
  }
  std::filesystem::remove(directory);
  Rcpp::Rcout << "  Graph CSR output: 100%\n";
  return List::create(
    Rcpp::Named("n_vertices") = static_cast<double>(graph.vertices()),
    Rcpp::Named("n_edges") = static_cast<double>(edge_count),
    Rcpp::Named("max_degree") = static_cast<double>(max_degree),
    Rcpp::Named("disk_budget_bytes") = disk_budget,
    Rcpp::Named("peak_sort_payload_bytes") = disk.peak());
}

// [[Rcpp::export]]
List massive_contract_louvain_cpp(std::string offsets_path,
                                  std::string indices_path,
                                  std::string weights_path,
                                  std::string membership_path,
                                  double n_vertices,
                                  int n_communities,
                                  std::string output_path,
                                  double memory_limit_bytes,
                                  int chunk_rows,
                                  double edge_budget,
                                  bool contracted) {
  if (n_communities < 1 || chunk_rows < 1 ||
      !std::isfinite(memory_limit_bytes) ||
      memory_limit_bytes < 64.0 * 1024 * 1024) {
    Rcpp::stop("Invalid Louvain contraction controls.");
  }
  const auto maximum = whole_number(edge_budget, "edge_budget");
  CsrGraphSource graph(offsets_path, indices_path, weights_path,
                       n_vertices, "stream", contracted);
  GraphFile labels(membership_path,
                   checked_product(graph.vertices(), 4), "mmap");
  const std::string work = output_path + ".work";
  if (std::filesystem::exists(output_path) ||
      std::filesystem::exists(output_path + ".part") ||
      std::filesystem::exists(work)) {
    Rcpp::stop("Louvain contraction output or work already exists.");
  }
  if (!std::filesystem::create_directory(work)) {
    Rcpp::stop("Cannot create Louvain contraction work directory.");
  }
  const auto capacity = static_cast<std::size_t>(
    std::min(memory_limit_bytes * 0.25, 512.0 * 1024 * 1024) /
    sizeof(CoarseRecord));
  GraphRunSorter<CoarseRecord, CoarseLess> sorter(
    work, "coarse", capacity, CoarseLess{});
  auto label_at = [&](std::uint64_t row) {
    const auto label = read_u32(labels.read((row - 1) * 4, 4));
    if (label < 1 || label > static_cast<std::uint32_t>(n_communities)) {
      Rcpp::stop("First-level Louvain label is out of range.");
    }
    return label;
  };
  const std::vector<std::string> inputs = {
    offsets_path, indices_path, weights_path, membership_path};
  std::vector<std::string> stamps;
  std::vector<std::uint64_t> sizes;
  for (const auto& path : inputs) {
    stamps.push_back(file_stamp(path));
    sizes.push_back(std::filesystem::file_size(path));
  }
  std::uint64_t original_pairs = 0;
  for (std::uint64_t first = 1; first <= graph.vertices();) {
    const auto rows = graph.bounded_rows(first, chunk_rows, maximum);
    graph.visit(first, rows,
      [&](std::uint64_t source, std::uint32_t target, double weight,
          std::uint64_t) {
        if (source > target || (!contracted && source == target)) return;
        const auto left = label_at(source);
        const auto right = label_at(target);
        sorter.push({std::min(left, right), std::max(left, right),
                     source == target ? weight / 2.0 : weight});
        ++original_pairs;
      });
    first += rows;
    Rcpp::checkUserInterrupt();
  }
  auto runs = reduce_graph_runs<CoarseRecord>(
    sorter.finish(), work, "coarse", CoarseLess{});
  RunSink<CoarseRecord> output(output_path + ".part");
  CoarseRecord current{};
  std::uint64_t edge_count = 0;
  double total_weight = 0.0;
  auto emit = [&]() {
    if (current.weight <= 0.0) return;
    if (!std::isfinite(current.weight)) {
      Rcpp::stop("Contracted Louvain weight is not finite.");
    }
    output.push(current);
    total_weight += current.weight;
    ++edge_count;
  };
  merge_graph_runs<CoarseRecord>(runs, CoarseLess{},
    [&](const CoarseRecord& edge) {
      if (edge.low != current.low || edge.high != current.high) {
        emit();
        current = edge;
      } else {
        current.weight += edge.weight;
      }
    });
  emit();
  output.finish();
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    std::error_code error;
    const auto bytes = std::filesystem::file_size(inputs[index], error);
    if (error || bytes != sizes[index] ||
        file_stamp(inputs[index]) != stamps[index]) {
      Rcpp::stop("Louvain contraction inputs changed during the scan.");
    }
  }
  std::filesystem::rename(output_path + ".part", output_path);
  for (const auto& path : runs) std::filesystem::remove(path);
  std::filesystem::remove(work);
  return List::create(
    Rcpp::Named("path") = output_path,
    Rcpp::Named("n_vertices") = n_communities,
    Rcpp::Named("n_edges") = static_cast<double>(edge_count),
    Rcpp::Named("original_edge_pairs") =
      static_cast<double>(original_pairs),
    Rcpp::Named("total_edge_weight") = total_weight,
    Rcpp::Named("backend") = "cpu");
}

// [[Rcpp::export]]
List massive_coarse_csr_cpp(std::string input, double n_vertices,
                            double n_edges, std::string output,
                            double memory_limit_bytes) {
  require_little_endian();
  const auto vertices = whole_number(n_vertices, "n_vertices");
  const auto pairs = whole_number(n_edges, "n_edges");
  if (vertices < 2 || vertices > INT32_MAX || pairs < 1 ||
      memory_limit_bytes < 64.0 * 1024 * 1024 ||
      !std::isfinite(memory_limit_bytes)) {
    Rcpp::stop("Invalid contracted Louvain CSR controls.");
  }
  std::error_code error;
  if (std::filesystem::file_size(input, error) !=
      checked_product(pairs, sizeof(CoarseRecord)) || error) {
    Rcpp::stop("Contracted Louvain edge file size is invalid.");
  }
  const auto input_stamp = file_stamp(input);
  const std::string offsets = output + ".offsets.u64";
  const std::string indices = output + ".indices.u32";
  const std::string weights = output + ".weights.f64";
  const std::string work = output + ".work";
  for (const auto& path : {offsets, indices, weights,
                           offsets + ".part", indices + ".part",
                           weights + ".part", work}) {
    if (std::filesystem::exists(path)) {
      Rcpp::stop("Contracted CSR output or work already exists.");
    }
  }
  const auto available = std::filesystem::space(
    std::filesystem::path(output).parent_path(), error);
  if (error || 64.0 * pairs + 8.0 * vertices >
      0.8 * static_cast<double>(available.available)) {
    Rcpp::stop("Contracted CSR exceeds the free-disk budget.");
  }
  std::filesystem::create_directory(work);
  const auto capacity = static_cast<std::size_t>(
    std::min(memory_limit_bytes * 0.25, 512.0 * 1024 * 1024) /
    sizeof(CoarseRowRecord));
  GraphRunSorter<CoarseRowRecord, CoarseRowLess> sorter(
    work, "rows", capacity, CoarseRowLess{});
  RunReader<CoarseRecord> reader(input);
  CoarseRecord edge{};
  std::uint32_t previous_low = 0;
  std::uint32_t previous_high = 0;
  double total = 0.0;
  for (std::uint64_t index = 0; index < pairs; ++index) {
    if (!reader.next(edge) || edge.low < 1 || edge.high < edge.low ||
        edge.high > vertices || !std::isfinite(edge.weight) ||
        edge.weight <= 0.0 ||
        edge.low < previous_low ||
        (edge.low == previous_low && edge.high <= previous_high)) {
      Rcpp::stop("Contracted Louvain edge is invalid or unsorted.");
    }
    previous_low = edge.low;
    previous_high = edge.high;
    total += edge.weight;
    if (!std::isfinite(total)) Rcpp::stop("Coarse weight overflow.");
    const double diagonal = 2.0 * edge.weight;
    if (edge.low == edge.high) {
      if (!std::isfinite(diagonal)) Rcpp::stop("Coarse loop overflow.");
      sorter.push({edge.low, edge.high, diagonal});
    } else {
      sorter.push({edge.low, edge.high, edge.weight});
      sorter.push({edge.high, edge.low, edge.weight});
    }
    if (index % 65536 == 0) Rcpp::checkUserInterrupt();
  }
  auto runs = reduce_graph_runs<CoarseRowRecord>(
    sorter.finish(), work, "rows", CoarseRowLess{});
  RunSink<std::uint64_t> offset_sink(offsets + ".part");
  RunSink<std::uint32_t> index_sink(indices + ".part");
  RunSink<double> weight_sink(weights + ".part");
  offset_sink.push(0);
  std::uint64_t row = 1;
  std::uint64_t count = 0;
  std::uint64_t degree = 0;
  std::uint64_t max_degree = 0;
  std::uint32_t last_target = 0;
  merge_graph_runs<CoarseRowRecord>(runs, CoarseRowLess{},
    [&](const CoarseRowRecord& item) {
      while (row < item.source) {
        offset_sink.push(count);
        max_degree = std::max(max_degree, degree);
        ++row;
        degree = 0;
        last_target = 0;
      }
      if (row != item.source || item.target <= last_target) {
        Rcpp::stop("Contracted CSR row ordering is invalid.");
      }
      index_sink.push(item.target);
      weight_sink.push(item.weight);
      last_target = item.target;
      ++degree;
      ++count;
    });
  while (row <= vertices) {
    offset_sink.push(count);
    max_degree = std::max(max_degree, degree);
    ++row;
    degree = 0;
  }
  offset_sink.finish();
  index_sink.finish();
  weight_sink.finish();
  if (std::filesystem::file_size(input, error) !=
      checked_product(pairs, sizeof(CoarseRecord)) || error ||
      file_stamp(input) != input_stamp) {
    Rcpp::stop("Contracted Louvain input changed during CSR build.");
  }
  std::filesystem::rename(offsets + ".part", offsets);
  std::filesystem::rename(indices + ".part", indices);
  std::filesystem::rename(weights + ".part", weights);
  for (const auto& path : runs) std::filesystem::remove(path);
  std::filesystem::remove(work);
  return List::create(
    Rcpp::Named("offsets_path") = offsets,
    Rcpp::Named("indices_path") = indices,
    Rcpp::Named("weights_path") = weights,
    Rcpp::Named("n_vertices") = n_vertices,
    Rcpp::Named("n_edges") = static_cast<double>(count),
    Rcpp::Named("max_degree") = static_cast<double>(max_degree),
    Rcpp::Named("total_edge_weight") = total);
}

// [[Rcpp::export]]
void massive_remap_louvain_cpp(std::string labels_path,
                               std::string mapping_path,
                               double n_vertices,
                               double n_mapping_vertices) {
  require_little_endian();
  const auto vertices = whole_number(n_vertices, "n_vertices");
  const auto mapped = whole_number(n_mapping_vertices,
                                   "n_mapping_vertices");
  if (vertices < 1 || mapped < 1 || mapped > INT32_MAX) {
    Rcpp::stop("Invalid Louvain label remapping dimensions.");
  }
  std::error_code error;
  if (std::filesystem::file_size(labels_path, error) !=
      checked_product(vertices, 4) || error) {
    Rcpp::stop("Louvain label file size is invalid.");
  }
  GraphFile mapping(mapping_path, checked_product(mapped, 4), "mmap");
  std::fstream labels(labels_path,
                      std::ios::in | std::ios::out | std::ios::binary);
  if (!labels) Rcpp::stop("Cannot open Louvain labels for remapping.");
  std::vector<std::uint32_t> buffer(65536);
  for (std::uint64_t first = 0; first < vertices;) {
    const auto count = static_cast<std::size_t>(
      std::min<std::uint64_t>(buffer.size(), vertices - first));
    const auto offset = checked_product(first, 4);
    labels.seekg(static_cast<std::streamoff>(offset));
    labels.read(reinterpret_cast<char*>(buffer.data()), count * 4);
    if (!labels) Rcpp::stop("Louvain labels could not be read.");
    for (std::size_t index = 0; index < count; ++index) {
      const auto old = buffer[index];
      if (old < 1 || old > mapped) {
        Rcpp::stop("Louvain label exceeds the contracted graph.");
      }
      buffer[index] = read_u32(mapping.read((old - 1) * 4, 4));
      if (buffer[index] < 1 || buffer[index] > mapped) {
        Rcpp::stop("Contracted Louvain mapping is invalid.");
      }
    }
    labels.seekp(static_cast<std::streamoff>(offset));
    labels.write(reinterpret_cast<const char*>(buffer.data()), count * 4);
    if (!labels) Rcpp::stop("Louvain labels could not be remapped.");
    first += count;
    Rcpp::checkUserInterrupt();
  }
  labels.flush();
  if (!labels) Rcpp::stop("Louvain label remapping flush failed.");
}

// [[Rcpp::export]]
List massive_read_contracted_cpp(std::string path,
                                 int n_vertices,
                                 double n_edges) {
  const auto count = whole_number(n_edges, "n_edges");
  if (n_vertices < 1 ||
      count > static_cast<std::uint64_t>(R_XLEN_T_MAX)) {
    Rcpp::stop("Contracted graph exceeds resident R limits.");
  }
  std::error_code error;
  if (std::filesystem::file_size(path, error) !=
        checked_product(count, sizeof(CoarseRecord)) || error) {
    Rcpp::stop("Contracted graph file size is invalid.");
  }
  RunReader<CoarseRecord> reader(path);
  Rcpp::IntegerVector from(count), to(count);
  Rcpp::NumericVector weight(count);
  CoarseRecord edge{};
  for (std::uint64_t index = 0; index < count; ++index) {
    if (!reader.next(edge) || edge.low < 1 || edge.high < edge.low ||
        edge.high > static_cast<std::uint32_t>(n_vertices) ||
        !std::isfinite(edge.weight) || edge.weight <= 0.0) {
      Rcpp::stop("Contracted Louvain edge is invalid.");
    }
    from[index] = edge.low;
    to[index] = edge.high;
    weight[index] = edge.weight;
  }
  return List::create(Rcpp::Named("from") = from,
                      Rcpp::Named("to") = to,
                      Rcpp::Named("weight") = weight,
                      Rcpp::Named("n_vertices") = n_vertices);
}

namespace {

double massive_umap_clip(double value) {
  return std::max(-4.0, std::min(4.0, value));
}

bool massive_umap_due(int epoch, double period) {
  if (epoch < 1) return false;
  return std::floor(epoch / period) >
    std::floor((epoch - 1) / period);
}

int massive_umap_negative_count(int epoch, double period, int rate) {
  if (rate == 0) return 0;
  const double sample_index = std::floor(epoch / period);
  const int previous_epoch = sample_index <= 1.0 ? 0 :
    static_cast<int>(std::ceil((sample_index - 1.0) * period));
  const double negative_period = period / rate;
  const double due = std::max(0.0,
    std::floor(epoch / negative_period - 1.0));
  const double previous = previous_epoch == 0 ? 0.0 :
    std::max(0.0,
      std::floor(previous_epoch / negative_period - 1.0));
  return static_cast<int>(std::max(0.0, due - previous));
}

void massive_umap_attract(float* layout,
                          std::uint64_t head, std::uint64_t tail,
                          int dimensions, double a, double b,
                          double alpha) {
  double difference[3]{};
  double distance = 0.0;
  for (int axis = 0; axis < dimensions; ++axis) {
    difference[axis] = layout[head * dimensions + axis] -
      layout[tail * dimensions + axis];
    distance += difference[axis] * difference[axis];
  }
  if (distance <= 0.0) return;
  const double power = fastembedr_umap_pow(distance, b);
  const double coefficient = -2.0 * a * b * power /
    (distance * (a * power + 1.0));
  for (int axis = 0; axis < dimensions; ++axis) {
    const auto gradient = alpha * massive_umap_clip(
      coefficient * difference[axis]);
    layout[head * dimensions + axis] += gradient;
    layout[tail * dimensions + axis] -= gradient;
  }
}

void massive_umap_repulse(float* layout,
                          std::uint64_t head, std::uint64_t negative,
                          int dimensions, double a, double b,
                          double gamma, double alpha) {
  double difference[3]{};
  double distance = 0.0;
  for (int axis = 0; axis < dimensions; ++axis) {
    difference[axis] = layout[head * dimensions + axis] -
      layout[negative * dimensions + axis];
    distance += difference[axis] * difference[axis];
  }
  if (distance <= 0.0) return;
  const double coefficient = 2.0 * gamma * b /
    ((0.001 + distance) *
      (a * fastembedr_umap_pow(distance, b) + 1.0));
  for (int axis = 0; axis < dimensions; ++axis) {
    layout[head * dimensions + axis] += alpha * massive_umap_clip(
      coefficient * difference[axis]);
  }
}

template <typename Function>
void massive_umap_scan(CsrGraphSource& graph, int chunk_rows,
                       std::uint64_t max_edges, Function consume);

void massive_umap_delta_add(std::atomic<std::int64_t>* delta,
                            std::uint64_t index, double value,
                            double scale) {
  if (!std::isfinite(value) ||
      std::abs(value) * scale >= std::ldexp(1.0, 62)) {
    throw std::runtime_error(
      "Synchronized UMAP gradient is non-finite or exceeds its budget.");
  }
  const auto fixed = static_cast<std::int64_t>(std::llround(value * scale));
  delta[index].fetch_add(fixed, std::memory_order_relaxed);
}

int massive_umap_synchronized_edge(const float* layout,
    std::atomic<std::int64_t>* delta, std::uint64_t head,
    std::uint64_t tail, int dimensions, double a, double b,
    double gamma, double alpha, double scale,
    int negatives, std::uint64_t vertices, int seed, int epoch,
    std::uint64_t sample_edge) {
  double difference[3]{};
  double distance = 0.0;
  for (int axis = 0; axis < dimensions; ++axis) {
    difference[axis] = layout[head * dimensions + axis] -
      layout[tail * dimensions + axis];
    distance += difference[axis] * difference[axis];
  }
  if (distance > 0.0) {
    const double power = fastembedr_umap_pow(distance, b);
    const double coefficient = -2.0 * a * b * power /
      (distance * (a * power + 1.0));
    for (int axis = 0; axis < dimensions; ++axis) {
      const auto gradient = alpha * massive_umap_clip(
        coefficient * difference[axis]);
      massive_umap_delta_add(delta, head * dimensions + axis,
                             gradient, scale);
      massive_umap_delta_add(delta, tail * dimensions + axis,
                             -gradient, scale);
    }
  }
  int sampled = 0;
  for (int sample = 0; sample < negatives; ++sample) {
    const auto negative = fastembedr_umap_negative_vertex(
      static_cast<int>(vertices), seed, epoch,
      static_cast<std::size_t>(sample_edge), sample);
    if (negative == static_cast<int>(head)) continue;
    ++sampled;
    double negative_distance = 0.0;
    for (int axis = 0; axis < dimensions; ++axis) {
      difference[axis] = layout[head * dimensions + axis] -
        layout[negative * dimensions + axis];
      negative_distance += difference[axis] * difference[axis];
    }
    if (negative_distance <= 0.0) continue;
    const double coefficient = 2.0 * gamma * b /
      ((0.001 + negative_distance) *
       (a * fastembedr_umap_pow(negative_distance, b) + 1.0));
    for (int axis = 0; axis < dimensions; ++axis) {
      massive_umap_delta_add(delta, head * dimensions + axis,
        alpha * massive_umap_clip(coefficient * difference[axis]),
        scale);
    }
  }
  return sampled;
}

std::vector<std::uint64_t> massive_umap_worker_offsets(
    CsrGraphSource& graph, int workers, int chunk_rows,
    std::uint64_t max_edges, double min_weight) {
  std::vector<std::uint64_t> offsets(workers + 1, 0);
  int worker = 0;
  massive_umap_scan(graph, chunk_rows, max_edges,
    [&](std::uint64_t head, std::uint64_t, float weight,
        std::uint64_t) {
      while (worker + 1 < workers &&
             head >= graph.vertices() * (worker + 1) / workers) {
        ++worker;
      }
      if (weight >= min_weight) ++offsets[worker + 1];
    });
  for (int i = 1; i <= workers; ++i) offsets[i] += offsets[i - 1];
  return offsets;
}

std::pair<std::uint64_t, std::uint64_t>
massive_umap_synchronized_epoch(CsrGraphSource& graph,
    const std::string& offsets_path, const std::string& indices_path,
    const std::string& weights_path, float* layout,
    std::atomic<std::int64_t>* delta, std::uint64_t elements,
    int dimensions, int chunk_rows, std::uint64_t max_edges,
    int workers, int epoch, int n_epochs, int negative_rate,
    double learning_rate, double a, double b, double gamma,
    int seed, float max_weight, double scale,
    const std::vector<std::uint64_t>& worker_offsets) {
  parallel_rows(elements, workers,
    [&](std::uint64_t begin, std::uint64_t end, int) {
      for (auto i = begin; i < end; ++i) {
        delta[i].store(0, std::memory_order_relaxed);
      }
    });
  const double alpha = learning_rate * (1.0 -
    static_cast<double>(epoch) / n_epochs);
  const double threshold = static_cast<double>(max_weight) / n_epochs;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> counts(workers);
  std::vector<std::exception_ptr> errors(workers);
  parallel_rows(graph.vertices(), workers,
    [&](std::uint64_t begin, std::uint64_t end, int worker) {
      try {
        CsrGraphSource source(offsets_path, indices_path,
                              weights_path, graph.vertices(), "stream");
        auto active_edge = worker_offsets[worker];
        for (auto first = begin + 1; first <= end;) {
          const auto rows = std::min(source.bounded_rows(first,
            chunk_rows, max_edges), end - first + 1);
          source.visit(first, rows,
            [&](std::uint64_t head, std::uint32_t tail,
                float weight, std::uint64_t) {
              if (weight < threshold) return;
              const auto sample_edge = active_edge++;
              const double period = max_weight /
                std::max(static_cast<double>(weight), 1e-6);
              if (!massive_umap_due(epoch, period)) return;
              const int negatives = massive_umap_negative_count(
                epoch, period, negative_rate);
              const int sampled = massive_umap_synchronized_edge(
                layout, delta,
                head - 1, tail - 1, dimensions, a, b, gamma,
                alpha, scale, negatives, graph.vertices(), seed,
                epoch, sample_edge);
              ++counts[worker].first;
              counts[worker].second += sampled;
            }, false);
          first += rows;
        }
      } catch (...) {
        errors[worker] = std::current_exception();
      }
    });
  for (const auto& error : errors) {
    if (error) std::rethrow_exception(error);
  }
  parallel_rows(elements, workers,
    [&](std::uint64_t begin, std::uint64_t end, int) {
      for (auto i = begin; i < end; ++i) {
        const double update = static_cast<double>(
          delta[i].load(std::memory_order_relaxed)) / scale;
        layout[i] += static_cast<float>(
          std::max(-16.0, std::min(16.0, update)));
      }
    });
  std::pair<std::uint64_t, std::uint64_t> total{};
  for (const auto& count : counts) {
    total.first += count.first;
    total.second += count.second;
  }
  return total;
}

template <typename Function>
void massive_umap_scan(CsrGraphSource& graph, int chunk_rows,
                       std::uint64_t max_edges, Function consume) {
  for (std::uint64_t first = 1; first <= graph.vertices();) {
    const auto count = graph.bounded_rows(first, chunk_rows, max_edges);
    const auto edge_start = graph.offset(first - 1);
    graph.visit(first, count,
      [&](std::uint64_t source, std::uint32_t target,
          float weight, std::uint64_t position) {
      consume(source - 1, target - 1, weight,
              edge_start + position);
    });
    first += count;
    Rcpp::checkUserInterrupt();
  }
}

void massive_umap_write_layout(const float* layout,
                               const std::string& path,
                               std::uint64_t vertices, int dimensions,
                               int chunk_rows, std::uint64_t bytes) {
  Float32Sink sink(path);
  for (std::uint64_t row = 0; row < vertices;) {
    const auto count = std::min<std::uint64_t>(chunk_rows,
      vertices - row);
    sink.append(layout + row * dimensions, count * dimensions);
    row += count;
  }
  sink.finish(bytes);
}

class MappedUmapLayout {
 public:
  MappedUmapLayout(const std::string& output, MatrixSource& init,
                   std::uint64_t vertices, int dimensions,
                   int chunk_rows, std::uint64_t bytes, bool resume)
      : partial_(output + ".part"), mapped_(nullptr), bytes_(bytes) {
#if defined(__unix__) || defined(__APPLE__)
    if (bytes > std::numeric_limits<std::size_t>::max()) {
      Rcpp::stop("UMAP layout exceeds the mmap address range.");
    }
    const int flags = resume ? O_WRONLY | O_NOFOLLOW :
      O_WRONLY | O_CREAT | O_EXCL;
    const int output_fd = open(partial_.c_str(), flags, 0666);
    if (output_fd < 0) {
      Rcpp::stop("Cannot open UMAP mapped output .part file.");
    }
    if (resume) {
      struct stat info{};
      if (fstat(output_fd, &info) != 0 || !S_ISREG(info.st_mode) ||
          info.st_size < 0 ||
          static_cast<std::uint64_t>(info.st_size) != bytes ||
          ftruncate(output_fd, 0) != 0) {
        close(output_fd);
        Rcpp::stop("UMAP mapped .part file does not match checkpoint.");
      }
    }
    std::FILE* raw_stream = fdopen(output_fd, "wb");
    if (raw_stream == nullptr) {
      close(output_fd);
      Rcpp::stop("Cannot open UMAP mapped output stream.");
    }
    const auto close_stream = [](std::FILE* handle) {
      std::fclose(handle);
    };
    std::unique_ptr<std::FILE, decltype(close_stream)> stream(
      raw_stream, close_stream);
    const auto block = std::min(chunk_rows, 8192);
    std::vector<float> buffer(static_cast<std::size_t>(block) *
                              dimensions);
    for (std::uint64_t row = 0; row < vertices;) {
      const auto count = std::min<std::uint64_t>(block,
                                                vertices - row);
      init.read_rows(row, count, buffer.data());
      const auto elements = static_cast<std::size_t>(count) * dimensions;
      if (std::fwrite(buffer.data(), sizeof(float), elements,
                      stream.get()) != elements) {
        Rcpp::stop("UMAP mapped output write failed.");
      }
      row += count;
      Rcpp::checkUserInterrupt();
    }
    if (std::fclose(stream.release()) != 0) {
      Rcpp::stop("UMAP mapped output close failed.");
    }
    const int fd = open(partial_.c_str(), O_RDWR);
    if (fd < 0) Rcpp::stop("Cannot open UMAP mapped output.");
    void* address = mmap(nullptr, static_cast<std::size_t>(bytes),
                         PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (address == MAP_FAILED) Rcpp::stop("Cannot mmap UMAP output.");
    mapped_ = static_cast<float*>(address);
#else
    Rcpp::stop("Writable UMAP mmap is unavailable on this system.");
#endif
  }

  ~MappedUmapLayout() {
#if defined(__unix__) || defined(__APPLE__)
    if (mapped_ != nullptr) munmap(mapped_,
                                  static_cast<std::size_t>(bytes_));
#endif
  }

  float* data() { return mapped_; }

  void finish(const std::string& output) {
#if defined(__unix__) || defined(__APPLE__)
    if (msync(mapped_, static_cast<std::size_t>(bytes_), MS_SYNC) != 0) {
      Rcpp::stop("UMAP mapped output sync failed; .part retained.");
    }
    if (munmap(mapped_, static_cast<std::size_t>(bytes_)) != 0) {
      Rcpp::stop("UMAP mapped output unmap failed; .part retained.");
    }
    mapped_ = nullptr;
    if (std::filesystem::exists(output)) {
      Rcpp::stop("UMAP output appeared during fitting; .part retained.");
    }
    std::error_code error;
    std::filesystem::rename(partial_, output, error);
    if (error) Rcpp::stop("UMAP output rename failed: " +
                          error.message());
#endif
  }

 private:
  std::string partial_;
  float* mapped_;
  std::uint64_t bytes_;
};

}  // namespace

// [[Rcpp::export]]
List massive_umap_optimize_cpp(std::string offsets_path,
                               std::string indices_path,
                               std::string weights_path,
                               double n_vertices,
                               std::string init_path,
                               std::string init_format,
                               int dimensions, std::string output,
                               int n_epochs, int negative_sample_rate,
                               double learning_rate, double min_dist,
                               double repulsion_strength, int seed,
                               int chunk_rows, double memory_limit_bytes,
                               int start_epoch = 0,
                               double positive_done = 0,
                               double negative_done = 0,
                               int checkpoint_every = 0,
                               SEXP checkpoint_callback = R_NilValue,
                               std::string layout_storage = "memory",
                               int n_threads = 1,
                               std::string cpu_update = "sgd") {
  require_little_endian();
  if (dimensions < 2 || dimensions > 3 || n_epochs < 2 ||
      negative_sample_rate < 0 || negative_sample_rate > 1000 ||
      !std::isfinite(learning_rate) || learning_rate <= 0.0 ||
      !std::isfinite(min_dist) || min_dist < 0.0 || min_dist > 1.0 ||
      !std::isfinite(repulsion_strength) || repulsion_strength <= 0.0 ||
      chunk_rows < 1 || start_epoch < 0 || start_epoch > n_epochs ||
      checkpoint_every < 0 ||
      (start_epoch > 0 && checkpoint_every == 0) ||
      (checkpoint_every > 0 && checkpoint_callback == R_NilValue) ||
      (checkpoint_every == 0 && checkpoint_callback != R_NilValue) ||
      (layout_storage != "memory" && layout_storage != "mmap") ||
      n_threads < 1 || n_threads > 128 ||
      (cpu_update != "sgd" && cpu_update != "synchronized") ||
      (cpu_update == "sgd" && n_threads != 1)) {
    Rcpp::stop("Invalid experimental UMAP optimizer settings.");
  }
  const auto initial_positive = whole_number(positive_done,
                                             "positive updates");
  const auto initial_negative = whole_number(negative_done,
                                             "negative updates");
  if (layout_storage == "mmap" &&
      init_path == output + ".part") {
    Rcpp::stop("UMAP mapped input cannot be its output .part file.");
  }
  CsrGraphSource graph(offsets_path, indices_path, weights_path,
                       n_vertices, "stream");
  FileMatrixSource init(init_path, init_format, "stream", n_vertices,
                        dimensions);
  const auto elements = checked_product(graph.vertices(), dimensions);
  const auto bytes = checked_product(elements, 4);
  if (static_cast<std::uint64_t>(n_threads) > graph.vertices()) {
    Rcpp::stop("UMAP CPU workers exceed graph vertices.");
  }
  const auto resident_bytes = layout_storage == "memory" ? bytes :
    static_cast<std::uint64_t>(std::min(chunk_rows, 8192)) *
      dimensions * 4;
  const auto delta_bytes = cpu_update == "synchronized" ?
    checked_product(elements, 8) : 0;
  if (!std::isfinite(memory_limit_bytes) ||
      memory_limit_bytes < 128.0 * 1024 * 1024 +
        resident_bytes * 1.25 + delta_bytes ||
      elements > std::numeric_limits<std::size_t>::max()) {
    Rcpp::stop("UMAP layout exceeds the experimental RAM budget.");
  }
  const auto edge_capacity = std::min(4000000.0,
    std::max(65536.0, (memory_limit_bytes - resident_bytes -
      delta_bytes) / 64.0));
  const auto max_edges = static_cast<std::uint64_t>(edge_capacity);
  std::unique_ptr<MappedUmapLayout> mapped;
  std::vector<float> resident;
  if (layout_storage == "mmap") {
    mapped.reset(new MappedUmapLayout(output, init, graph.vertices(),
                                     dimensions, chunk_rows, bytes,
                                     start_epoch > 0));
  } else {
    resident.resize(static_cast<std::size_t>(elements));
    for (std::uint64_t row = 0; row < graph.vertices();) {
      const auto count = std::min<std::uint64_t>(chunk_rows,
        graph.vertices() - row);
      init.read_rows(row, count,
                     resident.data() + row * dimensions);
      row += count;
    }
  }
  float* layout = mapped ? mapped->data() : resident.data();
  float max_weight = 0.0f;
  massive_umap_scan(graph, chunk_rows, max_edges,
    [&](std::uint64_t, std::uint64_t, float weight, std::uint64_t) {
    max_weight = std::max(max_weight, weight);
  });
  if (!(max_weight > 0.0f)) {
    Rcpp::stop("The experimental UMAP graph has no positive edges.");
  }
  const double min_sample_weight =
    static_cast<double>(max_weight) / n_epochs;
  const auto curve = fastembedr_umap_curve(min_dist);
  std::unique_ptr<std::atomic<std::int64_t>[]> delta;
  std::vector<std::uint64_t> worker_offsets;
  double delta_scale = 0.0;
  if (cpu_update == "synchronized") {
    for (std::uint64_t element = 0; element < elements; ++element) {
      if (!std::isfinite(layout[element])) {
        Rcpp::stop("Synchronized UMAP initialization is non-finite.");
      }
    }
    const double contributions = static_cast<double>(graph.edges()) *
      (1.0 + negative_sample_rate);
    const double limit = std::ldexp(1.0, 61) /
      (4.0 * learning_rate * contributions);
    if (!std::isfinite(limit) || limit < 256.0) {
      Rcpp::stop("Synchronized UMAP delta exceeds the fixed-point "
                 "overflow budget.");
    }
    delta_scale = std::exp2(std::floor(std::log2(
      std::min(limit, std::ldexp(1.0, 20)))));
    delta.reset(new std::atomic<std::int64_t>[
      static_cast<std::size_t>(elements)]);
    worker_offsets = massive_umap_worker_offsets(graph, n_threads,
      chunk_rows, max_edges, min_sample_weight);
  }
  std::uint64_t positive_updates = initial_positive;
  std::uint64_t negative_updates = initial_negative;
  int next_percent = (start_epoch * 100 / n_epochs / 10 + 1) * 10;
  for (int epoch = start_epoch; epoch < n_epochs; ++epoch) {
    const double alpha = learning_rate *
      (1.0 - static_cast<double>(epoch) / n_epochs);
    if (cpu_update == "synchronized") {
      const auto counts = massive_umap_synchronized_epoch(graph,
        offsets_path, indices_path, weights_path, layout,
        delta.get(), elements, dimensions, chunk_rows,
        max_edges, n_threads, epoch, n_epochs,
        negative_sample_rate, learning_rate, curve.first,
        curve.second, repulsion_strength, seed, max_weight,
        delta_scale, worker_offsets);
      positive_updates += counts.first;
      negative_updates += counts.second;
    } else {
      std::uint64_t active_edge = 0;
      massive_umap_scan(graph, chunk_rows, max_edges,
      [&](std::uint64_t head, std::uint64_t tail,
          float weight, std::uint64_t) {
      if (weight < min_sample_weight) return;
      const auto sample_edge = active_edge++;
      const double period = max_weight /
        std::max(static_cast<double>(weight), 1e-6);
      if (!massive_umap_due(epoch, period)) return;
      massive_umap_attract(layout, head, tail, dimensions,
                            curve.first, curve.second, alpha);
      ++positive_updates;
      const int count = massive_umap_negative_count(epoch, period,
                                                    negative_sample_rate);
      for (int sample = 0; sample < count; ++sample) {
        const auto negative = fastembedr_umap_negative_vertex(
          static_cast<int>(graph.vertices()), seed, epoch,
          static_cast<std::size_t>(sample_edge), sample);
        if (negative == static_cast<int>(head)) continue;
        massive_umap_repulse(layout, head, negative, dimensions,
          curve.first, curve.second, repulsion_strength, alpha);
        ++negative_updates;
      }
      });
    }
    for (std::uint64_t element = 0; element < elements; ++element) {
      if (!std::isfinite(layout[element])) {
        Rcpp::stop("Experimental UMAP layout became non-finite.");
      }
    }
    report_graph_progress("UMAP epochs", epoch + 1, n_epochs,
                          next_percent);
    if (checkpoint_every > 0 &&
        ((epoch + 1) % checkpoint_every == 0 || epoch + 1 == n_epochs)) {
      const auto snapshot = output + ".epoch_" +
        std::to_string(epoch + 1) + ".f32";
      massive_umap_write_layout(layout, snapshot, graph.vertices(),
                                dimensions, chunk_rows, bytes);
      Rcpp::Function callback(checkpoint_callback);
      callback(epoch + 1, snapshot,
        static_cast<double>(positive_updates),
        static_cast<double>(negative_updates));
    }
  }
  if (mapped) mapped->finish(output);
  else massive_umap_write_layout(layout, output, graph.vertices(),
                                 dimensions, chunk_rows, bytes);
  return List::create(
    Rcpp::Named("n_vertices") = static_cast<double>(graph.vertices()),
    Rcpp::Named("n_edges") = static_cast<double>(graph.edges()),
    Rcpp::Named("positive_updates") =
      static_cast<double>(positive_updates),
    Rcpp::Named("negative_updates") =
      static_cast<double>(negative_updates));
}

// [[Rcpp::export]]
List massive_umap_optimize_metal_cpp(std::string offsets_path,
    std::string indices_path, std::string weights_path,
    double n_vertices, std::string init_path,
    std::string init_format, std::string output,
    int n_epochs, int negative_sample_rate,
    double learning_rate, double min_dist,
    double repulsion_strength, int seed, int chunk_rows,
    int edge_capacity, double memory_limit_bytes,
    int start_epoch = 0, double positive_done = 0,
    double negative_done = 0, int checkpoint_every = 0,
    SEXP checkpoint_callback = R_NilValue,
    std::string layout_storage = "memory") {
  require_little_endian();
  if (n_epochs < 2 || negative_sample_rate < 0 ||
      negative_sample_rate > 1000 || chunk_rows < 1 ||
      edge_capacity < 1 || start_epoch < 0 ||
      start_epoch > n_epochs || checkpoint_every < 0 ||
      (start_epoch > 0 && checkpoint_every == 0) ||
      (checkpoint_every > 0 && checkpoint_callback == R_NilValue) ||
      (layout_storage != "memory" && layout_storage != "mmap")) {
    Rcpp::stop("Invalid full-graph Metal UMAP controls.");
  }
  const auto initial_positive = whole_number(positive_done,
                                              "positive updates");
  const auto initial_negative = whole_number(negative_done,
                                              "negative updates");
  CsrGraphSource graph(offsets_path, indices_path, weights_path,
                       n_vertices, "stream");
  FileMatrixSource init(init_path, init_format, "stream",
                        n_vertices, 2);
  const auto elements = checked_product(graph.vertices(), 2);
  const auto bytes = checked_product(elements, 4);
  const auto staging_bytes = layout_storage == "memory" ? bytes :
    static_cast<std::uint64_t>(std::min(chunk_rows, 8192)) * 8;
  const double required = 128.0 * 1024 * 1024 +
    bytes * 1.5 + staging_bytes + edge_capacity * 32.0;
  if (graph.vertices() > INT_MAX ||
      elements > std::numeric_limits<std::size_t>::max() ||
      !std::isfinite(memory_limit_bytes) ||
      required > memory_limit_bytes) {
    Rcpp::stop("Metal UMAP layout exceeds the shared-memory budget.");
  }
  std::unique_ptr<MappedUmapLayout> mapped;
  std::vector<float> resident;
  if (layout_storage == "mmap") {
    if (init_path == output + ".part") {
      Rcpp::stop("Metal UMAP mapped input cannot be its output .part file.");
    }
    mapped.reset(new MappedUmapLayout(output, init, graph.vertices(),
                                     2, chunk_rows, bytes,
                                     start_epoch > 0));
  } else {
    resident.resize(static_cast<std::size_t>(elements));
    for (std::uint64_t row = 0; row < graph.vertices();) {
      const auto count = std::min<std::uint64_t>(chunk_rows,
        graph.vertices() - row);
      init.read_rows(row, count, resident.data() + row * 2);
      row += count;
    }
  }
  float* layout = mapped ? mapped->data() : resident.data();
  float max_weight = 0.0f;
  massive_umap_scan(graph, chunk_rows, edge_capacity,
    [&](std::uint64_t, std::uint64_t, float weight,
        std::uint64_t) {
      max_weight = std::max(max_weight, weight);
    });
  if (!(max_weight > 0.0f)) {
    Rcpp::stop("The full-graph Metal UMAP graph has no positive edges.");
  }
  const auto curve = fastembedr_umap_curve(min_dist);
  using MetalContext = std::unique_ptr<void, decltype(
    &fastembedr_massive_umap_metal_destroy)>;
  MetalContext context(fastembedr_massive_umap_metal_create(
    layout, static_cast<int>(graph.vertices()), 2,
    n_epochs, negative_sample_rate, learning_rate,
    curve.first, curve.second, repulsion_strength, seed),
    fastembedr_massive_umap_metal_destroy);
  std::vector<int> heads(edge_capacity), tails(edge_capacity);
  std::vector<int> negatives(edge_capacity);
  auto positive = initial_positive;
  auto negative = initial_negative;
  const double min_weight = max_weight / n_epochs;
  int next_percent = (start_epoch * 100 / n_epochs / 10 + 1) * 10;
  for (int epoch = start_epoch; epoch < n_epochs; ++epoch) {
    int active = 0;
    auto flush = [&]() {
      negative += fastembedr_massive_umap_metal_step(
        context.get(), heads.data(), tails.data(),
        negatives.data(), active, epoch);
      active = 0;
    };
    massive_umap_scan(graph, chunk_rows, edge_capacity,
      [&](std::uint64_t head, std::uint64_t tail,
          float weight, std::uint64_t) {
      if (weight < min_weight) return;
      const double period = max_weight /
        std::max(static_cast<double>(weight), 1e-6);
      if (!massive_umap_due(epoch, period)) return;
      heads[active] = static_cast<int>(head);
      tails[active] = static_cast<int>(tail);
      negatives[active] = massive_umap_negative_count(
        epoch, period, negative_sample_rate);
      ++positive;
      if (++active == edge_capacity) flush();
    });
    flush();
    report_graph_progress("Metal UMAP epochs", epoch + 1,
                          n_epochs, next_percent);
    if (checkpoint_every > 0 &&
        ((epoch + 1) % checkpoint_every == 0 ||
          epoch + 1 == n_epochs)) {
      fastembedr_massive_umap_metal_download(
        context.get(), layout);
      const auto snapshot = output + ".epoch_" +
        std::to_string(epoch + 1) + ".f32";
      massive_umap_write_layout(layout, snapshot,
        graph.vertices(), 2, chunk_rows, bytes);
      Rcpp::Function callback(checkpoint_callback);
      callback(epoch + 1, snapshot,
        static_cast<double>(positive),
        static_cast<double>(negative));
    }
  }
  fastembedr_massive_umap_metal_download(context.get(), layout);
  for (std::size_t i = 0; i < elements; ++i) {
    if (!std::isfinite(layout[i])) {
      Rcpp::stop("Full-graph Metal UMAP produced nonfinite coordinates.");
    }
  }
  if (mapped) mapped->finish(output);
  else massive_umap_write_layout(layout, output,
                                graph.vertices(), 2, chunk_rows, bytes);
  return List::create(
    Rcpp::Named("n_vertices") =
      static_cast<double>(graph.vertices()),
    Rcpp::Named("n_edges") = static_cast<double>(graph.edges()),
    Rcpp::Named("positive_updates") =
      static_cast<double>(positive),
    Rcpp::Named("negative_updates") =
      static_cast<double>(negative));
}

#ifdef FASTEMBEDR_HAS_CUDA
namespace {

void massive_umap_cuda_check(int status) {
  if (status != 0) {
    Rcpp::stop(std::string("Massive CUDA UMAP failed: ") +
               fastembedr_cuda_embedding_last_error());
  }
}

#ifdef FASTEMBEDR_HAS_NCCL
struct UmapRankProgress {
  double graph_seconds = 0;
  double step_seconds = 0;
  double visits = 0;
  double rows = 0;
};

UmapRankProgress massive_umap_rank_epoch(CsrGraphSource& graph,
    void* context, int rank, std::uint64_t owner_rows,
    int chunk_rows, int edge_capacity, float min_weight,
    float max_weight, int epoch) {
  UmapRankProgress progress;
  std::vector<int> heads(edge_capacity), tails(edge_capacity);
  std::vector<float> weights(edge_capacity), periods(edge_capacity);
  const auto first_row = static_cast<std::uint64_t>(rank) *
    owner_rows + 1;
  const auto last_row = std::min(graph.vertices(),
    first_row + owner_rows - 1);
  for (auto first = first_row; first <= last_row;) {
    const auto rows = std::min(graph.bounded_rows(first,
      chunk_rows, edge_capacity), last_row - first + 1);
    int active = 0;
    const auto visit_started = std::chrono::steady_clock::now();
    graph.visit(first, rows,
      [&](std::uint64_t source, std::uint32_t target,
          float weight, std::uint64_t) {
        if (weight < min_weight) return;
        heads[active] = static_cast<int>(source - 1);
        tails[active] = static_cast<int>(target - 1);
        weights[active] = weight;
        periods[active] = max_weight /
          std::max(weight, 1.0e-6f);
        ++active;
      }, false);
    progress.rows += rows;
    progress.graph_seconds += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - visit_started).count();
    if (active > 0) {
      const auto step_started = std::chrono::steady_clock::now();
      const int status = fastembedr_massive_umap_multigpu_step(
        context, rank, heads.data(), tails.data(), weights.data(),
        periods.data(), active, epoch);
      if (status != 0) throw std::runtime_error(
        fastembedr_cuda_embedding_last_error());
      progress.step_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - step_started).count();
      progress.visits += active;
    }
    first += rows;
  }
  return progress;
}
#endif

void massive_umap_cuda_write(void* context, const std::string& output,
                             std::uint64_t vertices, int chunk_rows,
                             int dimensions, bool multi_device) {
  const auto rows = std::min(chunk_rows, 8192);
  std::vector<float> buffer(
    static_cast<std::size_t>(rows) * dimensions);
  Float32Sink sink(output);
  for (std::uint64_t first = 0; first < vertices;) {
    const auto count = static_cast<int>(std::min<std::uint64_t>(
      rows, vertices - first));
    int status = 1;
#ifdef FASTEMBEDR_HAS_NCCL
    if (multi_device) status =
      fastembedr_massive_umap_multigpu_download(context,
        static_cast<int>(first), count, buffer.data());
    else
#endif
      status = fastembedr_massive_umap_cuda_download(context,
        static_cast<int>(first), count, buffer.data());
    massive_umap_cuda_check(status);
    for (int i = 0; i < count * dimensions; ++i) {
      if (!std::isfinite(buffer[i])) {
        Rcpp::stop("Massive CUDA UMAP produced non-finite coordinates.");
      }
    }
    sink.append(buffer.data(),
      static_cast<std::uint64_t>(count) * dimensions);
    first += count;
    Rcpp::checkUserInterrupt();
  }
  sink.finish(checked_product(vertices, dimensions * 4));
}

}  // namespace
#endif

// [[Rcpp::export]]
bool massive_nccl_available_cpp() {
#ifdef FASTEMBEDR_HAS_NCCL
  return true;
#else
  return false;
#endif
}

// [[Rcpp::export]]
List massive_umap_optimize_cuda_cpp(std::string offsets_path,
    std::string indices_path, std::string weights_path,
    double n_vertices, std::string init_path,
    std::string init_format, std::string output,
    int n_epochs, int negative_sample_rate,
    double learning_rate, double min_dist,
    double repulsion_strength, int seed, int chunk_rows,
    int edge_capacity, double memory_limit_bytes,
    int start_epoch = 0, int checkpoint_every = 0,
    SEXP checkpoint_callback = R_NilValue,
    double visits_done = 0, int dimensions = 2,
    std::string layout_storage = "memory",
    SEXP devices = R_NilValue) {
#ifndef FASTEMBEDR_HAS_CUDA
  Rcpp::stop("Native massive CUDA UMAP is unavailable; no CPU fallback.");
#else
  require_little_endian();
  Rcpp::IntegerVector device_ids =
    Rf_isNull(devices) ? Rcpp::IntegerVector() :
    Rcpp::as<Rcpp::IntegerVector>(devices);
  const bool multi_device = device_ids.size() > 0;
#ifndef FASTEMBEDR_HAS_NCCL
  if (multi_device) Rcpp::stop(
    "Coordinated CUDA UMAP needs a build linked with NCCL.");
#endif
  if (multi_device && layout_storage != "memory") Rcpp::stop(
    "Coordinated CUDA UMAP needs device storage.");
  if (n_epochs < 2 || negative_sample_rate < 0 ||
      negative_sample_rate > 1000 || chunk_rows < 1 ||
      edge_capacity < 1 || start_epoch < 0 ||
      (dimensions != 2 && dimensions != 3) ||
      (layout_storage != "memory" && layout_storage != "managed") ||
      start_epoch > n_epochs || checkpoint_every < 0 ||
      (start_epoch > 0 && checkpoint_every == 0) ||
      (checkpoint_every > 0 && checkpoint_callback == R_NilValue) ||
      (checkpoint_every == 0 && checkpoint_callback != R_NilValue) ||
      !std::isfinite(learning_rate) ||
      learning_rate <= 0 || !std::isfinite(min_dist) ||
      min_dist < 0 || min_dist > 1 ||
      !std::isfinite(repulsion_strength) ||
      repulsion_strength <= 0) {
    Rcpp::stop("Invalid massive CUDA UMAP controls.");
  }
  int previous_device = -1;
  if (device_ids.size() == 1) {
    previous_device = Rcpp::as<int>(
      massive_cuda_memory_cpp()["device"]);
    massive_cuda_select_cpp(device_ids[0]);
  }
  struct DeviceRestore {
    int previous;
    ~DeviceRestore() {
      if (previous >= 0)
        fastembedr_massive_cuda_select_device(previous);
    }
  } restore{previous_device};
  const auto initial_visits = whole_number(visits_done, "edge visits");
  CsrGraphSource graph(offsets_path, indices_path, weights_path,
                       n_vertices, "stream");
  FileMatrixSource init(init_path, init_format, "stream",
                        n_vertices, dimensions);
  const auto host_bytes = 128.0 * 1024 * 1024 +
    static_cast<double>(edge_capacity) * 16 *
      (multi_device ? device_ids.size() : 1) +
    static_cast<double>(std::min(chunk_rows, 8192)) *
      dimensions * 4 +
    (layout_storage == "managed" ?
      static_cast<double>(graph.vertices()) * dimensions * 4 : 0);
  if (!std::isfinite(memory_limit_bytes) ||
      host_bytes > memory_limit_bytes) {
    Rcpp::stop("Massive CUDA UMAP exceeds the RAM budget.");
  }
  float max_weight = 0.0f;
  massive_umap_scan(graph, chunk_rows, edge_capacity,
    [&](std::uint64_t, std::uint64_t, float weight, std::uint64_t) {
      max_weight = std::max(max_weight, weight);
    });
  if (!(max_weight > 0.0f)) {
    Rcpp::stop("Massive CUDA UMAP graph has no positive edges.");
  }
  double delta_scale = NA_REAL;
  if (multi_device) {
    const double contributions = static_cast<double>(graph.edges()) *
      (1.0 + static_cast<double>(n_epochs) * negative_sample_rate);
    const double scale_limit =
      (std::ldexp(1.0, 62) / contributions - 0.5) /
      (4.0 * learning_rate);
    if (!std::isfinite(scale_limit) || scale_limit < 256.0) {
      Rcpp::stop("Coordinated CUDA UMAP fixed-point delta exceeds "
        "the precision or overflow budget.");
    }
    delta_scale = std::exp2(std::floor(std::log2(
      std::min(scale_limit, std::ldexp(1.0, 24)))));
  }
  const auto curve = fastembedr_umap_curve(min_dist);
  void* raw = nullptr;
  auto destroy = &fastembedr_massive_umap_cuda_destroy;
#ifdef FASTEMBEDR_HAS_NCCL
  if (multi_device) {
    raw = fastembedr_massive_umap_multigpu_create(
      static_cast<int>(graph.vertices()), dimensions,
      edge_capacity, n_epochs, negative_sample_rate,
      static_cast<float>(learning_rate),
      static_cast<float>(curve.first), static_cast<float>(curve.second),
      static_cast<float>(repulsion_strength), max_weight,
      static_cast<unsigned int>(seed), INTEGER(device_ids),
      device_ids.size(), delta_scale);
    destroy = &fastembedr_massive_umap_multigpu_destroy;
  } else
#endif
    raw = fastembedr_massive_umap_cuda_create(
      static_cast<int>(graph.vertices()), dimensions, edge_capacity,
      n_epochs, negative_sample_rate,
      static_cast<float>(learning_rate),
      static_cast<float>(curve.first), static_cast<float>(curve.second),
      static_cast<float>(repulsion_strength), max_weight,
      static_cast<unsigned int>(seed),
      layout_storage == "managed" ? 1 : 0);
  if (raw == nullptr) massive_umap_cuda_check(1);
  std::unique_ptr<void, decltype(destroy)> context(raw, destroy);
  const auto init_rows = std::min(chunk_rows, 8192);
  std::vector<float> init_buffer(
    static_cast<std::size_t>(init_rows) * dimensions);
  for (std::uint64_t first = 0; first < graph.vertices();) {
    const auto count = static_cast<int>(std::min<std::uint64_t>(
      init_rows, graph.vertices() - first));
    init.read_rows(first, count, init_buffer.data());
    for (int i = 0; i < count * dimensions; ++i) {
      if (!std::isfinite(init_buffer[i])) {
        Rcpp::stop("Massive CUDA UMAP initialization is non-finite.");
      }
    }
    int upload_status = 1;
#ifdef FASTEMBEDR_HAS_NCCL
    if (multi_device) upload_status =
      fastembedr_massive_umap_multigpu_upload(context.get(),
        static_cast<int>(first), count, init_buffer.data());
    else
#endif
      upload_status = fastembedr_massive_umap_cuda_upload(
        context.get(), static_cast<int>(first), count,
        init_buffer.data());
    massive_umap_cuda_check(upload_status);
    first += count;
  }
  std::vector<int> heads(multi_device ? 0 : edge_capacity);
  std::vector<int> tails(multi_device ? 0 : edge_capacity);
  std::vector<float> weights(multi_device ? 0 : edge_capacity);
  std::vector<float> periods(multi_device ? 0 : edge_capacity);
  const float min_weight = max_weight / n_epochs;
  double edge_visits = static_cast<double>(initial_visits);
  Rcpp::NumericVector per_device_visits(device_ids.size());
  Rcpp::NumericVector per_device_rows(device_ids.size());
  double finalization_block_seconds = 0;
  double graph_visit_seconds = 0;
  double device_step_seconds = 0;
  double owner_wall_seconds = 0;
  const auto owner_rows = multi_device ?
    (graph.vertices() + device_ids.size() - 1) /
      device_ids.size() :
    graph.vertices();
#ifdef FASTEMBEDR_HAS_NCCL
  std::vector<std::unique_ptr<CsrGraphSource>> rank_graphs;
  if (multi_device) {
    for (int rank = 0; rank < device_ids.size(); ++rank) {
      rank_graphs.emplace_back(std::make_unique<CsrGraphSource>(
        offsets_path, indices_path, weights_path, n_vertices,
        "stream"));
    }
  }
#endif
  int next_percent = (start_epoch * 100 / n_epochs / 10 + 1) * 10;
  for (int epoch = start_epoch; epoch < n_epochs; ++epoch) {
#ifdef FASTEMBEDR_HAS_NCCL
    if (multi_device) massive_umap_cuda_check(
      fastembedr_massive_umap_multigpu_start_epoch(context.get()));
    if (multi_device) {
      const auto owner_started = std::chrono::steady_clock::now();
      std::vector<std::future<UmapRankProgress>> pending;
      for (int rank = 0; rank < device_ids.size(); ++rank) {
        pending.emplace_back(std::async(std::launch::async,
          [&, rank]() {
            return massive_umap_rank_epoch(*rank_graphs[rank],
              context.get(), rank, owner_rows, chunk_rows,
              edge_capacity, min_weight, max_weight, epoch);
          }));
      }
      double graph_peak = 0;
      double step_peak = 0;
      for (int rank = 0; rank < device_ids.size(); ++rank) {
        const auto progress = pending[rank].get();
        graph_peak = std::max(graph_peak,
          progress.graph_seconds);
        step_peak = std::max(step_peak,
          progress.step_seconds);
        edge_visits += progress.visits;
        per_device_visits[rank] += progress.visits;
        per_device_rows[rank] += progress.rows;
      }
      graph_visit_seconds += graph_peak;
      device_step_seconds += step_peak;
      owner_wall_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - owner_started).count();
      Rcpp::checkUserInterrupt();
    } else
#endif
    for (std::uint64_t first = 1; first <= graph.vertices();) {
      auto rows = graph.bounded_rows(first, chunk_rows,
                                    edge_capacity);
      const auto rank = static_cast<int>((first - 1) / owner_rows);
      if (multi_device) rows = std::min<std::uint64_t>(rows,
        (static_cast<std::uint64_t>(rank) + 1) * owner_rows - first + 1);
      int active = 0;
      const auto visit_started = std::chrono::steady_clock::now();
      graph.visit(first, rows,
        [&](std::uint64_t source, std::uint32_t target,
            float weight, std::uint64_t) {
          if (weight < min_weight) return;
          heads[active] = static_cast<int>(source - 1);
          tails[active] = static_cast<int>(target - 1);
          weights[active] = weight;
          periods[active] = max_weight /
            std::max(weight, 1.0e-6f);
          ++active;
        });
      graph_visit_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - visit_started).count();
      if (active > 0) {
        int step_status = 1;
        const auto step_started = std::chrono::steady_clock::now();
#ifdef FASTEMBEDR_HAS_NCCL
        if (multi_device) step_status =
          fastembedr_massive_umap_multigpu_step(context.get(), rank,
            heads.data(), tails.data(), weights.data(),
            periods.data(), active, epoch);
        else
#endif
          step_status = fastembedr_massive_umap_cuda_step(
            context.get(), heads.data(), tails.data(),
            weights.data(), periods.data(), active, epoch);
        massive_umap_cuda_check(step_status);
        device_step_seconds += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - step_started).count();
        edge_visits += active;
        if (device_ids.size() > 0)
          per_device_visits[rank] += active;
      }
      first += rows;
      Rcpp::checkUserInterrupt();
    }
    int epoch_status = 1;
#ifdef FASTEMBEDR_HAS_NCCL
    if (multi_device) {
      const auto began = std::chrono::steady_clock::now();
      epoch_status = fastembedr_massive_umap_multigpu_finish_epoch(
        context.get());
      finalization_block_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - began).count();
    } else
#endif
      epoch_status = fastembedr_massive_umap_cuda_finish_epoch(
        context.get());
    massive_umap_cuda_check(epoch_status);
    report_graph_progress("CUDA UMAP epochs", epoch + 1,
                          n_epochs, next_percent);
    if (checkpoint_every > 0 &&
        ((epoch + 1) % checkpoint_every == 0 ||
         epoch + 1 == n_epochs)) {
      const auto snapshot = output + ".epoch_" +
        std::to_string(epoch + 1) + ".f32";
      massive_umap_cuda_write(context.get(), snapshot,
                              graph.vertices(), chunk_rows,
                              dimensions, multi_device);
      Rcpp::Function callback(checkpoint_callback);
      callback(epoch + 1, snapshot, 0.0, 0.0, edge_visits);
    }
  }
  massive_umap_cuda_write(context.get(), output, graph.vertices(),
                          chunk_rows, dimensions, multi_device);
  double synchronization_seconds = 0;
#ifdef FASTEMBEDR_HAS_NCCL
  if (multi_device) synchronization_seconds =
    fastembedr_massive_umap_multigpu_collective_seconds(
      context.get());
#endif
  return List::create(
    Rcpp::Named("backend_used") = multi_device ?
      "native_cuda_umap_coordinated_nccl" :
      "native_cuda_umap_stream",
    Rcpp::Named("devices") = device_ids,
    Rcpp::Named("n_vertices") = static_cast<double>(graph.vertices()),
    Rcpp::Named("n_edges") = static_cast<double>(graph.edges()),
    Rcpp::Named("edge_visits") = edge_visits,
    Rcpp::Named("max_weight") = static_cast<double>(max_weight),
    Rcpp::Named("per_device_edge_visits") = per_device_visits,
    Rcpp::Named("per_device_row_visits") = per_device_rows,
    Rcpp::Named("delta_scale") = delta_scale,
    Rcpp::Named("graph_visit_seconds") = graph_visit_seconds,
    Rcpp::Named("device_step_seconds") = device_step_seconds,
    Rcpp::Named("owner_wall_seconds") = owner_wall_seconds,
    Rcpp::Named("synchronization_seconds") =
      synchronization_seconds,
    Rcpp::Named("finalization_block_seconds") =
      finalization_block_seconds,
    Rcpp::Named("positive_updates") = NA_REAL,
    Rcpp::Named("negative_updates") = NA_REAL);
#endif
}

#ifdef FASTEMBEDR_HAS_CUDA
namespace {

void massive_tsne_cuda_check(int status) {
  if (status != 0) Rcpp::stop(std::string(
    "Massive CUDA t-SNE failed: ") +
    fastembedr_cuda_embedding_last_error());
}

int massive_tsne_cuda_upload(void* context, bool multi, int field,
                             int first, int rows, const float* values) {
#ifdef FASTEMBEDR_HAS_NCCL
  if (multi) return fastembedr_massive_tsne_multigpu_upload(
    context, field, first, rows, values);
#endif
  return fastembedr_massive_tsne_cuda_upload(
    context, field, first, rows, values);
}

int massive_tsne_cuda_download(void* context, bool multi, int field,
                               int first, int rows, float* values) {
#ifdef FASTEMBEDR_HAS_NCCL
  if (multi) return fastembedr_massive_tsne_multigpu_download(
    context, field, first, rows, values);
#endif
  return fastembedr_massive_tsne_cuda_download(
    context, field, first, rows, values);
}

void massive_tsne_cuda_write(void* context,
                             const std::string& path,
                             int n, int fields, bool multi) {
  constexpr int block_rows = 8192;
  std::vector<float> buffer(block_rows * 2);
  Float32Sink sink(path);
  for (int field = 0; field < fields; ++field) {
    for (int first = 0; first < n; first += block_rows) {
      const int rows = std::min(block_rows, n - first);
      massive_tsne_cuda_check(massive_tsne_cuda_download(context,
        multi, field, first, rows, buffer.data()));
      for (int i = 0; i < rows * 2; ++i) {
        if (!std::isfinite(buffer[i]) ||
            (field == 2 && buffer[i] <= 0.0f)) {
          Rcpp::stop("Massive CUDA t-SNE state is non-finite.");
        }
      }
      sink.append(buffer.data(), static_cast<std::uint64_t>(rows) * 2);
    }
  }
  sink.finish(checked_product(
    checked_product(n, fields * 2), sizeof(float)));
}

void massive_tsne_cuda_load(void* context, FileMatrixSource& init,
                            int n, int start_iter,
                            const std::string& state_path,
                            bool multi) {
  constexpr int block_rows = 8192;
  std::vector<float> buffer(block_rows * 2);
  const auto layout_bytes = checked_product(n, 8);
  if (start_iter > 0) {
    std::error_code error;
    const auto bytes = std::filesystem::file_size(state_path, error);
    if (error || bytes != 3 * layout_bytes) {
      Rcpp::stop("Massive CUDA t-SNE checkpoint size is invalid.");
    }
    std::ifstream input(state_path, std::ios::binary);
    for (int field = 0; field < 3; ++field) {
      for (int first = 0; first < n; first += block_rows) {
        const int rows = std::min(block_rows, n - first);
        const auto bytes_read = static_cast<std::streamsize>(rows * 8);
        input.read(reinterpret_cast<char*>(buffer.data()), bytes_read);
        if (!input) Rcpp::stop(
          "Massive CUDA t-SNE checkpoint read failed.");
        for (int i = 0; i < rows * 2; ++i) {
          if (!std::isfinite(buffer[i]) ||
              (field == 2 && buffer[i] <= 0.0f)) {
            Rcpp::stop("Massive CUDA t-SNE checkpoint is invalid.");
          }
        }
        massive_tsne_cuda_check(massive_tsne_cuda_upload(context,
          multi, field, first, rows, buffer.data()));
      }
    }
    return;
  }
  double sums[2] = {0.0, 0.0};
  for (int first = 0; first < n; first += block_rows) {
    const int rows = std::min(block_rows, n - first);
    init.read_rows(first, rows, buffer.data());
    for (int i = 0; i < rows; ++i) {
      for (int axis = 0; axis < 2; ++axis) {
        const float value = buffer[2 * i + axis];
        if (!std::isfinite(value)) Rcpp::stop(
          "Massive CUDA t-SNE initialization is non-finite.");
        sums[axis] += value;
      }
    }
  }
  const float means[2] = {
    static_cast<float>(sums[0] / n),
    static_cast<float>(sums[1] / n)
  };
  for (int first = 0; first < n; first += block_rows) {
    const int rows = std::min(block_rows, n - first);
    init.read_rows(first, rows, buffer.data());
    for (int i = 0; i < rows; ++i) {
      buffer[2 * i] -= means[0];
      buffer[2 * i + 1] -= means[1];
    }
    massive_tsne_cuda_check(massive_tsne_cuda_upload(context,
      multi, 0, first, rows, buffer.data()));
  }
}

#ifdef FASTEMBEDR_HAS_NCCL
struct TsneRankProgress {
  double graph_seconds = 0;
  double attraction_seconds = 0;
  double edges = 0;
};

TsneRankProgress massive_tsne_rank_attraction(
    CsrGraphSource& graph, void* context, int rank,
    std::uint64_t owner_rows, int edge_capacity,
    float exaggeration) {
  TsneRankProgress progress;
  std::vector<int> heads(edge_capacity), tails(edge_capacity);
  std::vector<float> weights(edge_capacity);
  const auto first_row = static_cast<std::uint64_t>(rank) *
    owner_rows + 1;
  const auto last_row = std::min(graph.vertices(),
    first_row + owner_rows - 1);
  for (auto first = first_row; first <= last_row;) {
    const auto rows = std::min(graph.bounded_rows(first, 8192,
      edge_capacity), last_row - first + 1);
    int active = 0;
    const auto read_started = std::chrono::steady_clock::now();
    graph.visit(first, rows,
      [&](std::uint64_t source, std::uint32_t target,
          float weight, std::uint64_t) {
        heads[active] = static_cast<int>(source - 1);
        tails[active] = static_cast<int>(target - 1);
        weights[active] = weight;
        ++active;
      }, false);
    progress.graph_seconds += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - read_started).count();
    if (active > 0) {
      const auto attraction_started = std::chrono::steady_clock::now();
      const int status = fastembedr_massive_tsne_multigpu_attract(
        context, rank, heads.data(), tails.data(), weights.data(),
        active, exaggeration);
      if (status != 0) throw std::runtime_error(
        fastembedr_cuda_embedding_last_error());
      progress.attraction_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - attraction_started).count();
      progress.edges += active;
    }
    first += rows;
  }
  return progress;
}
#endif

}  // namespace
#endif

// [[Rcpp::export]]
List massive_tsne_optimize_cuda_cpp(
    std::string offsets_path, std::string indices_path,
    std::string weights_path, std::string access,
    std::string init_path, std::string output_path,
    int n, int dims, int early_iter, int normal_iter,
    double early_exaggeration, double exaggeration,
    double learning_rate, bool learning_rate_auto,
    double initial_momentum, double final_momentum,
    double min_gain, double max_step_norm, int start_iter,
    std::string state_path, int checkpoint_every,
    SEXP checkpoint_callback, int edge_capacity,
    Rcpp::IntegerVector devices) {
#ifndef FASTEMBEDR_HAS_CUDA
  Rcpp::stop("Native massive CUDA t-SNE is unavailable; no CPU fallback.");
#else
  require_little_endian();
  if (n < 2 || n > INT_MAX / 2 || dims != 2 ||
      early_iter < 0 || normal_iter < 0 ||
      static_cast<double>(early_iter) + normal_iter < 1 ||
      static_cast<double>(early_iter) + normal_iter > INT_MAX ||
      !std::isfinite(early_exaggeration) ||
      early_exaggeration <= 0.0 ||
      !std::isfinite(exaggeration) || exaggeration <= 0.0 ||
      (!learning_rate_auto && (!std::isfinite(learning_rate) ||
        learning_rate <= 0.0)) ||
      !std::isfinite(initial_momentum) || initial_momentum < 0.0 ||
      !std::isfinite(final_momentum) || final_momentum < 0.0 ||
      !std::isfinite(min_gain) || min_gain <= 0.0 ||
      max_step_norm <= 0.0 || edge_capacity < 1 ||
      start_iter < 0 || start_iter > early_iter + normal_iter ||
      checkpoint_every < 0 ||
      (start_iter > 0 && (state_path.empty() ||
        checkpoint_every == 0)) ||
      (start_iter == 0 && !state_path.empty()) ||
      (checkpoint_every > 0 && checkpoint_callback == R_NilValue) ||
      (checkpoint_every == 0 && checkpoint_callback != R_NilValue)) {
    Rcpp::stop("Invalid massive CUDA t-SNE controls.");
  }
  if (devices.size() < 1 || devices.size() > 4 ||
      devices.size() > n) {
    Rcpp::stop("CUDA t-SNE requires one to four distinct devices.");
  }
  for (int rank = 0; rank < devices.size(); ++rank) {
    const int device = devices[rank];
    if (device == NA_INTEGER || device < 0 ||
        device >= massive_cuda_device_count_cpp()) {
      Rcpp::stop("CUDA t-SNE device is unavailable.");
    }
    for (int other = 0; other < rank; ++other) {
      if (device == devices[other]) {
        Rcpp::stop("CUDA t-SNE devices must be distinct.");
      }
    }
  }
  const bool multi = devices.size() > 1;
#ifndef FASTEMBEDR_HAS_NCCL
  if (multi) Rcpp::stop(
    "Multi-GPU t-SNE requires NCCL; no fallback was used.");
#endif
  CsrGraphSource graph(offsets_path, indices_path, weights_path,
                       n, access);
  FileMatrixSource init(init_path, "f32", "stream", n, dims);
  if (std::filesystem::exists(output_path) ||
      std::filesystem::exists(output_path + ".part")) {
    Rcpp::stop("Massive CUDA t-SNE output already exists.");
  }
  struct DeviceRestore {
    int previous;
    ~DeviceRestore() {
      fastembedr_massive_cuda_select_device(previous);
    }
  } restore{Rcpp::as<int>(massive_cuda_memory_cpp()["device"])};
  massive_cuda_select_cpp(devices[0]);
  void* raw = nullptr;
#ifdef FASTEMBEDR_HAS_NCCL
  if (multi) raw = fastembedr_massive_tsne_multigpu_create(
    n, edge_capacity, devices.begin(), devices.size());
#endif
  if (!multi) raw = fastembedr_massive_tsne_cuda_create(
    n, edge_capacity);
  if (raw == nullptr) massive_tsne_cuda_check(1);
  auto destroy = [multi](void* pointer) {
#ifdef FASTEMBEDR_HAS_NCCL
    if (multi) {
      fastembedr_massive_tsne_multigpu_destroy(pointer);
      return;
    }
#endif
    fastembedr_massive_tsne_cuda_destroy(pointer);
  };
  std::unique_ptr<void, decltype(destroy)> context(raw, destroy);
  massive_tsne_cuda_load(context.get(), init, n, start_iter,
                         state_path, multi);
  std::vector<int> heads(multi ? 0 : edge_capacity);
  std::vector<int> tails(multi ? 0 : edge_capacity);
  std::vector<float> weights(multi ? 0 : edge_capacity);
  const int total = early_iter + normal_iter;
  int next_percent = 10;
  // The first attraction sync may include the queued FFT repulsion.
  double repulsion_enqueue_seconds = 0.0;
  double repulsion_gpu_seconds = 0.0;
  double graph_visit_seconds = 0.0;
  double attraction_sync_seconds = 0.0;
  double update_sync_seconds = 0.0;
  double owner_wall_seconds = 0.0;
  std::vector<double> attraction_edges(devices.size(), 0.0);
  const auto owner_rows = static_cast<std::uint64_t>(
    (n - 1) / devices.size() + 1);
#ifdef FASTEMBEDR_HAS_NCCL
  std::vector<std::unique_ptr<CsrGraphSource>> rank_graphs;
  if (multi) {
    for (int rank = 0; rank < devices.size(); ++rank) {
      rank_graphs.emplace_back(std::make_unique<CsrGraphSource>(
        offsets_path, indices_path, weights_path, n, access));
    }
  }
#endif
  const auto started = std::chrono::steady_clock::now();
  for (int iter = start_iter; iter < total; ++iter) {
    const bool early = iter < early_iter;
    const float phase_exag = static_cast<float>(early ?
      early_exaggeration : exaggeration);
    const float phase_lr = static_cast<float>(learning_rate_auto ?
      static_cast<double>(n) / phase_exag : learning_rate);
    const float momentum = static_cast<float>(early ?
      initial_momentum : final_momentum);
    const auto repulsion_started = std::chrono::steady_clock::now();
    int begin_status = 0;
#ifdef FASTEMBEDR_HAS_NCCL
    if (multi) begin_status = fastembedr_massive_tsne_multigpu_begin(
      context.get());
#endif
    if (!multi) begin_status = fastembedr_massive_tsne_cuda_begin(
      context.get());
    massive_tsne_cuda_check(begin_status);
    repulsion_enqueue_seconds += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - repulsion_started).count();
#ifdef FASTEMBEDR_HAS_NCCL
    if (multi) {
      const auto owner_started = std::chrono::steady_clock::now();
      std::vector<std::future<TsneRankProgress>> pending;
      for (int rank = 0; rank < devices.size(); ++rank) {
        pending.emplace_back(std::async(std::launch::async,
          [&, rank]() {
            return massive_tsne_rank_attraction(*rank_graphs[rank],
              context.get(), rank, owner_rows, edge_capacity,
              phase_exag);
          }));
      }
      double graph_peak = 0;
      double attraction_peak = 0;
      for (int rank = 0; rank < devices.size(); ++rank) {
        const auto progress = pending[rank].get();
        graph_peak = std::max(graph_peak, progress.graph_seconds);
        attraction_peak = std::max(attraction_peak,
          progress.attraction_seconds);
        attraction_edges[rank] += progress.edges;
      }
      graph_visit_seconds += graph_peak;
      attraction_sync_seconds += attraction_peak;
      owner_wall_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - owner_started).count();
      Rcpp::checkUserInterrupt();
    } else
#endif
    for (std::uint64_t first = 1; first <= graph.vertices();) {
      const auto read_started = std::chrono::steady_clock::now();
      const auto rows = graph.bounded_rows(first, 8192,
        edge_capacity);
      int active = 0;
      graph.visit(first, rows,
        [&](std::uint64_t source, std::uint32_t target,
            float weight, std::uint64_t) {
          heads[active] = static_cast<int>(source - 1);
          tails[active] = static_cast<int>(target - 1);
          weights[active] = weight;
          ++active;
        });
      graph_visit_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - read_started).count();
      if (active > 0) {
        const auto attract_started = std::chrono::steady_clock::now();
        const int status = fastembedr_massive_tsne_cuda_attract(
          context.get(), heads.data(), tails.data(),
          weights.data(), active, phase_exag);
        massive_tsne_cuda_check(status);
        attraction_edges[0] += active;
        attraction_sync_seconds += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - attract_started).count();
      }
      first += rows;
      Rcpp::checkUserInterrupt();
    }
    const auto update_started = std::chrono::steady_clock::now();
    int finish_status = 0;
#ifdef FASTEMBEDR_HAS_NCCL
    if (multi) finish_status = fastembedr_massive_tsne_multigpu_finish(
      context.get(), phase_lr, momentum,
      static_cast<float>(min_gain),
      static_cast<float>(max_step_norm));
#endif
    if (!multi) finish_status = fastembedr_massive_tsne_cuda_finish(
      context.get(), phase_lr, momentum,
      static_cast<float>(min_gain),
      static_cast<float>(max_step_norm));
    massive_tsne_cuda_check(finish_status);
    float repulsion_ms = 0.0f;
    int timing_status = 0;
#ifdef FASTEMBEDR_HAS_NCCL
    if (multi) timing_status =
      fastembedr_massive_tsne_multigpu_repulsion_ms(
        context.get(), &repulsion_ms);
#endif
    if (!multi) timing_status = fastembedr_massive_tsne_cuda_repulsion_ms(
      context.get(), &repulsion_ms);
    massive_tsne_cuda_check(timing_status);
    if (!std::isfinite(repulsion_ms) || repulsion_ms < 0.0f) {
      Rcpp::stop("CUDA t-SNE FFT timing is invalid.");
    }
    repulsion_gpu_seconds += repulsion_ms / 1000.0;
    update_sync_seconds += std::chrono::duration<double>(
      std::chrono::steady_clock::now() - update_started).count();
    if (checkpoint_every > 0 &&
        ((iter + 1) % checkpoint_every == 0 || iter + 1 == total)) {
      const auto snapshot = output_path + ".iter_" +
        std::to_string(iter + 1) + ".state.f32";
#ifdef FASTEMBEDR_HAS_NCCL
      if (multi) massive_tsne_cuda_check(
        fastembedr_massive_tsne_multigpu_gather_state(context.get()));
#endif
      massive_tsne_cuda_write(context.get(), snapshot, n, 3,
                              multi);
      Rcpp::Function callback(checkpoint_callback);
      const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
      callback(iter + 1, snapshot, elapsed);
    }
    report_graph_progress("CUDA t-SNE iterations", iter + 1,
                          total, next_percent);
  }
  std::vector<double> expected_edges(devices.size(), 0.0);
  if (multi) for (int rank = 0; rank < devices.size(); ++rank) {
    const auto first = static_cast<std::uint64_t>(rank) * owner_rows;
    const auto last = std::min<std::uint64_t>(n, first + owner_rows);
    expected_edges[rank] = static_cast<double>(
      graph.offset(last) - graph.offset(first)) *
      (total - start_iter);
    if (attraction_edges[rank] != expected_edges[rank]) Rcpp::stop(
      "Coordinated CUDA t-SNE missed owner attraction edges.");
  }
  const auto output_started = std::chrono::steady_clock::now();
  massive_tsne_cuda_write(context.get(), output_path, n, 1,
                          multi);
  const auto output_seconds = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - output_started).count();
  const auto elapsed = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - started).count();
  return List::create(
    Rcpp::Named("backend_used") = multi ?
      "native_cuda_fft_graph_coordinated_nccl" :
      "native_cuda_fft_graph_stream",
    Rcpp::Named("repulsion") = "fft_grid_cuda_cufft",
    Rcpp::Named("fft_grid_size") =
      fastembedr_cuda_opentsne_fft_grid_size(n),
    Rcpp::Named("iterations") = total,
    Rcpp::Named("resumed_from_iteration") = start_iter,
    Rcpp::Named("n_threads") = 1,
    Rcpp::Named("devices") = devices,
    Rcpp::Named("per_device_attraction_edges") = attraction_edges,
    Rcpp::Named("per_device_expected_edges") = expected_edges,
    Rcpp::Named("owner_wall_seconds") = owner_wall_seconds,
    Rcpp::Named("repulsion_gpu_seconds") = repulsion_gpu_seconds,
    Rcpp::Named("stage_seconds") = List::create(
      Rcpp::Named("repulsion_enqueue") = repulsion_enqueue_seconds,
      Rcpp::Named("graph_visit") = graph_visit_seconds,
      Rcpp::Named("attraction_sync") = attraction_sync_seconds,
      Rcpp::Named("update_sync") = update_sync_seconds,
      Rcpp::Named("output_write") = output_seconds),
    Rcpp::Named("elapsed_seconds") = elapsed);
#endif
}

// [[Rcpp::export]]
List massive_read_graph_edges_cpp(std::string indices_path,
                                  std::string distances_path,
                                  double n_vertices, int k,
                                  std::string access, double first_row,
                                  int count, double max_bytes,
                                  std::string kind) {
  KnnGraphSource graph(indices_path, distances_path,
                       n_vertices, k, access, kind);
  const auto first = whole_number(first_row, "first_row");
  if (count < 1 || !std::isfinite(max_bytes) || max_bytes <= 0) {
    Rcpp::stop("Invalid massive graph read count or byte limit.");
  }
  if (first < 1 || first > graph.vertices() ||
      static_cast<std::uint64_t>(count) > graph.vertices() - first + 1) {
    Rcpp::stop("Invalid massive graph row range.");
  }
  const auto items = checked_product(count, k);
  if (items > static_cast<std::uint64_t>(max_bytes / 20) ||
      items > static_cast<std::uint64_t>(R_XLEN_T_MAX)) {
    Rcpp::stop("Massive graph read exceeds its byte limit.");
  }
  Rcpp::NumericVector from(static_cast<R_xlen_t>(items));
  Rcpp::IntegerVector to(static_cast<R_xlen_t>(items));
  Rcpp::NumericVector distance(static_cast<R_xlen_t>(items));
  graph.visit(first, count,
              [&](std::uint64_t source, std::uint32_t target,
                  float value, std::uint64_t row, int col) {
    const auto pos = static_cast<R_xlen_t>(row * k + col);
    from[pos] = static_cast<double>(source);
    to[pos] = static_cast<int>(target);
    distance[pos] = value;
  });
  if (kind == "weight") {
    return List::create(Rcpp::Named("from") = from,
                        Rcpp::Named("to") = to,
                        Rcpp::Named("weight") = distance);
  }
  return List::create(Rcpp::Named("from") = from,
                      Rcpp::Named("to") = to,
                      Rcpp::Named("distance") = distance);
}

// [[Rcpp::export]]
List massive_validate_csr_graph_cpp(std::string offsets_path,
                                    std::string indices_path,
                                    std::string weights_path,
                                    double n_vertices) {
  CsrGraphSource graph(offsets_path, indices_path, weights_path,
                       n_vertices, "stream");
  std::uint64_t max_degree = 0;
  for (std::uint64_t first = 1; first <= graph.vertices();) {
    const auto begin = graph.offset(first - 1);
    const auto end = graph.offset(first);
    const auto budget = std::max<std::uint64_t>(65536,
      end >= begin ? end - begin : 0);
    const auto count = graph.bounded_rows(first, 8192, budget);
    max_degree = std::max(max_degree, graph.visit(
      first, count, [](std::uint64_t, std::uint32_t, float,
                       std::uint64_t) {}));
    first += count;
    Rcpp::checkUserInterrupt();
  }
  return List::create(
    Rcpp::Named("n_vertices") = static_cast<double>(graph.vertices()),
    Rcpp::Named("n_edges") = static_cast<double>(graph.edges()),
    Rcpp::Named("max_degree") = static_cast<double>(max_degree),
    Rcpp::Named("validated") = true
  );
}

// [[Rcpp::export]]
List massive_read_csr_graph_edges_cpp(std::string offsets_path,
                                      std::string indices_path,
                                      std::string weights_path,
                                      double n_vertices,
                                      std::string access,
                                      double first_row, int count,
                                      double max_bytes) {
  CsrGraphSource graph(offsets_path, indices_path, weights_path,
                       n_vertices, access);
  const auto first = whole_number(first_row, "first_row");
  if (count < 1 || first < 1 || first > graph.vertices() ||
      static_cast<std::uint64_t>(count) > graph.vertices() - first + 1 ||
      !std::isfinite(max_bytes) || max_bytes <= 0) {
    Rcpp::stop("Invalid CSR graph read range or byte limit.");
  }
  const auto begin = graph.offset(first - 1);
  const auto end = graph.offset(first + count - 1);
  if (end < begin || end > graph.edges() ||
      end - begin > static_cast<std::uint64_t>(max_bytes / 20) ||
      end - begin > static_cast<std::uint64_t>(R_XLEN_T_MAX)) {
    Rcpp::stop("CSR graph read exceeds its byte limit.");
  }
  const auto items = static_cast<R_xlen_t>(end - begin);
  Rcpp::NumericVector from(items);
  Rcpp::IntegerVector to(items);
  Rcpp::NumericVector weight(items);
  graph.visit(first, count,
              [&](std::uint64_t source, std::uint32_t target,
                  float value, std::uint64_t position) {
    const auto pos = static_cast<R_xlen_t>(position);
    from[pos] = static_cast<double>(source);
    to[pos] = static_cast<int>(target);
    weight[pos] = value;
  });
  return List::create(Rcpp::Named("from") = from,
                      Rcpp::Named("to") = to,
                      Rcpp::Named("weight") = weight);
}

namespace {

List massive_exact_reference_batch_impl(List query_spec,
                                        List reference_spec,
                                        double first_row, int count,
                                        int k, int reference_chunk,
                                        int n_threads, bool exclude_self,
                                        const std::vector<std::uint64_t>*
                                          selected = nullptr) {
  auto query = make_source(query_spec);
  auto reference = make_source(reference_spec);
  const auto first_id = whole_number(first_row, "first_row");
  const auto n = query->nrow();
  const auto reference_rows = reference->nrow();
  const auto p = query->ncol();
  if (n < 1 || n > 9007199254740991ULL ||
      reference_rows < (exclude_self ? 2U : 1U) ||
      reference_rows > static_cast<std::uint64_t>(
        std::numeric_limits<std::int32_t>::max()) ||
      p < 1 || p > static_cast<std::uint64_t>(
        std::numeric_limits<int>::max()) ||
      first_id < 1 || first_id > n || count < 1 ||
      static_cast<std::uint64_t>(count) > n - first_id + 1 ||
      k < 1 || static_cast<std::uint64_t>(k) >
        reference_rows - (exclude_self ? 1U : 0U) || k > 65536 ||
      reference_chunk < 1 || n_threads < 1 ||
      checked_product(count, k) > 128000000 / 12 ||
      checked_product(checked_product(count, p), 4) > 256000000 ||
      checked_product(checked_product(reference_chunk, p), 4) >
        256000000) {
    Rcpp::stop("Invalid streamed exact KNN dimensions or chunk size.");
  }
  const auto first = first_id - 1;
  if (reference->ncol() != p || (exclude_self && reference_rows != n)) {
    Rcpp::stop("Exact KNN query and reference dimensions differ.");
  }
  std::vector<float> queries(checked_product(count, p));
  std::vector<float> refs(checked_product(reference_chunk, p));
  if (selected) {
    for (int qi = 0; qi < count; ++qi) {
      query->read_rows((*selected)[qi], 1,
                       queries.data() + static_cast<std::uint64_t>(qi) * p);
    }
  } else {
    query->read_rows(first, count, queries.data());
  }
  using Neighbor = std::pair<float, std::uint32_t>;
  const auto closer = [](const Neighbor& a, const Neighbor& b) {
    return a.first < b.first ||
      (a.first == b.first && a.second < b.second);
  };
  std::vector<std::vector<Neighbor>> heaps(count);
  for (auto& heap : heaps) heap.reserve(k);
  std::atomic<bool> invalid(false);
  for (std::uint64_t start = 0; start < reference_rows;) {
    const auto rows = std::min<std::uint64_t>(reference_chunk,
                                              reference_rows - start);
    reference->read_rows(start, rows, refs.data());
    parallel_rows(count, n_threads,
                  [&](std::uint64_t begin, std::uint64_t end, int) {
      for (auto qi = begin; qi < end; ++qi) {
        auto& heap = heaps[static_cast<std::size_t>(qi)];
        const float* q = queries.data() + qi * p;
        for (std::uint64_t ri = 0; ri < rows; ++ri) {
          const auto query_id = selected ? (*selected)[qi] : first + qi;
          if (exclude_self && query_id == start + ri) continue;
          const float distance = fastembedr::squared_l2_distance(
            q, refs.data() + ri * p, static_cast<int>(p));
          if (!std::isfinite(distance)) {
            invalid.store(true, std::memory_order_relaxed);
            continue;
          }
          const Neighbor candidate{
            distance, static_cast<std::uint32_t>(start + ri + 1)};
          if (static_cast<int>(heap.size()) < k) {
            heap.push_back(candidate);
            std::push_heap(heap.begin(), heap.end(), closer);
          } else if (closer(candidate, heap.front())) {
            std::pop_heap(heap.begin(), heap.end(), closer);
            heap.back() = candidate;
            std::push_heap(heap.begin(), heap.end(), closer);
          }
        }
      }
    });
    if (invalid.load(std::memory_order_relaxed)) {
      Rcpp::stop("Streamed exact KNN distance overflowed float32.");
    }
    start += rows;
    Rcpp::checkUserInterrupt();
  }
  Rcpp::IntegerMatrix indices(count, k);
  Rcpp::NumericMatrix distances(count, k);
  for (int qi = 0; qi < count; ++qi) {
    auto& heap = heaps[static_cast<std::size_t>(qi)];
    if (static_cast<int>(heap.size()) != k) {
      Rcpp::stop("Streamed exact search found fewer than k neighbors.");
    }
    std::sort(heap.begin(), heap.end(), closer);
    for (int rank = 0; rank < k; ++rank) {
      indices(qi, rank) = heap[rank].second;
      distances(qi, rank) = std::sqrt(
        std::max(0.0f, heap[rank].first));
    }
  }
  return List::create(Rcpp::Named("indices") = indices,
                      Rcpp::Named("distances") = distances,
                      Rcpp::Named("backend_used") =
                        "native_cpu_exact_stream",
                      Rcpp::Named("exact") = true);
}

}  // namespace

// [[Rcpp::export]]
List massive_exact_graph_batch_cpp(List spec, double first_row,
                                   int count, int k,
                                   int reference_chunk, int n_threads) {
  return massive_exact_reference_batch_impl(spec, spec, first_row,
    count, k, reference_chunk, n_threads, true);
}

// [[Rcpp::export]]
List massive_exact_reference_batch_cpp(List query_spec,
                                       List reference_spec,
                                       double first_row, int count,
                                       int k, int reference_chunk,
                                       int n_threads) {
  return massive_exact_reference_batch_impl(query_spec, reference_spec,
    first_row, count, k, reference_chunk, n_threads, false);
}

// [[Rcpp::export]]
List massive_exact_sample_batch_cpp(List query_spec,
                                    List reference_spec,
                                    Rcpp::NumericVector row_ids, int k,
                                    int reference_chunk, int n_threads,
                                    bool exclude_self) {
  if (row_ids.size() < 1 || row_ids.size() > 10000) {
    Rcpp::stop("Exact KNN sample needs 1 to 10,000 query rows.");
  }
  std::vector<std::uint64_t> selected;
  selected.reserve(row_ids.size());
  for (const double value : row_ids) {
    if (value < 1) Rcpp::stop("row_ids must be one-based.");
    selected.push_back(whole_number(value, "row_ids") - 1);
  }
  return massive_exact_reference_batch_impl(query_spec, reference_spec,
    1, static_cast<int>(selected.size()), k, reference_chunk,
    n_threads, exclude_self, &selected);
}

// [[Rcpp::export]]
List massive_vote_landmarks_cpp(Rcpp::IntegerMatrix indices,
                               NumericMatrix distances,
                               Rcpp::IntegerVector landmark_labels,
                               int n_threads) {
  const int rows = indices.nrow();
  const int k = indices.ncol();
  const int landmarks = landmark_labels.size();
  if (rows < 1 || k < 1 || landmarks < 1 || n_threads < 1 ||
      distances.nrow() != rows || distances.ncol() != k) {
    Rcpp::stop("Invalid landmark vote dimensions or worker count.");
  }
  int communities = 0;
  for (int i = 0; i < landmarks; ++i) {
    const int label = landmark_labels[i];
    if (label < 1 || label > landmarks) {
      Rcpp::stop("Landmark labels must be one-based community IDs.");
    }
    communities = std::max(communities, label);
  }
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < k; ++col) {
      const int id = indices(row, col);
      const double distance = distances(row, col);
      if (id < 1 || id > landmarks || !std::isfinite(distance) ||
          distance < 0) {
        Rcpp::stop("Landmark vote has an invalid neighbor or distance.");
      }
    }
  }
  Rcpp::IntegerVector membership(rows);
  NumericVector confidence(rows);
  parallel_rows(rows, n_threads, [&](std::uint64_t begin,
                                     std::uint64_t end, int) {
    const bool sparse = communities > 65536;
    std::vector<double> dense_votes;
    std::unordered_map<int, double> votes;
    if (sparse) {
      votes.reserve(k);
    } else {
      dense_votes.resize(static_cast<std::size_t>(communities) + 1, 0.0);
    }
    std::vector<int> touched;
    touched.reserve(k);
    for (std::uint64_t row = begin; row < end; ++row) {
      if (sparse) votes.clear();
      touched.clear();
      double total = 0.0;
      for (int col = 0; col < k; ++col) {
        const int label = landmark_labels[indices(row, col) - 1];
        const double weight = 1.0 / (distances(row, col) + 1e-3);
        if (sparse) {
          const auto entry = votes.emplace(label, 0.0);
          if (entry.second) touched.push_back(label);
          entry.first->second += weight;
        } else {
          if (dense_votes[label] == 0.0) touched.push_back(label);
          dense_votes[label] += weight;
        }
        total += weight;
      }
      int best = landmarks;
      double strongest = -1.0;
      for (const int label : touched) {
        const double score = sparse ? votes.at(label) :
          dense_votes[label];
        if (score > strongest ||
            (score == strongest && label < best)) {
          best = label;
          strongest = score;
        }
        if (!sparse) dense_votes[label] = 0.0;
      }
      membership[row] = best;
      confidence[row] = strongest / total;
    }
  });
  return List::create(
    Rcpp::Named("membership") = membership,
    Rcpp::Named("confidence") = confidence
  );
}

// [[Rcpp::export]]
List massive_read_cluster_rows_cpp(std::string labels_path,
                                   std::string confidence_path,
                                   double n_rows,
                                   double first_row,
                                   int count,
                                   double max_bytes) {
  require_little_endian();
  const auto rows = whole_number(n_rows, "n_rows");
  const auto first = whole_number(first_row, "first_row");
  if (count < 1 || first < 1 || first > rows ||
      static_cast<std::uint64_t>(count) > rows - first + 1 ||
      !std::isfinite(max_bytes) || max_bytes < 24.0 * count) {
    Rcpp::stop("Invalid massive cluster row range or byte limit.");
  }
  const auto expected = checked_product(rows, 4);
  const auto offset = checked_product(first - 1, 4);
  if (offset > static_cast<std::uint64_t>(
      std::numeric_limits<std::streamoff>::max())) {
    Rcpp::stop("Massive cluster file offset exceeds stream limits.");
  }
  std::error_code error;
  const auto label_size = std::filesystem::file_size(labels_path, error);
  if (error || label_size != expected) {
    Rcpp::stop("Massive cluster label file size mismatch.");
  }
  const auto confidence_size = std::filesystem::file_size(
    confidence_path, error
  );
  if (error || confidence_size != expected) {
    Rcpp::stop("Massive cluster confidence file size mismatch.");
  }
  std::ifstream label_stream(labels_path, std::ios::binary);
  std::ifstream confidence_stream(confidence_path, std::ios::binary);
  label_stream.seekg(static_cast<std::streamoff>(offset));
  confidence_stream.seekg(static_cast<std::streamoff>(offset));
  std::vector<unsigned char> ids(checked_product(count, 4));
  std::vector<unsigned char> values(checked_product(count, 4));
  label_stream.read(reinterpret_cast<char*>(ids.data()), ids.size());
  confidence_stream.read(reinterpret_cast<char*>(values.data()),
                         values.size());
  if (!label_stream || !confidence_stream) {
    Rcpp::stop("Cannot read massive cluster row range.");
  }
  Rcpp::IntegerVector membership(count);
  NumericVector confidence(count);
  for (int i = 0; i < count; ++i) {
    const auto label = read_u32(ids.data() + 4 * i);
    float value = 0.0f;
    std::memcpy(&value, values.data() + 4 * i, 4);
    if (label < 1 || label > static_cast<std::uint32_t>(
        std::numeric_limits<int>::max()) ||
        !std::isfinite(value) || value < 0 || value > 1) {
      Rcpp::stop("Massive cluster output contains invalid values.");
    }
    membership[i] = static_cast<int>(label);
    confidence[i] = value;
  }
  return List::create(
    Rcpp::Named("membership") = membership,
    Rcpp::Named("confidence") = confidence
  );
}

namespace {

NumericVector pca_streamed_means(MatrixSource& source,
                                  int chunk_rows,
                                  int workers,
                                  bool center,
                                  std::vector<float>& block) {
  const auto n = source.nrow();
  const auto p = source.ncol();
  NumericVector means(p);
  if (!center) return means;
  std::vector<std::vector<double>> local_mean(
    workers, std::vector<double>(p, 0)
  );
  for (std::uint64_t start = 0; start < n; start += chunk_rows) {
    const auto count = std::min<std::uint64_t>(chunk_rows, n - start);
    source.read_rows(start, count, block.data());
    parallel_rows(count, workers, [&](std::uint64_t begin,
                                      std::uint64_t end, int worker) {
      auto& sums = local_mean[worker];
      for (auto row = begin; row < end; ++row) {
        const float* values = block.data() + row * p;
        for (std::uint64_t col = 0; col < p; ++col) {
          sums[col] += values[col];
        }
      }
    });
    Rcpp::checkUserInterrupt();
  }
  for (const auto& sums : local_mean) {
    for (std::uint64_t col = 0; col < p; ++col) {
      means[col] += sums[col] / static_cast<double>(n);
    }
  }
  Rcpp::Rcout << "  PCA mean pass: 100%\n";
  return means;
}

void check_massive_pca_source(const MatrixSource& source,
                              int chunk_rows, int n_threads,
                              std::uint64_t max_columns) {
  if (source.nrow() < 2 || source.ncol() > max_columns ||
      chunk_rows < 1 || n_threads < 1) {
    Rcpp::stop("Streamed PCA needs n >= 2, p <= ", max_columns,
               ", and positive chunk rows and threads.");
  }
}

std::vector<double> pca_cross_state(SEXP saved, double completed_rows,
                                    std::uint64_t n, int chunk_rows,
                                    int p, int workers,
                                    SEXP progress, int every,
                                    std::uint64_t& start) {
  start = whole_number(completed_rows, "completed_rows");
  if (start > n || (start != n && start % chunk_rows != 0) ||
      every < 0 || (progress == R_NilValue) != (every == 0) ||
      (start > 0 && (saved == R_NilValue || every == 0))) {
    Rcpp::stop("Invalid PCA covariance checkpoint controls.");
  }
  const auto width = checked_product(p, p);
  const auto length = checked_product(width, workers);
  std::vector<double> cross(length, 0.0);
  if (saved != R_NilValue) {
    NumericMatrix previous(saved);
    if (previous.nrow() != static_cast<int>(width) ||
        previous.ncol() != workers ||
        std::any_of(previous.begin(), previous.end(),
                    [](double x) { return !std::isfinite(x); })) {
      Rcpp::stop("Saved PCA covariance sums are invalid.");
    }
    std::copy(previous.begin(), previous.end(), cross.begin());
  }
  return cross;
}

void pca_cross_checkpoint(SEXP progress, std::uint64_t rows,
                          const std::vector<double>& cross,
                          int p, int workers) {
  if (progress == R_NilValue) return;
  NumericMatrix snapshot(checked_product(p, p), workers);
  std::copy(cross.begin(), cross.end(), snapshot.begin());
  Rcpp::Function callback(progress);
  callback(static_cast<double>(rows), snapshot);
}

}  // namespace

// [[Rcpp::export]]
NumericVector massive_pca_means_cpp(List spec, int chunk_rows,
                                    int n_threads, bool center) {
  auto source = make_source(spec);
  check_massive_pca_source(*source, chunk_rows, n_threads, 2048);
  const int workers = static_cast<int>(std::min<std::uint64_t>(
    n_threads, chunk_rows
  ));
  std::vector<float> block(checked_product(chunk_rows, source->ncol()));
  return pca_streamed_means(*source, chunk_rows, workers, center, block);
}

// [[Rcpp::export]]
NumericMatrix massive_pca_mean_chunk_cpp(List spec, double first,
                                          int rows, int n_threads,
                                          NumericMatrix previous) {
  auto source = make_source(spec);
  check_massive_pca_source(*source, rows, n_threads, 2048);
  if (previous.nrow() != n_threads ||
      previous.ncol() != static_cast<int>(source->ncol()) ||
      std::any_of(previous.begin(), previous.end(),
                  [](double value) { return !std::isfinite(value); })) {
    Rcpp::stop("Invalid saved PCA worker sums.");
  }
  const auto start = whole_number(first, "first");
  if (start < 1 || start > source->nrow() ||
      static_cast<std::uint64_t>(rows) > source->nrow() - start + 1) {
    Rcpp::stop("Invalid PCA mean-pass row range.");
  }
  const auto p = source->ncol();
  std::vector<float> block(checked_product(rows, p));
  source->read_rows(start - 1, rows, block.data());
  NumericMatrix sums = Rcpp::clone(previous);
  parallel_rows(rows, n_threads, [&](std::uint64_t begin,
                                     std::uint64_t end, int worker) {
    for (auto row = begin; row < end; ++row) {
      const float* values = block.data() + row * p;
      for (std::uint64_t col = 0; col < p; ++col) {
        sums(worker, col) += values[col];
      }
    }
  });
  return sums;
}

// [[Rcpp::export]]
List massive_pca_covariance_cpp(List spec,
                                int chunk_rows,
                                int n_threads,
                                bool center,
                                SEXP saved_means = R_NilValue,
                                double completed_rows = 0,
                                SEXP saved_cross = R_NilValue,
                                SEXP progress = R_NilValue,
                                int checkpoint_every = 0) {
  auto source = make_source(spec);
  const auto n = source->nrow();
  const auto p = source->ncol();
  check_massive_pca_source(*source, chunk_rows, n_threads, 1024);
  const int workers = static_cast<int>(std::min<std::uint64_t>(
    n_threads, chunk_rows
  ));
  const auto capacity = checked_product(chunk_rows, p);
  std::vector<float> block(static_cast<std::size_t>(capacity));
  NumericVector means = saved_means == R_NilValue ?
    pca_streamed_means(*source, chunk_rows, workers, center, block) :
    Rcpp::as<NumericVector>(saved_means);
  if (means.size() != static_cast<R_xlen_t>(p) ||
      std::any_of(means.begin(), means.end(),
                  [](double value) { return !std::isfinite(value); })) {
    Rcpp::stop("Saved PCA means are invalid.");
  }
  std::uint64_t first = 0;
  auto local_cross = pca_cross_state(saved_cross, completed_rows, n,
    chunk_rows, p, workers, progress, checkpoint_every, first);
  int blocks = 0;
  for (std::uint64_t start = first; start < n; start += chunk_rows) {
    const auto count = std::min<std::uint64_t>(chunk_rows, n - start);
    source->read_rows(start, count, block.data());
    parallel_rows(count, workers, [&](std::uint64_t begin,
                                      std::uint64_t end, int worker) {
      double* cross = local_cross.data() +
        static_cast<std::size_t>(worker) * p * p;
      for (auto row = begin; row < end; ++row) {
        const float* values = block.data() + row * p;
        for (std::uint64_t col = 0; col < p; ++col) {
          const double value = values[col] - means[col];
          for (std::uint64_t other = col; other < p; ++other) {
            cross[col * p + other] +=
              value * (values[other] - means[other]);
          }
        }
      }
    });
    ++blocks;
    if (checkpoint_every > 0 &&
        (blocks % checkpoint_every == 0 || start + count == n)) {
      pca_cross_checkpoint(progress, start + count, local_cross,
                           p, workers);
    }
    Rcpp::checkUserInterrupt();
  }
  Rcpp::Rcout << "  PCA covariance pass: 100%\n";
  NumericMatrix covariance(p, p);
  for (int worker = 0; worker < workers; ++worker) {
    const double* cross = local_cross.data() +
      static_cast<std::size_t>(worker) * p * p;
    for (std::uint64_t col = 0; col < p; ++col) {
      for (std::uint64_t other = col; other < p; ++other) {
        covariance(col, other) += cross[col * p + other] /
          static_cast<double>(n - 1);
      }
    }
  }
  for (std::uint64_t col = 0; col < p; ++col) {
    for (std::uint64_t other = col + 1; other < p; ++other) {
      covariance(other, col) = covariance(col, other);
    }
  }
  return List::create(Rcpp::Named("center") = means,
                      Rcpp::Named("covariance") = covariance);
}

// [[Rcpp::export]]
List massive_pca_wide_moments_cpp(List spec,
                                  int chunk_rows,
                                  int n_threads,
                                  bool center,
                                  bool scale) {
  auto source = make_source(spec);
  const auto n = source->nrow();
  const auto p = source->ncol();
  if (n < 2 || p <= 1024 ||
      p > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      chunk_rows < 1 || n_threads < 1) {
    Rcpp::stop("Wide streamed PCA needs n >= 2, p > 1024, "
               "and positive chunk rows and threads.");
  }
  const int workers = std::min(n_threads, chunk_rows);
  std::vector<float> block(checked_product(chunk_rows, p));
  NumericVector means = pca_streamed_means(
    *source, chunk_rows, workers, center, block
  );
  NumericVector deviations(p, 1.0);
  if (scale) {
    std::fill(deviations.begin(), deviations.end(), 0.0);
    std::vector<std::vector<double>> local(
      workers, std::vector<double>(p, 0.0)
    );
    for (std::uint64_t start = 0; start < n; start += chunk_rows) {
      const auto count = std::min<std::uint64_t>(chunk_rows, n - start);
      source->read_rows(start, count, block.data());
      parallel_rows(count, workers, [&](std::uint64_t begin,
                                        std::uint64_t end, int worker) {
        auto& sums = local[worker];
        for (auto row = begin; row < end; ++row) {
          const float* values = block.data() + row * p;
          for (std::uint64_t col = 0; col < p; ++col) {
            const double delta = values[col] - means[col];
            sums[col] += delta * delta;
          }
        }
      });
      Rcpp::checkUserInterrupt();
    }
    for (const auto& sums : local) {
      for (std::uint64_t col = 0; col < p; ++col) {
        deviations[col] += sums[col] / static_cast<double>(n - 1);
      }
    }
    for (std::uint64_t col = 0; col < p; ++col) {
      const double variance = deviations[col];
      if (!std::isfinite(variance) || variance <= 0.0) {
        Rcpp::stop("PCA scaling requires positive column variance.");
      }
      deviations[col] = std::sqrt(variance);
    }
    Rcpp::Rcout << "  PCA scale pass: 100%\n";
  }
  return List::create(Rcpp::Named("center") = means,
                      Rcpp::Named("scale") = deviations);
}

// [[Rcpp::export]]
NumericMatrix massive_pca_wide_action_cpp(List spec,
                                           NumericVector center,
                                           NumericVector scale,
                                           NumericMatrix basis,
                                           int chunk_rows,
                                           int n_threads) {
  auto source = make_source(spec);
  const auto n = source->nrow();
  const auto p = source->ncol();
  const int width = basis.ncol();
  if (n < 2 || p <= 1024 ||
      p > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      center.size() != static_cast<int>(p) ||
      scale.size() != static_cast<int>(p) ||
      basis.nrow() != static_cast<int>(p) ||
      width < 1 || chunk_rows < 1 || n_threads < 1) {
    Rcpp::stop("Invalid wide streamed PCA sketch dimensions.");
  }
  for (std::uint64_t col = 0; col < p; ++col) {
    if (!std::isfinite(center[col]) || !std::isfinite(scale[col]) ||
        scale[col] <= 0.0) {
      Rcpp::stop("Wide streamed PCA needs finite center and scale.");
    }
  }
  for (double value : basis) {
    if (!std::isfinite(value)) {
      Rcpp::stop("Wide streamed PCA basis must be finite.");
    }
  }
  const int workers = std::min(n_threads, chunk_rows);
  const auto size = checked_product(p, width);
  std::vector<std::vector<double>> local(
    workers, std::vector<double>(size, 0.0)
  );
  std::vector<float> block(checked_product(chunk_rows, p));
  for (std::uint64_t start = 0; start < n; start += chunk_rows) {
    const auto count = std::min<std::uint64_t>(chunk_rows, n - start);
    source->read_rows(start, count, block.data());
    parallel_rows(count, workers, [&](std::uint64_t begin,
                                      std::uint64_t end, int worker) {
      auto& cross = local[worker];
      std::vector<double> values(p), projected(width);
      for (auto row = begin; row < end; ++row) {
        const float* input = block.data() + row * p;
        for (std::uint64_t col = 0; col < p; ++col) {
          values[col] = (input[col] - center[col]) / scale[col];
        }
        for (int component = 0; component < width; ++component) {
          const double* weights = basis.begin() +
            static_cast<std::size_t>(component) * p;
          double sum = 0.0;
          for (std::uint64_t col = 0; col < p; ++col) {
            sum += values[col] * weights[col];
          }
          projected[component] = sum;
        }
        for (int component = 0; component < width; ++component) {
          double* output = cross.data() +
            static_cast<std::size_t>(component) * p;
          for (std::uint64_t col = 0; col < p; ++col) {
            output[col] += values[col] * projected[component];
          }
        }
      }
    });
    Rcpp::checkUserInterrupt();
  }
  NumericMatrix action(p, width);
  for (const auto& cross : local) {
    for (std::size_t i = 0; i < size; ++i) action[i] += cross[i];
  }
  Rcpp::Rcout << "  PCA sketch pass: 100%\n";
  return action;
}

// [[Rcpp::export]]
NumericMatrix massive_pca_wide_action_cuda_cpp(List spec,
                                                NumericVector center,
                                                NumericVector scale,
                                                NumericMatrix basis,
                                                int chunk_rows) {
  auto source = make_source(spec);
  const auto n = source->nrow();
  const auto p = source->ncol();
  const int width = basis.ncol();
  if (n < 2 || p <= 1024 ||
      p > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      center.size() != static_cast<int>(p) ||
      scale.size() != static_cast<int>(p) ||
      basis.nrow() != static_cast<int>(p) ||
      width < 1 || chunk_rows < 1) {
    Rcpp::stop("Invalid wide CUDA PCA sketch dimensions.");
  }
  std::vector<float> center_float(p), scale_float(p);
  std::vector<float> components(checked_product(p, width));
  for (std::uint64_t col = 0; col < p; ++col) {
    center_float[col] = static_cast<float>(center[col]);
    scale_float[col] = static_cast<float>(scale[col]);
    if (!std::isfinite(center_float[col]) ||
        !std::isfinite(scale_float[col]) || scale_float[col] <= 0) {
      Rcpp::stop("Wide CUDA PCA needs finite center and scale.");
    }
  }
  for (std::size_t i = 0; i < components.size(); ++i) {
    components[i] = static_cast<float>(basis[i]);
    if (!std::isfinite(components[i])) {
      Rcpp::stop("Wide CUDA PCA basis must be finite.");
    }
  }
  using CudaContext = std::unique_ptr<void, decltype(
    &fastembedr_massive_pca_cuda_destroy)>;
  CudaContext context(fastembedr_massive_pca_cuda_create(
    chunk_rows, static_cast<int>(p), width, center_float.data(),
    scale_float.data(), components.data()
  ), fastembedr_massive_pca_cuda_destroy);
  if (!context) Rcpp::stop(fastembedr_massive_pca_cuda_error());
  std::vector<float> block(checked_product(chunk_rows, p));
  std::vector<float> chunk_cross(checked_product(p, width));
  NumericMatrix action(p, width);
  for (std::uint64_t start = 0; start < n; start += chunk_rows) {
    const int count = static_cast<int>(std::min<std::uint64_t>(
      chunk_rows, n - start
    ));
    source->read_rows(start, count, block.data());
    if (fastembedr_massive_pca_cuda_action(
          context.get(), block.data(), count, chunk_cross.data())) {
      Rcpp::stop(fastembedr_massive_pca_cuda_error());
    }
    for (std::size_t i = 0; i < chunk_cross.size(); ++i) {
      action[i] += chunk_cross[i];
    }
    Rcpp::checkUserInterrupt();
  }
  Rcpp::Rcout << "  CUDA PCA sketch pass: 100%\n";
  return action;
}

// [[Rcpp::export]]
List massive_pca_covariance_cuda_cpp(List spec,
                                     int chunk_rows,
                                     int n_threads,
                                     bool center,
                                     SEXP saved_means = R_NilValue,
                                     double completed_rows = 0,
                                     SEXP saved_cross = R_NilValue,
                                     SEXP progress = R_NilValue,
                                     int checkpoint_every = 0) {
  auto source = make_source(spec);
  check_massive_pca_source(*source, chunk_rows, n_threads, 2048);
  const auto n = source->nrow();
  const int p = static_cast<int>(source->ncol());
  const int workers = std::min(n_threads, chunk_rows);
  std::vector<float> block(checked_product(chunk_rows, p));
  NumericVector means = saved_means == R_NilValue ?
    pca_streamed_means(*source, chunk_rows, workers, center, block) :
    Rcpp::as<NumericVector>(saved_means);
  if (means.size() != p ||
      std::any_of(means.begin(), means.end(),
                  [](double value) { return !std::isfinite(value); })) {
    Rcpp::stop("Saved PCA means are invalid.");
  }
  std::vector<float> center_float(p);
  for (int col = 0; col < p; ++col) {
    center_float[col] = static_cast<float>(means[col]);
  }
  using CudaContext = std::unique_ptr<void, decltype(
    &fastembedr_massive_pca_cuda_destroy)>;
  CudaContext context(fastembedr_massive_pca_cuda_create(
    chunk_rows, p, 0, center_float.data(), nullptr, nullptr
  ), fastembedr_massive_pca_cuda_destroy);
  if (!context) {
    Rcpp::stop(fastembedr_massive_pca_cuda_error());
  }
  std::vector<float> chunk_cross(checked_product(p, p));
  std::uint64_t first = 0;
  auto cross = pca_cross_state(saved_cross, completed_rows, n,
    chunk_rows, p, 1, progress, checkpoint_every, first);
  int blocks = 0;
  for (std::uint64_t start = first; start < n; start += chunk_rows) {
    const int count = static_cast<int>(std::min<std::uint64_t>(
      chunk_rows, n - start
    ));
    source->read_rows(start, count, block.data());
    if (fastembedr_massive_pca_cuda_cross(
          context.get(), block.data(), count, chunk_cross.data())) {
      Rcpp::stop(fastembedr_massive_pca_cuda_error());
    }
    for (std::size_t i = 0; i < cross.size(); ++i) {
      cross[i] += chunk_cross[i];
    }
    ++blocks;
    if (checkpoint_every > 0 &&
        (blocks % checkpoint_every == 0 || start + count == n)) {
      pca_cross_checkpoint(progress, start + count, cross, p, 1);
    }
    Rcpp::checkUserInterrupt();
  }
  Rcpp::Rcout << "  CUDA PCA covariance pass: 100%\n";
  NumericMatrix covariance(p, p);
  for (int col = 0; col < p; ++col) {
    for (int other = col; other < p; ++other) {
      const double value = 0.5 * (
        cross[col * p + other] + cross[other * p + col]
      ) / static_cast<double>(n - 1);
      covariance(col, other) = value;
      covariance(other, col) = value;
    }
  }
  return List::create(Rcpp::Named("center") = means,
                      Rcpp::Named("covariance") = covariance);
}

// [[Rcpp::export]]
List massive_cuda_memory_cpp() {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  int device = -1;
  if (fastembedr_massive_pca_cuda_memory(
        &free_bytes, &total_bytes, &device)) {
    Rcpp::stop(fastembedr_massive_pca_cuda_error());
  }
  return List::create(
    Rcpp::Named("free_bytes") = static_cast<double>(free_bytes),
    Rcpp::Named("total_bytes") = static_cast<double>(total_bytes),
    Rcpp::Named("device") = device
  );
}

// [[Rcpp::export]]
int massive_cuda_device_count_cpp() {
  int count = 0;
  if (fastembedr_massive_cuda_device_count(&count)) {
    Rcpp::stop(fastembedr_massive_pca_cuda_error());
  }
  return count;
}

// [[Rcpp::export]]
int massive_cuda_select_cpp(int device) {
  const int count = massive_cuda_device_count_cpp();
  if (device < 0 || device >= count ||
      fastembedr_massive_cuda_select_device(device)) {
    Rcpp::stop("Requested CUDA device is unavailable; no fallback was used.");
  }
  return device;
}

// [[Rcpp::export]]
void massive_concat_word_files_cpp(Rcpp::CharacterVector paths,
                                    std::string output,
                                    Rcpp::NumericVector byte_counts,
                                    bool resume = false) {
  if (paths.size() != byte_counts.size() || paths.size() == 0) {
    Rcpp::stop("Shards and byte counts must match.");
  }
  std::uint64_t total = 0;
  for (R_xlen_t i = 0; i < paths.size(); ++i) {
    const auto bytes = whole_number(byte_counts[i], "shard byte count");
    if (bytes % 4 != 0) Rcpp::stop("Shard is not four-byte aligned.");
    const std::string path = Rcpp::as<std::string>(paths[i]);
    std::error_code error;
    if (std::filesystem::file_size(path, error) != bytes || error) {
      Rcpp::stop("Shard size mismatch.");
    }
    if (bytes > std::numeric_limits<std::uint64_t>::max() - total) {
      Rcpp::stop("Shard byte count overflows uint64.");
    }
    total += bytes;
  }
  std::vector<float> buffer(1048576);
  std::uint64_t committed = 0;
  if (resume) {
    std::error_code error;
    const auto size = std::filesystem::file_size(output + ".part", error);
    if (error || size > total) {
      Rcpp::stop("Shard merge partial size is invalid.");
    }
    const auto block = buffer.size() * sizeof(float);
    committed = size - size % block;
  }
  Float32Sink sink(output, committed, total, resume);
  auto skip = committed;
  for (R_xlen_t i = 0; i < paths.size(); ++i) {
    std::ifstream input(Rcpp::as<std::string>(paths[i]), std::ios::binary);
    const auto shard_bytes = whole_number(byte_counts[i],
                                          "shard byte count");
    if (skip >= shard_bytes) {
      skip -= shard_bytes;
      continue;
    }
    input.seekg(static_cast<std::streamoff>(skip));
    auto remaining = shard_bytes - skip;
    skip = 0;
    while (remaining > 0) {
      const auto bytes = std::min<std::uint64_t>(
        remaining, buffer.size() * sizeof(float));
      input.read(reinterpret_cast<char*>(buffer.data()), bytes);
      if (!input) Rcpp::stop("Projection shard read failed.");
      sink.append(buffer.data(), bytes / 4);
      remaining -= bytes;
      Rcpp::checkUserInterrupt();
    }
  }
  sink.finish(total);
}

// [[Rcpp::export]]
void massive_pca_project_cpp(List spec,
                              std::string output,
                              NumericMatrix loadings,
                              NumericVector center,
                              NumericVector scale,
                              int chunk_rows,
                              int n_threads,
                              double start_row = 0,
                              SEXP progress = R_NilValue,
                              bool resume = false) {
  auto source = make_source(spec);
  const auto n = source->nrow();
  const auto p = source->ncol();
  const auto rank = loadings.ncol();
  const auto first = whole_number(start_row, "start_row");
  if (loadings.nrow() != static_cast<int>(p) ||
      center.size() != static_cast<int>(p) ||
      scale.size() != static_cast<int>(p) ||
      chunk_rows < 1 || n_threads < 1 || rank < 1 || first > n ||
      (first != n && first % chunk_rows != 0) ||
      (first > 0 && !resume) || (resume && progress == R_NilValue)) {
    Rcpp::stop("Invalid streamed PCA projection dimensions or controls.");
  }
  const auto expected = checked_product(checked_product(n, rank), 4);
  const auto input_capacity = checked_product(chunk_rows, p);
  const auto output_capacity = checked_product(chunk_rows, rank);
  std::vector<float> block(static_cast<std::size_t>(input_capacity));
  std::vector<float> scores(static_cast<std::size_t>(output_capacity));
  Float32Sink sink(output,
    checked_product(checked_product(first, rank), 4), expected, resume);
  for (std::uint64_t start = first; start < n; start += chunk_rows) {
    const auto count = std::min<std::uint64_t>(chunk_rows, n - start);
    source->read_rows(start, count, block.data());
    parallel_rows(count, n_threads, [&](std::uint64_t begin,
                                        std::uint64_t end, int) {
      for (auto row = begin; row < end; ++row) {
        const float* values = block.data() + row * p;
        for (int component = 0; component < rank; ++component) {
          double value = 0;
          for (std::uint64_t col = 0; col < p; ++col) {
            value += ((values[col] - center[col]) / scale[col]) *
              loadings(col, component);
          }
          scores[row * rank + component] = static_cast<float>(value);
        }
      }
    });
    sink.append(scores.data(), checked_product(count, rank));
    checkpoint_output(sink, progress, start + count);
    Rcpp::checkUserInterrupt();
  }
  Rcpp::Rcout << "  PCA projection pass: 100%\n";
  sink.finish(expected);
}

// [[Rcpp::export]]
void massive_pca_project_cuda_cpp(List spec,
                                   std::string output,
                                   NumericMatrix loadings,
                                   NumericVector center,
                                   NumericVector scale,
                                   int chunk_rows,
                                   double start_row = 0,
                                   SEXP progress = R_NilValue,
                                   bool resume = false) {
  auto source = make_source(spec);
  const auto n = source->nrow();
  const int p = static_cast<int>(source->ncol());
  const int rank = loadings.ncol();
  const auto first = whole_number(start_row, "start_row");
  if (loadings.nrow() != p || center.size() != p ||
      scale.size() != p || chunk_rows < 1 || rank < 1 || first > n ||
      (first != n && first % chunk_rows != 0) ||
      (first > 0 && !resume) || (resume && progress == R_NilValue)) {
    Rcpp::stop("Invalid streamed CUDA PCA projection dimensions.");
  }
  const auto expected = checked_product(checked_product(n, rank), 4);
  std::vector<float> center_float(p), scale_float(p);
  std::vector<float> components(checked_product(p, rank));
  for (int col = 0; col < p; ++col) {
    center_float[col] = static_cast<float>(center[col]);
    scale_float[col] = static_cast<float>(scale[col]);
    if (!std::isfinite(scale_float[col]) || scale_float[col] <= 0) {
      Rcpp::stop("Invalid streamed CUDA PCA scale.");
    }
  }
  for (std::size_t i = 0; i < components.size(); ++i) {
    components[i] = static_cast<float>(loadings[i]);
  }
  using CudaContext = std::unique_ptr<void, decltype(
    &fastembedr_massive_pca_cuda_destroy)>;
  CudaContext context(fastembedr_massive_pca_cuda_create(
    chunk_rows, p, rank, center_float.data(), scale_float.data(),
    components.data()
  ), fastembedr_massive_pca_cuda_destroy);
  if (!context) Rcpp::stop(fastembedr_massive_pca_cuda_error());
  std::vector<float> block(checked_product(chunk_rows, p));
  std::vector<float> scores(checked_product(chunk_rows, rank));
  Float32Sink sink(output,
    checked_product(checked_product(first, rank), 4), expected, resume);
  for (std::uint64_t start = first; start < n; start += chunk_rows) {
    const int count = static_cast<int>(std::min<std::uint64_t>(
      chunk_rows, n - start
    ));
    source->read_rows(start, count, block.data());
    if (fastembedr_massive_pca_cuda_project(
          context.get(), block.data(), count, scores.data())) {
      Rcpp::stop(fastembedr_massive_pca_cuda_error());
    }
    sink.append(scores.data(), checked_product(count, rank));
    checkpoint_output(sink, progress, start + count);
    Rcpp::checkUserInterrupt();
  }
  Rcpp::Rcout << "  CUDA PCA projection pass: 100%\n";
  sink.finish(expected);
}

// [[Rcpp::export]]
NumericVector massive_landmark_sample_cpp(List spec,
                                           int landmarks,
                                           int seed,
                                           int chunk_rows,
                                           std::string output,
                                           std::string method) {
  auto source = make_source(spec);
  const auto n = source->nrow();
  const auto p = source->ncol();
  if (landmarks < 1 || static_cast<std::uint64_t>(landmarks) > n ||
      chunk_rows < 1) {
    Rcpp::stop("Invalid reservoir count or chunk size.");
  }
  std::vector<std::uint64_t> selected;
  selected.reserve(landmarks);
  std::mt19937_64 generator(static_cast<std::uint32_t>(seed));
  if (method == "reservoir") {
    selected.resize(landmarks);
    std::iota(selected.begin(), selected.end(), std::uint64_t(0));
    for (std::uint64_t row = landmarks; row < n; ++row) {
      std::uniform_int_distribution<std::uint64_t> draw(0, row);
      const auto slot = draw(generator);
      if (slot < static_cast<std::uint64_t>(landmarks)) {
        selected[slot] = row;
      }
      if ((row & 1048575) == 0) Rcpp::checkUserInterrupt();
    }
  } else if (method == "random") {
    std::unordered_set<std::uint64_t> sampled;
    sampled.reserve(landmarks);
    for (std::uint64_t row = n - landmarks; row < n; ++row) {
      std::uniform_int_distribution<std::uint64_t> draw(0, row);
      const auto candidate = draw(generator);
      if (!sampled.insert(candidate).second) sampled.insert(row);
      if ((sampled.size() & 1048575) == 0) Rcpp::checkUserInterrupt();
    }
    selected.assign(sampled.begin(), sampled.end());
  } else {
    Rcpp::stop("Unknown experimental landmark method.");
  }
  Rcpp::Rcout << "  Landmark selection: 100%\n";
  std::sort(selected.begin(), selected.end());
  std::vector<float> block(static_cast<std::size_t>(
    checked_product(chunk_rows, p)
  ));
  Float32Sink sink(output);
  std::uint64_t next = 0;
  if (method == "reservoir") {
    for (std::uint64_t start = 0; start < n; start += chunk_rows) {
      const auto count = std::min<std::uint64_t>(chunk_rows, n - start);
      source->read_rows(start, count, block.data());
      while (next < static_cast<std::uint64_t>(landmarks) &&
             selected[next] < start + count) {
        const auto offset = checked_product(selected[next] - start, p);
        sink.append(block.data() + offset, p);
        ++next;
      }
      Rcpp::checkUserInterrupt();
    }
  } else {
    while (next < static_cast<std::uint64_t>(landmarks)) {
      const auto count = std::min<std::uint64_t>(chunk_rows,
        landmarks - next);
      for (std::uint64_t row = 0; row < count; ++row) {
        source->read_rows(selected[next + row], 1,
          block.data() + checked_product(row, p));
      }
      sink.append(block.data(), checked_product(count, p));
      next += count;
      Rcpp::checkUserInterrupt();
    }
  }
  if (next != static_cast<std::uint64_t>(landmarks)) {
    Rcpp::stop("Reservoir extraction missed selected rows.");
  }
  sink.finish(checked_product(checked_product(landmarks, p), 4));
  Rcpp::Rcout << "  Landmark extraction: 100%\n";
  NumericVector indices(landmarks);
  for (int i = 0; i < landmarks; ++i) {
    indices[i] = static_cast<double>(selected[i] + 1);
  }
  return indices;
}

namespace {

std::vector<float> posting_centers(NumericMatrix centers_r) {
  const auto lists = static_cast<std::size_t>(centers_r.nrow());
  const auto columns = static_cast<std::size_t>(centers_r.ncol());
  std::vector<float> centers(checked_product(lists, columns));
  for (std::size_t list = 0; list < lists; ++list) {
    for (std::size_t col = 0; col < columns; ++col) {
      const double value = centers_r(list, col);
      if (!std::isfinite(value) ||
          std::abs(value) > std::numeric_limits<float>::max()) {
        Rcpp::stop("Posting centers must be finite float32 values.");
      }
      centers[list * columns + col] = static_cast<float>(value);
    }
  }
  return centers;
}

std::uint32_t coarse_posting(const float* row,
    const std::vector<float>& centers, std::size_t lists,
    std::size_t columns, double* nearest = nullptr) {
  double best = std::numeric_limits<double>::infinity();
  std::uint32_t selected = 0;
  for (std::size_t list = 0; list < lists; ++list) {
    double distance = 0;
    for (std::size_t col = 0; col < columns; ++col) {
      const double delta = static_cast<double>(row[col]) -
        centers[list * columns + col];
      distance += delta * delta;
    }
    if (distance < best) {
      best = distance;
      selected = static_cast<std::uint32_t>(list);
    }
  }
  if (nearest != nullptr) *nearest = best;
  return selected;
}

void assign_postings(const std::vector<float>& block,
    std::vector<std::uint32_t>& assignments,
    const std::vector<float>& centers, std::size_t lists,
    std::size_t columns, std::uint64_t count, int workers) {
  parallel_rows(count, workers, [&](std::uint64_t begin,
      std::uint64_t end, int) {
    for (auto row = begin; row < end; ++row) {
      assignments[row] = coarse_posting(
        block.data() + row * columns, centers, lists, columns);
    }
  });
}

void write_posting(std::fstream& stream, std::uint64_t offset,
    const char* data, std::uint64_t bytes) {
  if (offset > static_cast<std::uint64_t>(
      std::numeric_limits<std::streamoff>::max()) ||
      bytes > static_cast<std::uint64_t>(
      std::numeric_limits<std::streamsize>::max())) {
    Rcpp::stop("Posting offset exceeds stream limits.");
  }
  stream.seekp(static_cast<std::streamoff>(offset));
  stream.write(data, static_cast<std::streamsize>(bytes));
  if (!stream) Rcpp::stop("Posting output write failed.");
}

void verify_completed_posting_file(const std::string& path,
    std::uint64_t bytes) {
  const bool final = std::filesystem::exists(path);
  const bool part = std::filesystem::exists(path + ".part");
  if (final == part) Rcpp::stop(
    "Completed posting output has missing or duplicate files.");
  if (std::filesystem::file_size(final ? path : path + ".part") != bytes) {
    Rcpp::stop("Completed posting output has an invalid byte count.");
  }
}

void finish_completed_postings(const std::string& features,
    const std::string& ids, const std::string& offsets_path,
    const std::vector<std::uint64_t>& offsets,
    std::uint64_t feature_bytes, std::uint64_t id_bytes) {
  verify_completed_posting_file(features, feature_bytes);
  verify_completed_posting_file(ids, id_bytes);
  const auto part = offsets_path + ".part";
  if (!std::filesystem::exists(offsets_path) &&
      !std::filesystem::exists(part)) {
    std::ofstream output(part, std::ios::binary);
    output.write(reinterpret_cast<const char*>(offsets.data()),
      static_cast<std::streamsize>(offsets.size() * 8));
    output.close();
    if (!output) Rcpp::stop("Cannot write posting offsets.");
  } else {
    verify_completed_posting_file(offsets_path, offsets.size() * 8);
    const auto& existing = std::filesystem::exists(offsets_path) ?
      offsets_path : part;
    std::vector<std::uint64_t> saved(offsets.size());
    std::ifstream input(existing, std::ios::binary);
    input.read(reinterpret_cast<char*>(saved.data()),
      static_cast<std::streamsize>(saved.size() * 8));
    if (!input || saved != offsets) Rcpp::stop(
      "Completed posting offsets disagree with checkpoint.");
  }
  for (const auto& path : {features, ids, offsets_path}) {
    if (!std::filesystem::exists(path)) {
      std::filesystem::rename(path + ".part", path);
    }
  }
}

}  // namespace

// [[Rcpp::export]]
Rcpp::IntegerVector massive_posting_distant_rows_cpp(List spec,
    NumericMatrix centers_r, int sample, int chunk_rows, int workers,
    int candidate_rows = 0) {
  require_little_endian();
  auto source = make_source(spec);
  const auto rows = source->nrow();
  const auto columns = source->ncol();
  const auto lists = static_cast<std::size_t>(centers_r.nrow());
  if (rows > static_cast<std::uint64_t>(
        std::numeric_limits<int>::max()) ||
      columns != static_cast<std::uint64_t>(centers_r.ncol()) ||
      lists < 2 || lists > 65536 || sample < 1 ||
      static_cast<std::uint64_t>(sample) > rows ||
      chunk_rows < 1 || workers < 1 || candidate_rows < 0 ||
      (candidate_rows > 0 &&
       (candidate_rows < sample ||
        static_cast<std::uint64_t>(candidate_rows) > rows))) {
    Rcpp::stop("Posting audit dimensions or controls are invalid.");
  }
  const auto centers = posting_centers(centers_r);
  using RankedRow = std::pair<double, int>;
  std::priority_queue<RankedRow, std::vector<RankedRow>,
                      std::greater<RankedRow>> farthest;
  auto consider = [&](std::uint64_t row, double distance) {
    if (!std::isfinite(distance)) Rcpp::stop(
      "Posting audit input has non-finite center distance.");
    const RankedRow candidate{distance, static_cast<int>(row + 1)};
    if (static_cast<int>(farthest.size()) < sample) {
      farthest.push(candidate);
    } else if (candidate > farthest.top()) {
      farthest.pop();
      farthest.push(candidate);
    }
  };
  if (candidate_rows > 0) {
    std::vector<float> row(columns);
    for (int i = 0; i < candidate_rows; ++i) {
      const auto midpoint = 2 * static_cast<std::uint64_t>(i) + 1;
      const auto index = checked_product(midpoint, rows) /
        (2 * static_cast<std::uint64_t>(candidate_rows));
      source->read_rows(index, 1, row.data());
      double distance = 0;
      coarse_posting(row.data(), centers, lists, columns, &distance);
      consider(index, distance);
      if ((i & 255) == 0) Rcpp::checkUserInterrupt();
    }
  } else {
    std::vector<float> block(checked_product(chunk_rows, columns));
    std::vector<double> distances(chunk_rows);
    for (std::uint64_t first = 0; first < rows;) {
      const auto count = std::min<std::uint64_t>(chunk_rows, rows - first);
      source->read_rows(first, count, block.data());
      parallel_rows(count, workers, [&](std::uint64_t begin,
          std::uint64_t end, int) {
        for (auto row = begin; row < end; ++row) {
          coarse_posting(block.data() + row * columns, centers,
                         lists, columns, &distances[row]);
        }
      });
      for (std::uint64_t row = 0; row < count; ++row) {
        consider(first + row, distances[row]);
      }
      first += count;
      Rcpp::checkUserInterrupt();
    }
  }
  std::vector<RankedRow> ranked;
  ranked.reserve(sample);
  while (!farthest.empty()) {
    ranked.push_back(farthest.top());
    farthest.pop();
  }
  std::sort(ranked.rbegin(), ranked.rend());
  Rcpp::IntegerVector result(sample);
  for (int i = 0; i < sample; ++i) result[i] = ranked[i].second;
  return result;
}

// [[Rcpp::export]]
List massive_coarse_postings_cpp(List spec, NumericMatrix centers_r,
    std::string prefix, int chunk_rows, int workers,
    SEXP resume_state, SEXP checkpoint_callback,
    int checkpoint_every) {
  require_little_endian();
  auto source = make_source(spec);
  const auto rows = source->nrow();
  const auto columns = source->ncol();
  const auto lists = static_cast<std::size_t>(centers_r.nrow());
  if (rows > static_cast<std::uint64_t>(
      std::numeric_limits<int>::max()) ||
      columns != static_cast<std::uint64_t>(centers_r.ncol()) ||
      lists < 2 || lists > 65536 || chunk_rows < 1 || workers < 1 ||
      checkpoint_every < 0 ||
      (checkpoint_every == 0 && checkpoint_callback != R_NilValue) ||
      (checkpoint_every > 0 && checkpoint_callback == R_NilValue) ||
      (resume_state != R_NilValue && checkpoint_every == 0)) {
    Rcpp::stop("Posting dimensions or controls are invalid.");
  }
  const auto centers = posting_centers(centers_r);
  const std::string features = prefix + ".features.f32";
  const std::string ids = prefix + ".ids.u32";
  const std::string offsets_path = prefix + ".offsets.u64";
  const bool resume = resume_state != R_NilValue;
  const auto block_size = std::min<std::uint64_t>(rows, chunk_rows);
  std::vector<float> block(checked_product(block_size, columns));
  std::vector<float> grouped(block.size());
  std::vector<std::uint32_t> assignments(block_size);
  std::vector<std::uint32_t> grouped_ids(block_size);
  std::vector<std::uint64_t> counts(lists, 0);
  std::uint64_t completed = 0;
  int next_percent = 10;
  if (resume) {
    List state(resume_state);
    NumericVector saved_counts = state["counts"];
    if (saved_counts.size() != static_cast<int>(lists)) {
      Rcpp::stop("Posting checkpoint count length is invalid.");
    }
    completed = whole_number(Rcpp::as<double>(state["completed_rows"]),
                             "Posting completed rows");
    for (std::size_t list = 0; list < lists; ++list) {
      counts[list] = whole_number(saved_counts[list], "Posting count");
    }
  }
  for (const auto& path : {features, ids, offsets_path}) {
    if ((std::filesystem::exists(path) &&
         (!resume || completed != rows)) ||
        (!resume && std::filesystem::exists(path + ".part"))) {
      Rcpp::stop("Posting output or partial file already exists.");
    }
  }
  if (!resume) {
    for (std::uint64_t first = 0; first < rows; first += block_size) {
      const auto count = std::min(block_size, rows - first);
      source->read_rows(first, count, block.data());
      assign_postings(block, assignments, centers, lists, columns,
                      count, workers);
      for (std::uint64_t row = 0; row < count; ++row) {
        ++counts[assignments[row]];
      }
      report_graph_progress("Posting count pass", first + count,
                            rows, next_percent);
      Rcpp::checkUserInterrupt();
    }
  }
  std::vector<std::uint64_t> offsets(lists + 1, 0);
  for (std::size_t list = 0; list < lists; ++list) {
    offsets[list + 1] = offsets[list] + counts[list];
  }
  if (offsets.back() != rows) Rcpp::stop(
    "Posting counts do not cover the source.");
  const auto feature_bytes =
    checked_product(checked_product(rows, columns), 4);
  const auto id_bytes = checked_product(rows, 4);
  if (resume && completed == rows) {
    List state(resume_state);
    NumericVector saved_next = state["positions"];
    if (saved_next.size() != static_cast<int>(lists)) Rcpp::stop(
      "Posting checkpoint position is invalid.");
    for (std::size_t list = 0; list < lists; ++list) {
      if (whole_number(saved_next[list], "Posting position") !=
          offsets[list + 1]) Rcpp::stop(
        "Posting checkpoint positions disagree with completed rows.");
    }
    finish_completed_postings(features, ids, offsets_path, offsets,
      feature_bytes, id_bytes);
    return List::create(
      Rcpp::Named("counts") = Rcpp::wrap(counts),
      Rcpp::Named("offsets") = Rcpp::wrap(offsets));
  }
  if (resume) {
    if (!std::filesystem::exists(features + ".part") ||
        !std::filesystem::exists(ids + ".part") ||
        std::filesystem::file_size(features + ".part") != feature_bytes ||
        std::filesystem::file_size(ids + ".part") != id_bytes) {
      Rcpp::stop("Posting partial files disagree with checkpoint.");
    }
  } else {
    std::ofstream(features + ".part", std::ios::binary).close();
    std::ofstream(ids + ".part", std::ios::binary).close();
    std::filesystem::resize_file(features + ".part", feature_bytes);
    std::filesystem::resize_file(ids + ".part", id_bytes);
  }
  std::fstream feature_stream(features + ".part",
    std::ios::in | std::ios::out | std::ios::binary);
  std::fstream id_stream(ids + ".part",
    std::ios::in | std::ios::out | std::ios::binary);
  if (!feature_stream || !id_stream) Rcpp::stop(
    "Cannot open posting outputs.");
  std::vector<std::uint64_t> next(offsets.begin(), offsets.end() - 1);
  if (resume) {
    List state(resume_state);
    NumericVector saved_next = state["positions"];
    if (saved_next.size() != static_cast<int>(lists) ||
        completed > rows ||
        (completed != rows && completed % block_size != 0)) {
      Rcpp::stop("Posting checkpoint position is invalid.");
    }
    std::uint64_t placed = 0;
    for (std::size_t list = 0; list < lists; ++list) {
      next[list] = whole_number(saved_next[list], "Posting position");
      if (next[list] < offsets[list] ||
          next[list] > offsets[list + 1]) {
        Rcpp::stop("Posting checkpoint position is invalid.");
      }
      placed += next[list] - offsets[list];
    }
    if (placed != completed) Rcpp::stop(
      "Posting checkpoint positions disagree with completed rows.");
  }
  if (!resume && checkpoint_every > 0) {
    Rcpp::Function callback(checkpoint_callback);
    callback(0, Rcpp::wrap(counts), Rcpp::wrap(next));
  }
  std::vector<std::uint64_t> local(lists), cursor(lists);
  next_percent = 10;
  std::uint64_t blocks = completed / block_size;
  for (std::uint64_t first = completed; first < rows;
       first += block_size, ++blocks) {
    const auto count = std::min(block_size, rows - first);
    source->read_rows(first, count, block.data());
    assign_postings(block, assignments, centers, lists, columns,
                    count, workers);
    std::fill(local.begin(), local.end(), 0);
    for (std::uint64_t row = 0; row < count; ++row) {
      ++local[assignments[row]];
    }
    std::uint64_t placed = 0;
    for (std::size_t list = 0; list < lists; ++list) {
      cursor[list] = placed;
      placed += local[list];
    }
    for (std::uint64_t row = 0; row < count; ++row) {
      const auto position = cursor[assignments[row]]++;
      std::copy_n(block.data() + row * columns, columns,
                  grouped.data() + position * columns);
      grouped_ids[position] = static_cast<std::uint32_t>(first + row + 1);
    }
    placed = 0;
    for (std::size_t list = 0; list < lists; ++list) {
      if (local[list] != 0) {
        write_posting(feature_stream,
          checked_product(checked_product(next[list], columns), 4),
          reinterpret_cast<const char*>(
            grouped.data() + placed * columns),
          checked_product(checked_product(local[list], columns), 4));
        write_posting(id_stream, checked_product(next[list], 4),
          reinterpret_cast<const char*>(grouped_ids.data() + placed),
          checked_product(local[list], 4));
        next[list] += local[list];
      }
      placed += local[list];
    }
    report_graph_progress("Posting write pass", first + count,
                          rows, next_percent);
    if (checkpoint_every > 0 &&
        ((blocks + 1) % checkpoint_every == 0 ||
         first + count == rows)) {
      feature_stream.flush();
      id_stream.flush();
      if (!feature_stream || !id_stream) Rcpp::stop(
        "Cannot flush posting checkpoint outputs.");
      Rcpp::Function callback(checkpoint_callback);
      callback(static_cast<double>(first + count),
               Rcpp::wrap(counts), Rcpp::wrap(next));
    }
    Rcpp::checkUserInterrupt();
  }
  if (!std::equal(next.begin(), next.end(), offsets.begin() + 1)) {
    Rcpp::stop("Posting write counts do not match offsets.");
  }
  feature_stream.close();
  id_stream.close();
  std::ofstream offset_stream(offsets_path + ".part",
    std::ios::binary);
  offset_stream.write(reinterpret_cast<const char*>(offsets.data()),
    static_cast<std::streamsize>(offsets.size() * sizeof(offsets[0])));
  offset_stream.close();
  if (!offset_stream) Rcpp::stop("Cannot write posting offsets.");
  std::filesystem::rename(features + ".part", features);
  std::filesystem::rename(ids + ".part", ids);
  std::filesystem::rename(offsets_path + ".part", offsets_path);
  return List::create(
    Rcpp::Named("counts") = Rcpp::wrap(counts),
    Rcpp::Named("offsets") = Rcpp::wrap(offsets)
  );
}

namespace {

std::vector<std::uint64_t> posting_offsets(const std::string& path,
    std::size_t lists, std::uint64_t rows) {
  if (!std::filesystem::exists(path) ||
      std::filesystem::file_size(path) !=
        checked_product(lists + 1, 8)) {
    Rcpp::stop("Posting offsets file has the wrong size.");
  }
  std::vector<std::uint64_t> offsets(lists + 1);
  std::ifstream input(path, std::ios::binary);
  input.read(reinterpret_cast<char*>(offsets.data()),
             static_cast<std::streamsize>((lists + 1) * 8));
  if (!input || offsets.front() != 0 || offsets.back() != rows) {
    Rcpp::stop("Posting offsets do not cover the source.");
  }
  for (std::size_t list = 0; list < lists; ++list) {
    if (offsets[list] > offsets[list + 1]) Rcpp::stop(
      "Posting offsets are not ordered.");
  }
  return offsets;
}

using PostingNeighbor = std::pair<float, std::uint32_t>;

bool posting_closer(const PostingNeighbor& left,
                    const PostingNeighbor& right) {
  return left.first < right.first ||
    (left.first == right.first && left.second < right.second);
}

void posting_route(const std::vector<float>& queries,
    const std::vector<float>& centers, std::size_t lists,
    std::size_t columns, int count, int probes, int workers,
    std::vector<std::vector<int>>& active) {
  std::vector<std::uint32_t> routes(checked_product(count, probes));
  std::atomic<bool> invalid(false);
  parallel_rows(count, workers, [&](std::uint64_t begin,
      std::uint64_t end, int) {
    std::vector<std::pair<double, std::uint32_t>> ranked(lists);
    for (auto qi = begin; qi < end; ++qi) {
      const float* query = queries.data() + qi * columns;
      bool bad = false;
      for (std::size_t list = 0; list < lists; ++list) {
        double distance = 0;
        for (std::size_t col = 0; col < columns; ++col) {
          const double delta = static_cast<double>(query[col]) -
            centers[list * columns + col];
          distance += delta * delta;
        }
        if (!std::isfinite(distance)) {
          invalid.store(true, std::memory_order_relaxed);
          bad = true;
        }
        ranked[list] = {distance, static_cast<std::uint32_t>(list)};
      }
      if (bad) continue;
      if (probes < static_cast<int>(lists)) {
        std::nth_element(ranked.begin(), ranked.begin() + probes,
                         ranked.end());
      }
      std::sort(ranked.begin(), ranked.begin() + probes);
      for (int probe = 0; probe < probes; ++probe) {
        routes[qi * probes + probe] = ranked[probe].second;
      }
    }
  });
  if (invalid.load(std::memory_order_relaxed)) Rcpp::stop(
    "Posting center distance is non-finite.");
  for (int qi = 0; qi < count; ++qi) {
    for (int probe = 0; probe < probes; ++probe) {
      active[routes[static_cast<std::size_t>(qi) * probes + probe]]
        .push_back(qi);
    }
  }
  Rcpp::checkUserInterrupt();
}

void posting_scan(std::ifstream& features, std::ifstream& ids,
    std::uint64_t first, std::uint64_t count,
    std::size_t columns, int chunk,
    std::vector<float>& block,
    std::vector<std::uint32_t>& row_ids,
    const std::vector<int>& active,
    const std::vector<float>& queries,
    std::vector<std::vector<PostingNeighbor>>& heaps,
    std::vector<std::uint64_t>& candidates,
    const std::vector<std::uint32_t>& query_ids,
    int k, int workers,
    bool exclude_self, std::uint64_t rows) {
  std::atomic<bool> invalid(false);
  for (std::uint64_t offset = first; offset < first + count;) {
    const auto size = std::min<std::uint64_t>(chunk,
                                             first + count - offset);
    const auto feature_bytes =
      checked_product(checked_product(size, columns), 4);
    const auto id_bytes = checked_product(size, 4);
    const auto feature_offset =
      checked_product(checked_product(offset, columns), 4);
    const auto id_offset = checked_product(offset, 4);
    if (feature_offset > static_cast<std::uint64_t>(
        std::numeric_limits<std::streamoff>::max()) ||
        id_offset > static_cast<std::uint64_t>(
        std::numeric_limits<std::streamoff>::max())) {
      Rcpp::stop("Posting read offset exceeds stream limits.");
    }
    features.seekg(static_cast<std::streamoff>(feature_offset));
    ids.seekg(static_cast<std::streamoff>(id_offset));
    features.read(reinterpret_cast<char*>(block.data()),
                  static_cast<std::streamsize>(feature_bytes));
    ids.read(reinterpret_cast<char*>(row_ids.data()),
             static_cast<std::streamsize>(id_bytes));
    if (!features || !ids) Rcpp::stop("Posting read failed.");
    for (std::uint64_t ri = 0; ri < size; ++ri) {
      if (row_ids[ri] < 1 || row_ids[ri] > rows) Rcpp::stop(
        "Posting row ID is invalid.");
    }
    parallel_rows(active.size(), workers,
                  [&](std::uint64_t begin, std::uint64_t end, int) {
      for (auto ai = begin; ai < end; ++ai) {
        const auto qi = active[ai];
        auto& heap = heaps[qi];
        const float* query = queries.data() +
          static_cast<std::size_t>(qi) * columns;
        for (std::uint64_t ri = 0; ri < size; ++ri) {
          const auto id = row_ids[ri];
          if (exclude_self && id == query_ids[qi]) continue;
          const float distance = fastembedr::squared_l2_distance(
            query, block.data() + ri * columns,
            static_cast<int>(columns));
          if (!std::isfinite(distance)) {
            invalid.store(true, std::memory_order_relaxed);
            continue;
          }
          ++candidates[qi];
          const PostingNeighbor candidate{distance, id};
          if (static_cast<int>(heap.size()) < k) {
            heap.push_back(candidate);
            std::push_heap(heap.begin(), heap.end(), posting_closer);
          } else if (posting_closer(candidate, heap.front())) {
            std::pop_heap(heap.begin(), heap.end(), posting_closer);
            heap.back() = candidate;
            std::push_heap(heap.begin(), heap.end(), posting_closer);
          }
        }
      }
    });
    if (invalid.load(std::memory_order_relaxed)) Rcpp::stop(
      "Posting distance overflowed float32.");
    offset += size;
    Rcpp::checkUserInterrupt();
  }
}

}  // namespace

// [[Rcpp::export]]
List massive_posting_search_cpp(List query_spec,
    std::string feature_path, std::string ids_path,
    std::string offsets_path, NumericMatrix centers,
    double first_row, int count, int k, int nprobe,
    int reference_chunk, int workers, bool exclude_self,
    double posting_rows, bool verbose) {
  require_little_endian();
  auto query = make_source(query_spec);
  const auto rows = whole_number(posting_rows, "posting_rows");
  const auto first = whole_number(first_row, "first_row");
  const auto columns = query->ncol();
  const auto lists = static_cast<std::size_t>(centers.nrow());
  if (rows < 1 || rows > static_cast<std::uint64_t>(
        std::numeric_limits<std::int32_t>::max()) ||
      columns != static_cast<std::uint64_t>(centers.ncol()) ||
      lists < 2 || lists > 65536 || first < 1 ||
      first > query->nrow() || count < 1 ||
      static_cast<std::uint64_t>(count) > query->nrow() - first + 1 ||
      k < 1 || k > 65536 || nprobe < 1 ||
      nprobe > static_cast<int>(lists) || reference_chunk < 1 ||
      workers < 1 ||
      checked_product(checked_product(count, columns), 4) > 256000000 ||
      checked_product(checked_product(reference_chunk, columns), 4) >
        256000000 || checked_product(count, k) > 128000000 / 12 ||
      checked_product(count, nprobe) > 128000000 / 4) {
    Rcpp::stop("Posting search dimensions or buffers are invalid.");
  }
  const auto center_values = posting_centers(centers);
  const auto offsets = posting_offsets(offsets_path, lists, rows);
  if (!std::filesystem::exists(feature_path) ||
      !std::filesystem::exists(ids_path) ||
      std::filesystem::file_size(feature_path) !=
        checked_product(checked_product(rows, columns), 4) ||
      std::filesystem::file_size(ids_path) != checked_product(rows, 4)) {
    Rcpp::stop("Posting feature or ID file has the wrong size.");
  }
  const bool selected = query_spec.containsElementNamed(
    "selected_rows");
  if (selected && first != 1) Rcpp::stop(
    "Selected query rows require first_row = 1.");
  if (selected && query_spec.containsElementNamed("query_ids_path")) {
    Rcpp::stop("Selected rows cannot use grouped query IDs.");
  }
  std::vector<float> queries(checked_product(count, columns));
  std::vector<std::uint32_t> query_ids(count);
  if (selected) {
    Rcpp::NumericVector selected_rows = query_spec["selected_rows"];
    if (selected_rows.size() != count) Rcpp::stop(
      "Selected query row count does not match the batch.");
    for (int qi = 0; qi < count; ++qi) {
      const auto id = whole_number(selected_rows[qi], "selected_rows");
      if (id < 1 || id > query->nrow() || id > rows) Rcpp::stop(
        "Selected query row is outside the source.");
      query_ids[qi] = static_cast<std::uint32_t>(id);
      query->read_rows(id - 1, 1,
        queries.data() + static_cast<std::size_t>(qi) * columns);
    }
  } else if (query_spec.containsElementNamed("query_ids_path")) {
    query->read_rows(first - 1, count, queries.data());
    const std::string path = Rcpp::as<std::string>(
      query_spec["query_ids_path"]);
    if (!std::filesystem::exists(path) ||
        std::filesystem::file_size(path) != checked_product(rows, 4)) {
      Rcpp::stop("Posting query ID file has the wrong size.");
    }
    std::ifstream query_ids_file(path, std::ios::binary);
    query_ids_file.seekg(static_cast<std::streamoff>(
      checked_product(first - 1, 4)));
    query_ids_file.read(reinterpret_cast<char*>(query_ids.data()),
      static_cast<std::streamsize>(checked_product(count, 4)));
    if (!query_ids_file) Rcpp::stop("Posting query ID read failed.");
    for (const auto id : query_ids) {
      if (id < 1 || id > rows) Rcpp::stop(
        "Posting query ID is invalid.");
    }
  } else {
    query->read_rows(first - 1, count, queries.data());
    for (int qi = 0; qi < count; ++qi) {
      query_ids[qi] = static_cast<std::uint32_t>(first + qi);
    }
  }
  std::vector<std::vector<int>> active(lists);
  posting_route(queries, center_values, lists, columns, count,
                nprobe, workers, active);
  std::vector<std::vector<PostingNeighbor>> heaps(count);
  for (auto& heap : heaps) heap.reserve(k);
  std::vector<std::uint64_t> candidates(count, 0);
  std::vector<float> block(
    checked_product(reference_chunk, columns));
  std::vector<std::uint32_t> row_ids(reference_chunk);
  std::ifstream features(feature_path, std::ios::binary);
  std::ifstream ids(ids_path, std::ios::binary);
  if (!features || !ids) Rcpp::stop("Cannot open posting files.");
  int next_percent = 10;
  std::uint64_t posting_rows_read = 0;
  for (std::size_t list = 0; list < lists; ++list) {
    if (!active[list].empty()) {
      posting_rows_read += offsets[list + 1] - offsets[list];
      posting_scan(features, ids, offsets[list],
        offsets[list + 1] - offsets[list], columns, reference_chunk,
        block, row_ids, active[list], queries, heaps, candidates,
        query_ids, k, workers,
        exclude_self, rows);
    }
    if (verbose) report_graph_progress("Posting search",
      list + 1, lists, next_percent);
  }
  Rcpp::IntegerMatrix indices(count, k);
  Rcpp::NumericMatrix distances(count, k);
  Rcpp::NumericVector candidate_count(count);
  for (int qi = 0; qi < count; ++qi) {
    auto& heap = heaps[qi];
    if (static_cast<int>(heap.size()) != k) Rcpp::stop(
      "Probed postings contain fewer than k neighbors.");
    std::sort(heap.begin(), heap.end(), posting_closer);
    candidate_count[qi] = static_cast<double>(candidates[qi]);
    for (int rank = 0; rank < k; ++rank) {
      indices(qi, rank) = static_cast<int>(heap[rank].second);
      distances(qi, rank) = std::sqrt(std::max(0.0f,
                                               heap[rank].first));
    }
  }
  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("candidate_count") = candidate_count,
    Rcpp::Named("posting_read_bytes") = static_cast<double>(
      checked_product(posting_rows_read,
        checked_product(columns, 4) + 4)),
    Rcpp::Named("backend_used") = "native_cpu_posting_scan",
    Rcpp::Named("exact") = nprobe == static_cast<int>(lists)
  );
}
