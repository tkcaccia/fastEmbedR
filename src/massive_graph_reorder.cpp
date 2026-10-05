/*
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT
 */

#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
constexpr std::uint64_t fanout = 32;
constexpr std::size_t flush_bytes = 65536;

struct WorkGuard {
  fs::path path;
  ~WorkGuard() {
    std::error_code error;
    fs::remove_all(path, error);
  }
};

std::uint64_t checked_file_size(const fs::path& path) {
  std::error_code error;
  const auto bytes = fs::file_size(path, error);
  if (error) Rcpp::stop("Grouped graph file is missing or unreadable.");
  return bytes;
}

std::uint32_t read_id(const char* bytes) {
  std::uint32_t value;
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}

void append_bytes(std::vector<char>& target, const void* source,
                  std::size_t bytes) {
  const auto* first = static_cast<const char*>(source);
  target.insert(target.end(), first, first + bytes);
}

void flush_bins(std::vector<std::ofstream>& files,
                std::vector<std::vector<char>>& buffers) {
  for (std::size_t i = 0; i < files.size(); ++i) {
    if (!buffers[i].empty()) {
      files[i].write(buffers[i].data(), buffers[i].size());
      buffers[i].clear();
    }
    if (!files[i]) Rcpp::stop("Grouped graph partition write failed.");
  }
}

void open_bins(const fs::path& stem, std::uint64_t count,
               std::vector<std::ofstream>& files,
               std::vector<std::vector<char>>& buffers) {
  files.resize(count);
  buffers.resize(count);
  for (std::uint64_t i = 0; i < count; ++i) {
    const auto path = stem.string() + "." + std::to_string(i);
    files[i].open(path, std::ios::binary | std::ios::trunc);
    if (!files[i]) Rcpp::stop("Cannot create graph partition.");
    buffers[i].reserve(flush_bytes);
  }
}

void append_record(std::ofstream& file, std::vector<char>& buffer,
                   const char* record, std::size_t bytes) {
  append_bytes(buffer, record, bytes);
  if (buffer.size() >= flush_bytes) {
    file.write(buffer.data(), buffer.size());
    buffer.clear();
    if (!file) Rcpp::stop("Grouped graph partition write failed.");
  }
}

void write_sorted(const fs::path& path, std::uint64_t first,
                  std::uint64_t rows, std::uint64_t vertices,
                  int k, std::size_t record_bytes,
                  std::ofstream& ids, std::ofstream& distances) {
  if (checked_file_size(path) != rows * record_bytes) {
    Rcpp::stop("Grouped graph has missing or duplicated rows.");
  }
  std::vector<char> records(rows * record_bytes);
  std::ifstream source(path, std::ios::binary);
  source.read(records.data(), records.size());
  if (!source) Rcpp::stop("Grouped graph partition read failed.");
  std::vector<std::size_t> order(rows);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](auto left, auto right) {
    return read_id(records.data() + left * record_bytes) <
      read_id(records.data() + right * record_bytes);
  });
  std::vector<char> id_buffer;
  std::vector<char> distance_buffer;
  for (std::uint64_t offset = 0; offset < rows; ++offset) {
    if (offset % 65536 == 0) Rcpp::checkUserInterrupt();
    const char* record = records.data() + order[offset] * record_bytes;
    const auto row = read_id(record);
    if (row != first + offset + 1) Rcpp::stop(
      "Grouped graph has missing or duplicated row IDs.");
    for (int rank = 0; rank < k; ++rank) {
      const auto id = read_id(record + 4 + rank * 4);
      float distance;
      std::memcpy(&distance, record + 4 + 4 * k + rank * 4, 4);
      if (id < 1 || id > vertices || id == row ||
          !std::isfinite(distance) || distance < 0) {
        Rcpp::stop("Grouped graph has invalid neighbors.");
      }
    }
    append_bytes(id_buffer, record + 4, 4 * k);
    append_bytes(distance_buffer, record + 4 + 4 * k, 4 * k);
    if (id_buffer.size() >= flush_bytes) {
      ids.write(id_buffer.data(), id_buffer.size());
      distances.write(distance_buffer.data(), distance_buffer.size());
      id_buffer.clear();
      distance_buffer.clear();
      if (!ids || !distances) Rcpp::stop(
        "Reordered graph output write failed.");
    }
  }
  if (!id_buffer.empty()) {
    ids.write(id_buffer.data(), id_buffer.size());
    distances.write(distance_buffer.data(), distance_buffer.size());
  }
  if (!ids || !distances) Rcpp::stop(
    "Reordered graph output write failed.");
}

void sort_partition(const fs::path& path, std::uint64_t first,
                    std::uint64_t rows, std::uint64_t vertices,
                    int k, std::uint64_t bucket_rows,
                    std::size_t record_bytes, int read_rows,
                    std::ofstream& ids, std::ofstream& distances) {
  if (rows <= bucket_rows) {
    write_sorted(path, first, rows, vertices, k, record_bytes,
                 ids, distances);
    fs::remove(path);
    return;
  }
  if (checked_file_size(path) != rows * record_bytes) Rcpp::stop(
    "Grouped graph partition has missing or duplicated rows.");
  const auto children = std::min(fanout,
    (rows + bucket_rows - 1) / bucket_rows);
  const auto width = (rows + children - 1) / children;
  std::vector<std::ofstream> files;
  std::vector<std::vector<char>> buffers;
  open_bins(path, children, files, buffers);
  std::ifstream input(path, std::ios::binary);
  std::vector<char> block(read_rows * record_bytes);
  for (std::uint64_t done = 0; done < rows;) {
    Rcpp::checkUserInterrupt();
    const auto count = std::min<std::uint64_t>(read_rows, rows - done);
    input.read(block.data(), count * record_bytes);
    if (!input) Rcpp::stop("Grouped graph partition read failed.");
    for (std::uint64_t i = 0; i < count; ++i) {
      const char* record = block.data() + i * record_bytes;
      const auto id = read_id(record);
      if (id <= first || id > first + rows) Rcpp::stop(
        "Grouped graph row ID is outside its partition.");
      const auto child = (id - first - 1) / width;
      append_record(files[child], buffers[child], record,
                    record_bytes);
    }
    done += count;
  }
  flush_bins(files, buffers);
  files.clear();
  input.close();
  fs::remove(path);
  for (std::uint64_t i = 0; i < children; ++i) {
    const auto offset = i * width;
    const auto count = std::min(width, rows - offset);
    const auto child = path.string() + "." + std::to_string(i);
    sort_partition(child, first + offset, count, vertices, k,
                   bucket_rows, record_bytes, read_rows,
                   ids, distances);
  }
}

void distribute_grouped(const fs::path& ids_path,
                        const fs::path& grouped_ids,
                        const fs::path& grouped_distances,
                        const fs::path& stem, std::uint64_t rows,
                        int k, std::uint64_t bucket_rows,
                        int read_rows) {
  const auto children = std::min(fanout,
    (rows + bucket_rows - 1) / bucket_rows);
  const auto width = (rows + children - 1) / children;
  std::vector<std::ofstream> files;
  std::vector<std::vector<char>> buffers;
  open_bins(stem, children, files, buffers);
  std::ifstream row_ids(ids_path, std::ios::binary);
  std::ifstream neighbors(grouped_ids, std::ios::binary);
  std::ifstream distances(grouped_distances, std::ios::binary);
  if (!row_ids || !neighbors || !distances) Rcpp::stop(
    "Cannot open grouped graph inputs.");
  std::vector<std::uint32_t> ids(read_rows);
  std::vector<char> knn(static_cast<std::size_t>(read_rows) * k * 4);
  std::vector<char> values(knn.size());
  for (std::uint64_t done = 0; done < rows;) {
    Rcpp::checkUserInterrupt();
    const auto count = std::min<std::uint64_t>(read_rows, rows - done);
    row_ids.read(reinterpret_cast<char*>(ids.data()), count * 4);
    neighbors.read(knn.data(), count * k * 4);
    distances.read(values.data(), count * k * 4);
    if (!row_ids || !neighbors || !distances) Rcpp::stop(
      "Grouped graph input read failed.");
    for (std::uint64_t i = 0; i < count; ++i) {
      const auto row = ids[i];
      if (row < 1 || row > rows) Rcpp::stop(
        "Grouped graph has an invalid row ID.");
      const auto child = (row - 1) / width;
      auto& buffer = buffers[child];
      append_bytes(buffer, &row, 4);
      append_bytes(buffer, knn.data() + i * k * 4, k * 4);
      append_bytes(buffer, values.data() + i * k * 4, k * 4);
      if (buffer.size() >= flush_bytes) {
        files[child].write(buffer.data(), buffer.size());
        buffer.clear();
        if (!files[child]) Rcpp::stop(
          "Grouped graph partition write failed.");
      }
    }
    done += count;
  }
  flush_bins(files, buffers);
}

}  // namespace

// [[Rcpp::export]]
void massive_reorder_posting_graph_cpp(
    std::string posting_ids_path, std::string grouped_indices_path,
    std::string grouped_distances_path, std::string output_indices_path,
    std::string output_distances_path, double n_vertices, int k,
    int bucket_rows, int read_rows, bool restart_work) {
  if (!std::isfinite(n_vertices) || n_vertices < 2 ||
      n_vertices > std::numeric_limits<std::int32_t>::max() ||
      n_vertices != std::floor(n_vertices) || k < 1 ||
      k >= n_vertices || k > 65536 || bucket_rows < 1 ||
      read_rows < 1) Rcpp::stop("Invalid grouped graph dimensions.");
  const auto rows = static_cast<std::uint64_t>(n_vertices);
  const auto record_bytes = static_cast<std::size_t>(4 + 8 * k);
  if (checked_file_size(posting_ids_path) != rows * 4 ||
      checked_file_size(grouped_indices_path) != rows * k * 4 ||
      checked_file_size(grouped_distances_path) != rows * k * 4) Rcpp::stop(
        "Grouped graph input files have incorrect byte counts.");
  const fs::path work = output_indices_path + ".reorder-work";
  const auto marker = work / "fastEmbedR-reorder-v1";
  if (fs::exists(work)) {
    if (!restart_work || !fs::is_regular_file(marker)) Rcpp::stop(
      "Grouped graph reorder work exists; inspect it before retry.");
    fs::remove_all(work);
  }
  if (!fs::create_directory(work)) Rcpp::stop(
    "Cannot create grouped graph reorder work directory.");
  WorkGuard cleanup{work};
  std::ofstream mark(marker, std::ios::binary);
  mark << "fastEmbedR-reorder-v1\n";
  mark.close();
  const auto stem = work / "partition";
  distribute_grouped(posting_ids_path, grouped_indices_path,
                     grouped_distances_path, stem, rows, k,
                     bucket_rows, read_rows);
  std::ofstream ids(output_indices_path,
                    std::ios::binary | std::ios::trunc);
  std::ofstream distances(output_distances_path,
                          std::ios::binary | std::ios::trunc);
  if (!ids || !distances) Rcpp::stop(
    "Cannot create reordered graph outputs.");
  const auto children = std::min(fanout,
    (rows + static_cast<std::uint64_t>(bucket_rows) - 1) /
    bucket_rows);
  const auto width = (rows + children - 1) / children;
  for (std::uint64_t i = 0; i < children; ++i) {
    const auto offset = i * width;
    const auto count = std::min(width, rows - offset);
    const auto path = stem.string() + "." + std::to_string(i);
    sort_partition(path, offset, count, rows, k, bucket_rows,
                   record_bytes, read_rows, ids, distances);
  }
  ids.flush();
  distances.flush();
  if (!ids || !distances) Rcpp::stop(
    "Reordered graph output flush failed.");
}
