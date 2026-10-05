/*
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT
 */

#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using Neighbor = std::pair<float, std::uint32_t>;

void check_file_size(const std::string& path, std::uint64_t expected,
                     bool allow_missing) {
  std::error_code error;
  if (allow_missing && !std::filesystem::exists(path, error)) {
    if (expected == 0) return;
  }
  const auto size = std::filesystem::file_size(path, error);
  if (error || size != expected) {
    Rcpp::stop("Sharded KNN partial file has an unexpected size.");
  }
}

std::vector<Neighbor> merge_row(const std::vector<std::uint32_t>& old_ids,
                                const std::vector<float>& old_distances,
                                const Rcpp::IntegerMatrix& indices,
                                const Rcpp::NumericMatrix& distances,
                                int row, std::uint64_t query,
                                std::uint64_t shard_first,
                                std::uint64_t shard_rows,
                                int n_vertices, int k,
                                bool exclude_self) {
  std::vector<Neighbor> candidates;
  candidates.reserve(static_cast<std::size_t>(k) + indices.ncol());
  for (int rank = 0; rank < k; ++rank) {
    const auto id = old_ids[rank];
    if (id == 0) continue;
    if (id > static_cast<std::uint32_t>(n_vertices) ||
        (exclude_self && id == query) ||
        !std::isfinite(old_distances[rank]) ||
        old_distances[rank] < 0.0f) {
      Rcpp::stop("Sharded KNN partial row is invalid.");
    }
    candidates.emplace_back(old_distances[rank], id);
  }
  for (int rank = 0; rank < indices.ncol(); ++rank) {
    const int local = indices(row, rank);
    const double distance = distances(row, rank);
    if (local < 1 || static_cast<std::uint64_t>(local) > shard_rows ||
        !std::isfinite(distance) || distance < 0.0) {
      Rcpp::stop("Native HNSW returned an invalid neighbor.");
    }
    const auto id = shard_first + static_cast<std::uint64_t>(local) - 1;
    if (id > static_cast<std::uint64_t>(n_vertices)) {
      Rcpp::stop("Native HNSW neighbor exceeds the graph.");
    }
    if (!exclude_self || id != query) {
      candidates.emplace_back(static_cast<float>(distance),
                              static_cast<std::uint32_t>(id));
    }
  }
  std::sort(candidates.begin(), candidates.end());
  std::vector<Neighbor> selected;
  selected.reserve(k);
  std::unordered_set<std::uint32_t> seen;
  seen.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    if (seen.insert(candidate.second).second) {
      selected.push_back(candidate);
    }
    if (static_cast<int>(selected.size()) == k) break;
  }
  return selected;
}

}  // namespace

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
  if (n_queries < 0) n_queries = n_vertices;
  if (!std::isfinite(query_first) || !std::isfinite(shard_first) ||
      !std::isfinite(shard_rows) || !std::isfinite(n_vertices) ||
      !std::isfinite(n_queries) ||
      query_first < 1 || shard_first < 1 || shard_rows < 2 ||
      n_vertices < 2 || n_vertices > std::numeric_limits<int>::max() ||
      n_queries < 1 || n_queries > std::numeric_limits<int>::max() ||
      query_first != std::floor(query_first) ||
      shard_first != std::floor(shard_first) ||
      shard_rows != std::floor(shard_rows) ||
      n_vertices != std::floor(n_vertices) ||
      n_queries != std::floor(n_queries) || k < 1 ||
      k > n_vertices || (exclude_self && k >= n_vertices) ||
      indices.nrow() != distances.nrow() ||
      indices.ncol() != distances.ncol() || indices.nrow() < 1 ||
      indices.ncol() < 1 ||
      shard_first + shard_rows - 1 > n_vertices ||
      query_first + indices.nrow() - 1 > n_queries) {
    Rcpp::stop("Invalid sharded KNN merge dimensions.");
  }
  const auto start = static_cast<std::uint64_t>(query_first) - 1;
  const auto total_bytes = static_cast<std::uint64_t>(n_queries) * k * 4;
  const auto prior_bytes = start * k * 4;
  const auto expected = first_shard ? prior_bytes : total_bytes;
  check_file_size(indices_path, expected, first_shard);
  check_file_size(distances_path, expected, first_shard);
  const auto flags = std::ios::binary | std::ios::out |
    (first_shard ? std::ios::app : std::ios::in);
  std::fstream ids(indices_path, flags);
  std::fstream values(distances_path, flags);
  if (!ids || !values) Rcpp::stop("Cannot open sharded KNN partial files.");
  std::vector<std::uint32_t> old_ids(k, 0);
  std::vector<float> old_distances(
    k, std::numeric_limits<float>::infinity());
  for (int row = 0; row < indices.nrow(); ++row) {
    const auto offset = static_cast<std::streamoff>(
      (start + static_cast<std::uint64_t>(row)) * k * 4);
    if (first_shard) {
      std::fill(old_ids.begin(), old_ids.end(), 0);
      std::fill(old_distances.begin(), old_distances.end(),
                std::numeric_limits<float>::infinity());
    } else {
      ids.seekg(offset);
      values.seekg(offset);
      ids.read(reinterpret_cast<char*>(old_ids.data()), k * 4);
      values.read(reinterpret_cast<char*>(old_distances.data()), k * 4);
      if (!ids || !values) Rcpp::stop("Sharded KNN partial read failed.");
    }
    const auto selected = merge_row(old_ids, old_distances,
      indices, distances, row, start + row + 1,
      static_cast<std::uint64_t>(shard_first),
      static_cast<std::uint64_t>(shard_rows),
      static_cast<int>(n_vertices), k, exclude_self);
    std::fill(old_ids.begin(), old_ids.end(), 0);
    std::fill(old_distances.begin(), old_distances.end(),
              std::numeric_limits<float>::infinity());
    for (std::size_t rank = 0; rank < selected.size(); ++rank) {
      old_ids[rank] = selected[rank].second;
      old_distances[rank] = selected[rank].first;
    }
    ids.seekp(offset);
    values.seekp(offset);
    ids.write(reinterpret_cast<const char*>(old_ids.data()), k * 4);
    values.write(reinterpret_cast<const char*>(old_distances.data()), k * 4);
    if (!ids || !values) Rcpp::stop("Sharded KNN partial write failed.");
  }
  ids.flush();
  values.flush();
  if (!ids || !values) Rcpp::stop("Sharded KNN partial flush failed.");
}
