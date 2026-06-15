#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using Rcpp::IntegerMatrix;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::LogicalMatrix;
using Rcpp::NumericMatrix;

extern "C" {
void fastembedr_knn_euclidean_range(const double* data,
                                    const double* points,
                                    int n_data,
                                    int n_points,
                                    int n_features,
                                    int k,
                                    int exclude_self,
                                    int query_start,
                                    int query_end,
                                    int* indices,
                                    double* distances);
}

namespace {

struct Neighbor {
  double distance;
  int index;
};

enum class DistanceKind {
  Euclidean,
  Manhattan,
  Minkowski,
  Cosine,
  Correlation
};

bool neighbor_less(const Neighbor& a, const Neighbor& b) {
  if (a.distance == b.distance) return a.index < b.index;
  return a.distance < b.distance;
}

bool fortran_nn_enabled() {
  const char* value = std::getenv("FASTEMBEDR_USE_FORTRAN_NN");
  return value != nullptr && std::string(value) == "1";
}

bool env_is_truthy(const char* value) {
  if (value == nullptr) return false;
  const std::string text(value);
  return text == "1" || text == "true" || text == "TRUE" ||
    text == "yes" || text == "YES" || text == "on" || text == "ON";
}

bool env_is_falsey(const char* value) {
  if (value == nullptr) return false;
  const std::string text(value);
  return text == "0" || text == "false" || text == "FALSE" ||
    text == "no" || text == "NO" || text == "off" || text == "OFF";
}

double env_positive_double(const char* name, const double fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr) return fallback;
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value || !std::isfinite(parsed) || parsed <= 0.0) return fallback;
  return parsed;
}

bool use_row_major_distance_layout(const double copy_bytes) {
  const char* forced = std::getenv("FASTEMBEDR_NN_ROW_MAJOR");
  if (env_is_truthy(forced)) return true;
  if (env_is_falsey(forced)) return false;

  const double max_mb = env_positive_double(
    "FASTEMBEDR_NN_ROW_MAJOR_MAX_MB",
    2048.0
  );
  return copy_bytes <= max_mb * 1024.0 * 1024.0;
}

void copy_row_major(const double* src, std::vector<double>& dest, const int nrow, const int ncol) {
  dest.resize(static_cast<std::size_t>(nrow) * static_cast<std::size_t>(ncol));
  for (int c = 0; c < ncol; ++c) {
    const double* src_col = src + static_cast<std::size_t>(c) * nrow;
    for (int r = 0; r < nrow; ++r) {
      dest[static_cast<std::size_t>(r) * ncol + c] = src_col[r];
    }
  }
}

DistanceKind distance_kind_from_method(const std::string& method) {
  if (method == "euclidean") return DistanceKind::Euclidean;
  if (method == "manhattan") return DistanceKind::Manhattan;
  if (method == "minkowski") return DistanceKind::Minkowski;
  if (method == "cosine") return DistanceKind::Cosine;
  if (method == "correlation") return DistanceKind::Correlation;
  Rcpp::stop("unsupported method");
}

template <DistanceKind kind, bool row_major>
double distance_value_layout(const double* data,
                             const double* points,
                             const int data_row,
                             const int point_row,
                             const int n_data,
                             const int n_points,
                             const int n_features,
                             const double p) {
  double acc = 0.0;
  if constexpr (kind == DistanceKind::Euclidean) {
    if constexpr (row_major) {
      const double* x = data + static_cast<std::size_t>(data_row) * n_features;
      const double* y = points + static_cast<std::size_t>(point_row) * n_features;
      for (int c = 0; c < n_features; ++c) {
        const double diff = x[c] - y[c];
        acc += diff * diff;
      }
    } else {
      for (int c = 0; c < n_features; ++c) {
        const double diff = data[static_cast<std::size_t>(c) * n_data + data_row] -
                            points[static_cast<std::size_t>(c) * n_points + point_row];
        acc += diff * diff;
      }
    }
    return acc;
  }
  if constexpr (kind == DistanceKind::Manhattan) {
    if constexpr (row_major) {
      const double* x = data + static_cast<std::size_t>(data_row) * n_features;
      const double* y = points + static_cast<std::size_t>(point_row) * n_features;
      for (int c = 0; c < n_features; ++c) {
        acc += std::abs(x[c] - y[c]);
      }
    } else {
      for (int c = 0; c < n_features; ++c) {
        const double diff = data[static_cast<std::size_t>(c) * n_data + data_row] -
                            points[static_cast<std::size_t>(c) * n_points + point_row];
        acc += std::abs(diff);
      }
    }
    return acc;
  }
  if constexpr (kind == DistanceKind::Cosine) {
    double dot = 0.0;
    double x_norm = 0.0;
    double y_norm = 0.0;
    if constexpr (row_major) {
      const double* x = data + static_cast<std::size_t>(data_row) * n_features;
      const double* y = points + static_cast<std::size_t>(point_row) * n_features;
      for (int c = 0; c < n_features; ++c) {
        dot += x[c] * y[c];
        x_norm += x[c] * x[c];
        y_norm += y[c] * y[c];
      }
    } else {
      for (int c = 0; c < n_features; ++c) {
        const double x = data[static_cast<std::size_t>(c) * n_data + data_row];
        const double y = points[static_cast<std::size_t>(c) * n_points + point_row];
        dot += x * y;
        x_norm += x * x;
        y_norm += y * y;
      }
    }
    if (x_norm <= 0.0 && y_norm <= 0.0) return 0.0;
    if (x_norm <= 0.0 || y_norm <= 0.0) return 1.0;
    const double denom = std::sqrt(x_norm) * std::sqrt(y_norm);
    double cosine = dot / denom;
    if (cosine > 1.0) cosine = 1.0;
    if (cosine < -1.0) cosine = -1.0;
    return 1.0 - cosine;
  }
  if constexpr (kind == DistanceKind::Correlation) {
    double x_mean = 0.0;
    double y_mean = 0.0;
    if constexpr (row_major) {
      const double* x = data + static_cast<std::size_t>(data_row) * n_features;
      const double* y = points + static_cast<std::size_t>(point_row) * n_features;
      for (int c = 0; c < n_features; ++c) {
        x_mean += x[c];
        y_mean += y[c];
      }
      x_mean /= static_cast<double>(n_features);
      y_mean /= static_cast<double>(n_features);
      double dot = 0.0;
      double x_norm = 0.0;
      double y_norm = 0.0;
      for (int c = 0; c < n_features; ++c) {
        const double xc = x[c] - x_mean;
        const double yc = y[c] - y_mean;
        dot += xc * yc;
        x_norm += xc * xc;
        y_norm += yc * yc;
      }
      if (x_norm <= 0.0 && y_norm <= 0.0) return 0.0;
      if (x_norm <= 0.0 || y_norm <= 0.0) return 1.0;
      double corr = dot / (std::sqrt(x_norm) * std::sqrt(y_norm));
      if (corr > 1.0) corr = 1.0;
      if (corr < -1.0) corr = -1.0;
      return 1.0 - corr;
    } else {
      for (int c = 0; c < n_features; ++c) {
        x_mean += data[static_cast<std::size_t>(c) * n_data + data_row];
        y_mean += points[static_cast<std::size_t>(c) * n_points + point_row];
      }
      x_mean /= static_cast<double>(n_features);
      y_mean /= static_cast<double>(n_features);
      double dot = 0.0;
      double x_norm = 0.0;
      double y_norm = 0.0;
      for (int c = 0; c < n_features; ++c) {
        const double xc = data[static_cast<std::size_t>(c) * n_data + data_row] - x_mean;
        const double yc = points[static_cast<std::size_t>(c) * n_points + point_row] - y_mean;
        dot += xc * yc;
        x_norm += xc * xc;
        y_norm += yc * yc;
      }
      if (x_norm <= 0.0 && y_norm <= 0.0) return 0.0;
      if (x_norm <= 0.0 || y_norm <= 0.0) return 1.0;
      double corr = dot / (std::sqrt(x_norm) * std::sqrt(y_norm));
      if (corr > 1.0) corr = 1.0;
      if (corr < -1.0) corr = -1.0;
      return 1.0 - corr;
    }
  }

  if constexpr (row_major) {
    const double* x = data + static_cast<std::size_t>(data_row) * n_features;
    const double* y = points + static_cast<std::size_t>(point_row) * n_features;
    for (int c = 0; c < n_features; ++c) {
      acc += std::pow(std::abs(x[c] - y[c]), p);
    }
  } else {
    for (int c = 0; c < n_features; ++c) {
      const double diff = data[static_cast<std::size_t>(c) * n_data + data_row] -
                          points[static_cast<std::size_t>(c) * n_points + point_row];
      acc += std::pow(std::abs(diff), p);
    }
  }
  return acc;
}

int requested_threads(const bool parallel, const int cores, const int n_points) {
  if (!parallel || n_points < 2) return 1;
  int n_threads = cores;
  if (n_threads <= 0) {
    n_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (n_threads <= 0) n_threads = 1;
  }
  return std::max(1, std::min(n_threads, n_points));
}

template <DistanceKind kind, bool row_major>
void write_knn_rows(const double* data_ptr,
                    const double* points_ptr,
                    const int n_data,
                    const int n_points,
                    const int n_features,
                    const int k,
                    const bool square,
                    const bool sorted,
                    const double p,
                    const bool use_fixed_topk,
                    const bool exclude_self,
                    int* indices_ptr,
                    double* distances_ptr,
                    const int query_start,
                    const int query_end) {
  std::vector<Neighbor> candidates;
  std::vector<Neighbor> top;
  candidates.reserve(use_fixed_topk ? 0L : n_data);
  top.reserve(k);
  for (int q = query_start; q < query_end; ++q) {
    if (use_fixed_topk) {
      top.clear();
      for (int i = 0; i < n_data; ++i) {
        if (exclude_self && i == q) continue;
        const Neighbor candidate{
          distance_value_layout<kind, row_major>(
            data_ptr, points_ptr, i, q, n_data, n_points, n_features, p
          ),
          i
        };
        if (static_cast<int>(top.size()) < k) {
          top.push_back(candidate);
          if (static_cast<int>(top.size()) == k) {
            std::make_heap(top.begin(), top.end(), neighbor_less);
          }
        } else if (neighbor_less(candidate, top.front())) {
          std::pop_heap(top.begin(), top.end(), neighbor_less);
          top.back() = candidate;
          std::push_heap(top.begin(), top.end(), neighbor_less);
        }
      }
      std::sort_heap(top.begin(), top.end(), neighbor_less);
    } else {
      candidates.clear();
      for (int i = 0; i < n_data; ++i) {
        if (exclude_self && i == q) continue;
        candidates.push_back({
          distance_value_layout<kind, row_major>(
            data_ptr, points_ptr, i, q, n_data, n_points, n_features, p
          ),
          i
        });
      }
      std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(), neighbor_less);
      top.assign(candidates.begin(), candidates.begin() + k);
      if (sorted) std::sort(top.begin(), top.end(), neighbor_less);
    }

    for (int j = 0; j < k; ++j) {
      double dist = top[static_cast<std::size_t>(j)].distance;
      if constexpr (kind == DistanceKind::Euclidean) {
        if (!square) dist = std::sqrt(std::max(dist, 0.0));
      } else if constexpr (kind == DistanceKind::Minkowski) {
        dist = std::pow(std::max(dist, 0.0), 1.0 / p);
      }
      indices_ptr[static_cast<std::size_t>(j) * n_points + q] = top[static_cast<std::size_t>(j)].index + 1;
      distances_ptr[static_cast<std::size_t>(j) * n_points + q] = dist;
    }
  }
}

double euclidean_row_major(const double* data,
                           const int a,
                           const int b,
                           const int n_features) {
  const double* x = data + static_cast<std::size_t>(a) * n_features;
  const double* y = data + static_cast<std::size_t>(b) * n_features;
  double acc = 0.0;
  for (int c = 0; c < n_features; ++c) {
    const double diff = x[c] - y[c];
    acc += diff * diff;
  }
  return std::sqrt(std::max(acc, 0.0));
}

void add_candidate_topk(const double* data,
                        const int query,
                        const int candidate,
                        const int n_features,
                        const int k,
                        std::vector<Neighbor>& top) {
  const Neighbor item{euclidean_row_major(data, query, candidate, n_features), candidate};
  if (static_cast<int>(top.size()) < k) {
    top.push_back(item);
    if (static_cast<int>(top.size()) == k) {
      std::make_heap(top.begin(), top.end(), neighbor_less);
    }
  } else if (neighbor_less(item, top.front())) {
    std::pop_heap(top.begin(), top.end(), neighbor_less);
    top.back() = item;
    std::push_heap(top.begin(), top.end(), neighbor_less);
  }
}

struct NeighborF {
  float distance;
  int index;
};

struct ProjectionScore {
  float score;
  int index;
};

bool neighborf_less(const NeighborF& a, const NeighborF& b) {
  if (a.distance == b.distance) return a.index < b.index;
  return a.distance < b.distance;
}

bool projection_score_less(const ProjectionScore& a, const ProjectionScore& b) {
  if (a.score == b.score) return a.index < b.index;
  return a.score < b.score;
}

float squared_euclidean_row_major_float(const float* data,
                                        const int a,
                                        const int b,
                                        const int n_features) {
  const float* x = data + static_cast<std::size_t>(a) * n_features;
  const float* y = data + static_cast<std::size_t>(b) * n_features;
  float acc = 0.0f;
  for (int c = 0; c < n_features; ++c) {
    const float diff = x[c] - y[c];
    acc += diff * diff;
  }
  return acc;
}

float squared_euclidean_cross_row_major_float(const float* data,
                                              const float* points,
                                              const int data_row,
                                              const int point_row,
                                              const int n_features) {
  const float* x = data + static_cast<std::size_t>(data_row) * n_features;
  const float* y = points + static_cast<std::size_t>(point_row) * n_features;
  float acc = 0.0f;
  for (int c = 0; c < n_features; ++c) {
    const float diff = x[c] - y[c];
    acc += diff * diff;
  }
  return acc;
}

void copy_row_major_float(const double* src,
                          std::vector<float>& dest,
                          const int nrow,
                          const int ncol) {
  dest.resize(static_cast<std::size_t>(nrow) * static_cast<std::size_t>(ncol));
  for (int c = 0; c < ncol; ++c) {
    const double* src_col = src + static_cast<std::size_t>(c) * nrow;
    for (int r = 0; r < nrow; ++r) {
      dest[static_cast<std::size_t>(r) * ncol + c] =
        static_cast<float>(src_col[r]);
    }
  }
}

bool sorted_top_contains(const std::vector<NeighborF>& top, const int candidate) {
  for (const NeighborF& item : top) {
    if (item.index == candidate) return true;
  }
  return false;
}

bool insert_sorted_top_float(const float* data,
                             const int query,
                             const int candidate,
                             const int n_features,
                             const int capacity,
                             std::vector<NeighborF>& top) {
  if (candidate == query || candidate < 0) return false;
  if (sorted_top_contains(top, candidate)) return false;
  const NeighborF item{
    squared_euclidean_row_major_float(data, query, candidate, n_features),
    candidate
  };
  if (static_cast<int>(top.size()) == capacity &&
      !neighborf_less(item, top.back())) {
    return false;
  }

  if (static_cast<int>(top.size()) < capacity) {
    top.push_back(item);
  } else {
    top.back() = item;
  }

  int pos = static_cast<int>(top.size()) - 1;
  while (pos > 0 && neighborf_less(top[static_cast<std::size_t>(pos)],
                                   top[static_cast<std::size_t>(pos - 1)])) {
    std::swap(top[static_cast<std::size_t>(pos)],
              top[static_cast<std::size_t>(pos - 1)]);
    --pos;
  }
  return true;
}

void add_projection_candidate_topk(const float* landmarks,
                                   const float* queries,
                                   const int query,
                                   const int candidate,
                                   const int n_features,
                                   const int k,
                                   std::vector<NeighborF>& top) {
  const NeighborF item{
    squared_euclidean_cross_row_major_float(
      landmarks, queries, candidate, query, n_features
    ),
    candidate
  };
  if (static_cast<int>(top.size()) < k) {
    top.push_back(item);
    if (static_cast<int>(top.size()) == k) {
      std::make_heap(top.begin(), top.end(), neighborf_less);
    }
  } else if (neighborf_less(item, top.front())) {
    std::pop_heap(top.begin(), top.end(), neighborf_less);
    top.back() = item;
    std::push_heap(top.begin(), top.end(), neighborf_less);
  }
}

void write_sorted_top_to_graph(const std::vector<NeighborF>& top,
                               std::vector<int>& graph_indices,
                               std::vector<float>& graph_distances,
                               const int row,
                               const int pool_size) {
  const std::size_t base = static_cast<std::size_t>(row) * pool_size;
  const int n_top = static_cast<int>(top.size());
  for (int j = 0; j < pool_size; ++j) {
    if (j < n_top) {
      graph_indices[base + j] = top[static_cast<std::size_t>(j)].index;
      graph_distances[base + j] = top[static_cast<std::size_t>(j)].distance;
    } else {
      graph_indices[base + j] = -1;
      graph_distances[base + j] = std::numeric_limits<float>::infinity();
    }
  }
}

bool insert_heap_top_double(const int candidate,
                            const double distance,
                            const int capacity,
                            std::vector<Neighbor>& top) {
  if (candidate < 0 || !std::isfinite(distance)) return false;
  const Neighbor item{distance, candidate};
  if (static_cast<int>(top.size()) < capacity) {
    top.push_back(item);
    if (static_cast<int>(top.size()) == capacity) {
      std::make_heap(top.begin(), top.end(), neighbor_less);
    }
    return true;
  }
  if (neighbor_less(item, top.front())) {
    std::pop_heap(top.begin(), top.end(), neighbor_less);
    top.back() = item;
    std::push_heap(top.begin(), top.end(), neighbor_less);
    return true;
  }
  return false;
}

void write_heap_top_double(std::vector<Neighbor>& top,
                           int* indices_ptr,
                           double* distances_ptr,
                           const int row,
                           const int n_rows,
                           const int k) {
  if (static_cast<int>(top.size()) == k) {
    std::sort_heap(top.begin(), top.end(), neighbor_less);
  } else {
    std::sort(top.begin(), top.end(), neighbor_less);
  }
  for (int j = 0; j < k; ++j) {
    indices_ptr[static_cast<std::size_t>(j) * n_rows + row] =
      top[static_cast<std::size_t>(j)].index + 1;
    distances_ptr[static_cast<std::size_t>(j) * n_rows + row] =
      top[static_cast<std::size_t>(j)].distance;
  }
}

void add_probe_center(const float* data,
                      const std::vector<int>& centers,
                      const int query,
                      const int center_slot,
                      const int n_features,
                      const int nprobe,
                      std::vector<NeighborF>& probes) {
  const int center_row = centers[static_cast<std::size_t>(center_slot)];
  const NeighborF item{
    squared_euclidean_row_major_float(data, query, center_row, n_features),
    center_slot
  };
  if (static_cast<int>(probes.size()) < nprobe) {
    probes.push_back(item);
    if (static_cast<int>(probes.size()) == nprobe) {
      std::make_heap(probes.begin(), probes.end(), neighborf_less);
    }
  } else if (neighborf_less(item, probes.front())) {
    std::pop_heap(probes.begin(), probes.end(), neighborf_less);
    probes.back() = item;
    std::push_heap(probes.begin(), probes.end(), neighborf_less);
  }
}

struct VpNode {
  int index = -1;
  float threshold = 0.0f;
  int left = -1;
  int right = -1;
};

bool insert_heap_top_double(const int candidate,
                            const double distance,
                            const int capacity,
                            std::vector<Neighbor>& top);

struct Grid2DIndex {
  int bins_x = 1;
  int bins_y = 1;
  double min_x = 0.0;
  double min_y = 0.0;
  double cell_w = 1.0;
  double cell_h = 1.0;
  std::vector<int> offsets;
  std::vector<int> rows;
};

struct Grid3DIndex {
  int bins_x = 1;
  int bins_y = 1;
  int bins_z = 1;
  double min_x = 0.0;
  double min_y = 0.0;
  double min_z = 0.0;
  double cell_w = 1.0;
  double cell_h = 1.0;
  double cell_d = 1.0;
  std::vector<int> offsets;
  std::vector<int> rows;
};

double grid2d_lower_bound_outside_square(const double x,
                                         const double y,
                                         const Grid2DIndex& grid,
                                         const int x0,
                                         const int x1,
                                         const int y0,
                                         const int y1) {
  double best = std::numeric_limits<double>::infinity();
  if (x0 > 0) {
    const double border = grid.min_x + static_cast<double>(x0) * grid.cell_w;
    const double dx = std::max(0.0, x - border);
    best = std::min(best, dx * dx);
  }
  if (x1 + 1 < grid.bins_x) {
    const double border = grid.min_x + static_cast<double>(x1 + 1) * grid.cell_w;
    const double dx = std::max(0.0, border - x);
    best = std::min(best, dx * dx);
  }
  if (y0 > 0) {
    const double border = grid.min_y + static_cast<double>(y0) * grid.cell_h;
    const double dy = std::max(0.0, y - border);
    best = std::min(best, dy * dy);
  }
  if (y1 + 1 < grid.bins_y) {
    const double border = grid.min_y + static_cast<double>(y1 + 1) * grid.cell_h;
    const double dy = std::max(0.0, border - y);
    best = std::min(best, dy * dy);
  }
  return best;
}

inline int grid2d_cell_id(const int ix, const int iy, const int bins_x) {
  return iy * bins_x + ix;
}

inline int grid3d_cell_id(const int ix,
                          const int iy,
                          const int iz,
                          const int bins_x,
                          const int bins_y) {
  return (iz * bins_y + iy) * bins_x + ix;
}

inline int grid2d_coord(const double value,
                        const double min_value,
                        const double cell_size,
                        const int bins) {
  int out = static_cast<int>((value - min_value) / cell_size);
  if (out < 0) out = 0;
  if (out >= bins) out = bins - 1;
  return out;
}

Grid2DIndex build_grid2d_index(const std::vector<double>& x,
                               const std::vector<double>& y,
                               const int bins_per_dim) {
  const int n = static_cast<int>(x.size());
  Grid2DIndex grid;
  grid.bins_x = std::max(1, bins_per_dim);
  grid.bins_y = std::max(1, bins_per_dim);
  auto mmx = std::minmax_element(x.begin(), x.end());
  auto mmy = std::minmax_element(y.begin(), y.end());
  grid.min_x = *mmx.first;
  grid.min_y = *mmy.first;
  const double max_x = *mmx.second;
  const double max_y = *mmy.second;
  const double span_x = std::max(max_x - grid.min_x, std::numeric_limits<double>::epsilon());
  const double span_y = std::max(max_y - grid.min_y, std::numeric_limits<double>::epsilon());
  grid.cell_w = std::nextafter(span_x, std::numeric_limits<double>::infinity()) /
    static_cast<double>(grid.bins_x);
  grid.cell_h = std::nextafter(span_y, std::numeric_limits<double>::infinity()) /
    static_cast<double>(grid.bins_y);

  const int n_cells = grid.bins_x * grid.bins_y;
  grid.offsets.assign(static_cast<std::size_t>(n_cells + 1), 0);
  std::vector<int> cell_ids(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const int ix = grid2d_coord(x[static_cast<std::size_t>(i)], grid.min_x, grid.cell_w, grid.bins_x);
    const int iy = grid2d_coord(y[static_cast<std::size_t>(i)], grid.min_y, grid.cell_h, grid.bins_y);
    const int cell = grid2d_cell_id(ix, iy, grid.bins_x);
    cell_ids[static_cast<std::size_t>(i)] = cell;
    ++grid.offsets[static_cast<std::size_t>(cell + 1)];
  }
  for (int c = 1; c <= n_cells; ++c) {
    grid.offsets[static_cast<std::size_t>(c)] += grid.offsets[static_cast<std::size_t>(c - 1)];
  }
  grid.rows.assign(static_cast<std::size_t>(n), 0);
  std::vector<int> cursor = grid.offsets;
  for (int i = 0; i < n; ++i) {
    const int cell = cell_ids[static_cast<std::size_t>(i)];
    grid.rows[static_cast<std::size_t>(cursor[static_cast<std::size_t>(cell)]++)] = i;
  }
  return grid;
}

Grid3DIndex build_grid3d_index(const std::vector<double>& x,
                               const std::vector<double>& y,
                               const std::vector<double>& z,
                               const int bins_per_dim) {
  const int n = static_cast<int>(x.size());
  Grid3DIndex grid;
  grid.bins_x = std::max(1, bins_per_dim);
  grid.bins_y = std::max(1, bins_per_dim);
  grid.bins_z = std::max(1, bins_per_dim);
  auto mmx = std::minmax_element(x.begin(), x.end());
  auto mmy = std::minmax_element(y.begin(), y.end());
  auto mmz = std::minmax_element(z.begin(), z.end());
  grid.min_x = *mmx.first;
  grid.min_y = *mmy.first;
  grid.min_z = *mmz.first;
  const double max_x = *mmx.second;
  const double max_y = *mmy.second;
  const double max_z = *mmz.second;
  const double span_x = std::max(max_x - grid.min_x, std::numeric_limits<double>::epsilon());
  const double span_y = std::max(max_y - grid.min_y, std::numeric_limits<double>::epsilon());
  const double span_z = std::max(max_z - grid.min_z, std::numeric_limits<double>::epsilon());
  grid.cell_w = std::nextafter(span_x, std::numeric_limits<double>::infinity()) /
    static_cast<double>(grid.bins_x);
  grid.cell_h = std::nextafter(span_y, std::numeric_limits<double>::infinity()) /
    static_cast<double>(grid.bins_y);
  grid.cell_d = std::nextafter(span_z, std::numeric_limits<double>::infinity()) /
    static_cast<double>(grid.bins_z);

  const int n_cells = grid.bins_x * grid.bins_y * grid.bins_z;
  grid.offsets.assign(static_cast<std::size_t>(n_cells + 1), 0);
  std::vector<int> cell_ids(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const int ix = grid2d_coord(x[static_cast<std::size_t>(i)], grid.min_x, grid.cell_w, grid.bins_x);
    const int iy = grid2d_coord(y[static_cast<std::size_t>(i)], grid.min_y, grid.cell_h, grid.bins_y);
    const int iz = grid2d_coord(z[static_cast<std::size_t>(i)], grid.min_z, grid.cell_d, grid.bins_z);
    const int cell = grid3d_cell_id(ix, iy, iz, grid.bins_x, grid.bins_y);
    cell_ids[static_cast<std::size_t>(i)] = cell;
    ++grid.offsets[static_cast<std::size_t>(cell + 1)];
  }
  for (int c = 1; c <= n_cells; ++c) {
    grid.offsets[static_cast<std::size_t>(c)] += grid.offsets[static_cast<std::size_t>(c - 1)];
  }
  grid.rows.assign(static_cast<std::size_t>(n), 0);
  std::vector<int> cursor = grid.offsets;
  for (int i = 0; i < n; ++i) {
    const int cell = cell_ids[static_cast<std::size_t>(i)];
    grid.rows[static_cast<std::size_t>(cursor[static_cast<std::size_t>(cell)]++)] = i;
  }
  return grid;
}

void add_grid2d_cell_candidates(const std::vector<double>& x,
                                const std::vector<double>& y,
                                const Grid2DIndex& grid,
                                const int query,
                                const int ix,
                                const int iy,
                                const int k,
                                std::vector<Neighbor>& top) {
  if (ix < 0 || iy < 0 || ix >= grid.bins_x || iy >= grid.bins_y) return;
  const int cell = grid2d_cell_id(ix, iy, grid.bins_x);
  const int start = grid.offsets[static_cast<std::size_t>(cell)];
  const int end = grid.offsets[static_cast<std::size_t>(cell + 1)];
  const double qx = x[static_cast<std::size_t>(query)];
  const double qy = y[static_cast<std::size_t>(query)];
  for (int pos = start; pos < end; ++pos) {
    const int candidate = grid.rows[static_cast<std::size_t>(pos)];
    if (candidate == query) continue;
    const double dx = qx - x[static_cast<std::size_t>(candidate)];
    const double dy = qy - y[static_cast<std::size_t>(candidate)];
    insert_heap_top_double(candidate, dx * dx + dy * dy, k, top);
  }
}

double grid3d_lower_bound_outside_cube(const double x,
                                       const double y,
                                       const double z,
                                       const Grid3DIndex& grid,
                                       const int x0,
                                       const int x1,
                                       const int y0,
                                       const int y1,
                                       const int z0,
                                       const int z1) {
  double best = std::numeric_limits<double>::infinity();
  if (x0 > 0) {
    const double border = grid.min_x + static_cast<double>(x0) * grid.cell_w;
    const double dx = std::max(0.0, x - border);
    best = std::min(best, dx * dx);
  }
  if (x1 + 1 < grid.bins_x) {
    const double border = grid.min_x + static_cast<double>(x1 + 1) * grid.cell_w;
    const double dx = std::max(0.0, border - x);
    best = std::min(best, dx * dx);
  }
  if (y0 > 0) {
    const double border = grid.min_y + static_cast<double>(y0) * grid.cell_h;
    const double dy = std::max(0.0, y - border);
    best = std::min(best, dy * dy);
  }
  if (y1 + 1 < grid.bins_y) {
    const double border = grid.min_y + static_cast<double>(y1 + 1) * grid.cell_h;
    const double dy = std::max(0.0, border - y);
    best = std::min(best, dy * dy);
  }
  if (z0 > 0) {
    const double border = grid.min_z + static_cast<double>(z0) * grid.cell_d;
    const double dz = std::max(0.0, z - border);
    best = std::min(best, dz * dz);
  }
  if (z1 + 1 < grid.bins_z) {
    const double border = grid.min_z + static_cast<double>(z1 + 1) * grid.cell_d;
    const double dz = std::max(0.0, border - z);
    best = std::min(best, dz * dz);
  }
  return best;
}

void add_grid3d_cell_candidates(const std::vector<double>& x,
                                const std::vector<double>& y,
                                const std::vector<double>& z,
                                const Grid3DIndex& grid,
                                const int query,
                                const int ix,
                                const int iy,
                                const int iz,
                                const int k,
                                std::vector<Neighbor>& top) {
  if (ix < 0 || iy < 0 || iz < 0 ||
      ix >= grid.bins_x || iy >= grid.bins_y || iz >= grid.bins_z) {
    return;
  }
  const int cell = grid3d_cell_id(ix, iy, iz, grid.bins_x, grid.bins_y);
  const int start = grid.offsets[static_cast<std::size_t>(cell)];
  const int end = grid.offsets[static_cast<std::size_t>(cell + 1)];
  const double qx = x[static_cast<std::size_t>(query)];
  const double qy = y[static_cast<std::size_t>(query)];
  const double qz = z[static_cast<std::size_t>(query)];
  for (int pos = start; pos < end; ++pos) {
    const int candidate = grid.rows[static_cast<std::size_t>(pos)];
    if (candidate == query) continue;
    const double dx = qx - x[static_cast<std::size_t>(candidate)];
    const double dy = qy - y[static_cast<std::size_t>(candidate)];
    const double dz = qz - z[static_cast<std::size_t>(candidate)];
    insert_heap_top_double(candidate, dx * dx + dy * dy + dz * dz, k, top);
  }
}

void search_grid2d_exact(const std::vector<double>& x,
                         const std::vector<double>& y,
                         const Grid2DIndex& grid,
                         const int query,
                         const int k,
                         std::vector<Neighbor>& top) {
  top.clear();
  const double qx = x[static_cast<std::size_t>(query)];
  const double qy = y[static_cast<std::size_t>(query)];
  const int cx = grid2d_coord(qx, grid.min_x, grid.cell_w, grid.bins_x);
  const int cy = grid2d_coord(qy, grid.min_y, grid.cell_h, grid.bins_y);
  const int max_radius = std::max(grid.bins_x, grid.bins_y);

  for (int radius = 0; radius <= max_radius; ++radius) {
    const int raw_x0 = cx - radius;
    const int raw_x1 = cx + radius;
    const int raw_y0 = cy - radius;
    const int raw_y1 = cy + radius;
    const int x0 = std::max(0, raw_x0);
    const int x1 = std::min(grid.bins_x - 1, raw_x1);
    const int y0 = std::max(0, raw_y0);
    const int y1 = std::min(grid.bins_y - 1, raw_y1);

    if (radius == 0) {
      add_grid2d_cell_candidates(x, y, grid, query, cx, cy, k, top);
    } else {
      for (int ix = raw_x0; ix <= raw_x1; ++ix) {
        if (ix < 0 || ix >= grid.bins_x) continue;
        if (raw_y0 >= 0 && raw_y0 < grid.bins_y) {
          add_grid2d_cell_candidates(x, y, grid, query, ix, raw_y0, k, top);
        }
        if (raw_y1 != raw_y0 && raw_y1 >= 0 && raw_y1 < grid.bins_y) {
          add_grid2d_cell_candidates(x, y, grid, query, ix, raw_y1, k, top);
        }
      }
      for (int iy = raw_y0 + 1; iy <= raw_y1 - 1; ++iy) {
        if (iy < 0 || iy >= grid.bins_y) continue;
        if (raw_x0 >= 0 && raw_x0 < grid.bins_x) {
          add_grid2d_cell_candidates(x, y, grid, query, raw_x0, iy, k, top);
        }
        if (raw_x1 != raw_x0 && raw_x1 >= 0 && raw_x1 < grid.bins_x) {
          add_grid2d_cell_candidates(x, y, grid, query, raw_x1, iy, k, top);
        }
      }
    }

    if (static_cast<int>(top.size()) == k) {
      const double kth = top.front().distance;
      const double lower = grid2d_lower_bound_outside_square(qx, qy, grid, x0, x1, y0, y1);
      if (lower > kth) break;
    }
  }
}

void search_grid3d_exact(const std::vector<double>& x,
                         const std::vector<double>& y,
                         const std::vector<double>& z,
                         const Grid3DIndex& grid,
                         const int query,
                         const int k,
                         std::vector<Neighbor>& top) {
  top.clear();
  const double qx = x[static_cast<std::size_t>(query)];
  const double qy = y[static_cast<std::size_t>(query)];
  const double qz = z[static_cast<std::size_t>(query)];
  const int cx = grid2d_coord(qx, grid.min_x, grid.cell_w, grid.bins_x);
  const int cy = grid2d_coord(qy, grid.min_y, grid.cell_h, grid.bins_y);
  const int cz = grid2d_coord(qz, grid.min_z, grid.cell_d, grid.bins_z);
  const int max_radius = std::max(grid.bins_x, std::max(grid.bins_y, grid.bins_z));

  for (int radius = 0; radius <= max_radius; ++radius) {
    const int raw_x0 = cx - radius;
    const int raw_x1 = cx + radius;
    const int raw_y0 = cy - radius;
    const int raw_y1 = cy + radius;
    const int raw_z0 = cz - radius;
    const int raw_z1 = cz + radius;
    const int x0 = std::max(0, raw_x0);
    const int x1 = std::min(grid.bins_x - 1, raw_x1);
    const int y0 = std::max(0, raw_y0);
    const int y1 = std::min(grid.bins_y - 1, raw_y1);
    const int z0 = std::max(0, raw_z0);
    const int z1 = std::min(grid.bins_z - 1, raw_z1);

    if (radius == 0) {
      add_grid3d_cell_candidates(x, y, z, grid, query, cx, cy, cz, k, top);
    } else {
      for (int iz = raw_z0; iz <= raw_z1; ++iz) {
        if (iz < 0 || iz >= grid.bins_z) continue;
        for (int iy = raw_y0; iy <= raw_y1; ++iy) {
          if (iy < 0 || iy >= grid.bins_y) continue;
          for (int ix = raw_x0; ix <= raw_x1; ++ix) {
            if (ix < 0 || ix >= grid.bins_x) continue;
            if (ix != raw_x0 && ix != raw_x1 &&
                iy != raw_y0 && iy != raw_y1 &&
                iz != raw_z0 && iz != raw_z1) {
              continue;
            }
            add_grid3d_cell_candidates(x, y, z, grid, query, ix, iy, iz, k, top);
          }
        }
      }
    }

    if (static_cast<int>(top.size()) == k) {
      const double kth = top.front().distance;
      const double lower = grid3d_lower_bound_outside_cube(qx, qy, qz, grid, x0, x1, y0, y1, z0, z1);
      if (lower > kth) break;
    }
  }
}

void build_annoy_leaves_recursive(const float* data,
                                  const int n_features,
                                  const int leaf_size,
                                  std::vector<int>& items,
                                  std::vector<std::vector<int>>& leaves,
                                  std::vector<int>& row_leaf,
                                  std::mt19937& rng,
                                  std::normal_distribution<float>& normal) {
  if (items.empty()) return;
  if (static_cast<int>(items.size()) <= leaf_size) {
    const int leaf_id = static_cast<int>(leaves.size());
    for (const int item : items) {
      row_leaf[static_cast<std::size_t>(item)] = leaf_id;
    }
    leaves.push_back(items);
    return;
  }

  std::uniform_int_distribution<int> pick(0, static_cast<int>(items.size()) - 1);
  int a_pos = pick(rng);
  int b_pos = pick(rng);
  if (items.size() > 1u) {
    int guard = 0;
    while (b_pos == a_pos && guard < 16) {
      b_pos = pick(rng);
      ++guard;
    }
  }
  const int a = items[static_cast<std::size_t>(a_pos)];
  const int b = items[static_cast<std::size_t>(b_pos)];

  std::vector<float> direction(static_cast<std::size_t>(n_features));
  float norm = 0.0f;
  const float* xa = data + static_cast<std::size_t>(a) * n_features;
  const float* xb = data + static_cast<std::size_t>(b) * n_features;
  for (int c = 0; c < n_features; ++c) {
    const float value = xa[c] - xb[c];
    direction[static_cast<std::size_t>(c)] = value;
    norm += value * value;
  }
  if (!(norm > 1e-12f)) {
    for (int c = 0; c < n_features; ++c) {
      direction[static_cast<std::size_t>(c)] = normal(rng);
    }
  }

  std::vector<std::pair<float, int>> scores;
  scores.reserve(items.size());
  for (const int item : items) {
    const float* row = data + static_cast<std::size_t>(item) * n_features;
    float score = 0.0f;
    for (int c = 0; c < n_features; ++c) {
      score += row[c] * direction[static_cast<std::size_t>(c)];
    }
    scores.emplace_back(score, item);
  }

  const std::size_t median = scores.size() / 2u;
  std::nth_element(
    scores.begin(),
    scores.begin() + static_cast<std::ptrdiff_t>(median),
    scores.end(),
    [](const auto& left, const auto& right) {
      if (left.first == right.first) return left.second < right.second;
      return left.first < right.first;
    }
  );

  std::vector<int> left_items;
  std::vector<int> right_items;
  left_items.reserve(median);
  right_items.reserve(scores.size() - median);
  for (std::size_t i = 0; i < scores.size(); ++i) {
    if (i < median) {
      left_items.push_back(scores[i].second);
    } else {
      right_items.push_back(scores[i].second);
    }
  }

  if (left_items.empty() || right_items.empty()) {
    std::sort(items.begin(), items.end());
    const std::size_t half = items.size() / 2u;
    left_items.assign(items.begin(), items.begin() + static_cast<std::ptrdiff_t>(half));
    right_items.assign(items.begin() + static_cast<std::ptrdiff_t>(half), items.end());
  }

  build_annoy_leaves_recursive(
    data, n_features, leaf_size, left_items, leaves, row_leaf, rng, normal
  );
  build_annoy_leaves_recursive(
    data, n_features, leaf_size, right_items, leaves, row_leaf, rng, normal
  );
}

float euclidean_row_major_float(const float* data,
                                const int a,
                                const int b,
                                const int n_features) {
  return std::sqrt(std::max(
    squared_euclidean_row_major_float(data, a, b, n_features),
    0.0f
  ));
}

float euclidean_query_to_row_major_float(const float* data,
                                         const float* query,
                                         const int row,
                                         const int n_features) {
  const float* x = data + static_cast<std::size_t>(row) * n_features;
  float acc = 0.0f;
  for (int c = 0; c < n_features; ++c) {
    const float diff = query[c] - x[c];
    acc += diff * diff;
  }
  return std::sqrt(std::max(acc, 0.0f));
}

int build_vptree_recursive(const float* data,
                           const int n_features,
                           std::vector<int>& items,
                           std::vector<VpNode>& nodes) {
  if (items.empty()) return -1;
  const int node_id = static_cast<int>(nodes.size());
  nodes.push_back(VpNode{});

  const int vantage = items.back();
  items.pop_back();
  nodes[static_cast<std::size_t>(node_id)].index = vantage;
  if (items.empty()) return node_id;

  std::vector<std::pair<float, int>> distances;
  distances.reserve(items.size());
  for (const int item : items) {
    distances.emplace_back(
      euclidean_row_major_float(data, vantage, item, n_features),
      item
    );
  }

  const std::size_t median = distances.size() / 2u;
  std::nth_element(
    distances.begin(),
    distances.begin() + static_cast<std::ptrdiff_t>(median),
    distances.end(),
    [](const auto& a, const auto& b) {
      if (a.first == b.first) return a.second < b.second;
      return a.first < b.first;
    }
  );
  const float threshold = distances[median].first;
  nodes[static_cast<std::size_t>(node_id)].threshold = threshold;

  std::vector<int> left;
  std::vector<int> right;
  left.reserve(median + 1u);
  right.reserve(distances.size() - median);
  for (const auto& item : distances) {
    if (item.first <= threshold) {
      left.push_back(item.second);
    } else {
      right.push_back(item.second);
    }
  }

  nodes[static_cast<std::size_t>(node_id)].left =
    build_vptree_recursive(data, n_features, left, nodes);
  nodes[static_cast<std::size_t>(node_id)].right =
    build_vptree_recursive(data, n_features, right, nodes);
  return node_id;
}

void search_vptree(const float* data,
                   const int n_features,
                   const std::vector<VpNode>& nodes,
                   const int node_id,
                   const int query,
                   const int k,
                   std::vector<Neighbor>& top) {
  if (node_id < 0) return;
  const VpNode& node = nodes[static_cast<std::size_t>(node_id)];
  const double dist = static_cast<double>(
    euclidean_row_major_float(data, query, node.index, n_features)
  );
  if (node.index != query) {
    insert_heap_top_double(node.index, dist, k, top);
  }

  if (node.left < 0 && node.right < 0) return;

  if (dist < node.threshold) {
    search_vptree(data, n_features, nodes, node.left, query, k, top);
    const double tau_after = static_cast<int>(top.size()) == k
      ? top.front().distance
      : std::numeric_limits<double>::infinity();
    if (dist + tau_after >= node.threshold) {
      search_vptree(data, n_features, nodes, node.right, query, k, top);
    }
  } else {
    search_vptree(data, n_features, nodes, node.right, query, k, top);
    const double tau_after = static_cast<int>(top.size()) == k
      ? top.front().distance
      : std::numeric_limits<double>::infinity();
    if (dist - tau_after <= node.threshold) {
      search_vptree(data, n_features, nodes, node.left, query, k, top);
    }
  }
}

void search_vptree_query(const float* data,
                         const int n_features,
                         const std::vector<VpNode>& nodes,
                         const int node_id,
                         const float* query,
                         const int k,
                         std::vector<Neighbor>& top) {
  if (node_id < 0) return;
  const VpNode& node = nodes[static_cast<std::size_t>(node_id)];
  const double dist = static_cast<double>(
    euclidean_query_to_row_major_float(data, query, node.index, n_features)
  );
  insert_heap_top_double(node.index, dist, k, top);

  if (node.left < 0 && node.right < 0) return;

  if (dist < node.threshold) {
    search_vptree_query(data, n_features, nodes, node.left, query, k, top);
    const double tau_after = static_cast<int>(top.size()) == k
      ? top.front().distance
      : std::numeric_limits<double>::infinity();
    if (dist + tau_after >= node.threshold) {
      search_vptree_query(data, n_features, nodes, node.right, query, k, top);
    }
  } else {
    search_vptree_query(data, n_features, nodes, node.right, query, k, top);
    const double tau_after = static_cast<int>(top.size()) == k
      ? top.front().distance
      : std::numeric_limits<double>::infinity();
    if (dist - tau_after <= node.threshold) {
      search_vptree_query(data, n_features, nodes, node.left, query, k, top);
    }
  }
}

} // namespace

// [[Rcpp::export]]
List nn_cpp(NumericMatrix data,
            NumericMatrix points,
            int k,
            std::string method,
            bool square,
            bool sorted,
            double p,
            bool parallel,
            int cores,
            bool exclude_self) {
  const int n_data = data.nrow();
  const int n_points = points.nrow();
  const int n_features = data.ncol();
  if (points.ncol() != n_features) Rcpp::stop("data and points must have the same number of columns");
  if (exclude_self && n_data != n_points) {
    Rcpp::stop("exclude_self is only valid when data and points have the same number of rows");
  }
  const int max_k = exclude_self ? n_data - 1 : n_data;
  if (k < 1 || k > max_k) Rcpp::stop("k must be in the available neighbor range");
  if (method != "euclidean" && method != "manhattan" && method != "minkowski" &&
      method != "cosine" && method != "correlation") {
    Rcpp::stop("unsupported method");
  }
  if (method == "minkowski" && (!std::isfinite(p) || p <= 0.0)) {
    Rcpp::stop("p must be positive for minkowski distance");
  }
  const DistanceKind distance_kind = distance_kind_from_method(method);

  IntegerMatrix indices(n_points, k);
  NumericMatrix distances(n_points, k);
  int* indices_ptr = indices.begin();
  double* distances_ptr = distances.begin();
  const double work_size = static_cast<double>(n_data) *
    static_cast<double>(n_points) *
    static_cast<double>(n_features);
  const bool use_parallel = parallel && work_size >= 1000000.0;
  const int n_threads = requested_threads(use_parallel, cores, n_points);

  if (distance_kind == DistanceKind::Euclidean && !square && fortran_nn_enabled()) {
    auto write_fortran = [&](const int query_start, const int query_end) {
      if (query_end <= query_start) return;
      fastembedr_knn_euclidean_range(
        data.begin(),
        points.begin(),
        n_data,
        n_points,
        n_features,
        k,
        exclude_self ? 1 : 0,
        query_start + 1,
        query_end,
        indices_ptr,
        distances_ptr
      );
    };

    if (n_threads == 1) {
      write_fortran(0, n_points);
    } else {
      std::vector<std::thread> workers;
      workers.reserve(n_threads);
      for (int t = 0; t < n_threads; ++t) {
        const int start = (n_points * t) / n_threads;
        const int end = (n_points * (t + 1)) / n_threads;
        workers.emplace_back(write_fortran, start, end);
      }
      for (auto& worker : workers) worker.join();
    }

    List result = List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances
    );
    result.attr("memory_layout") = "fortran_column_major";
    result.attr("row_major_copy") = false;
    result.attr("row_major_copy_mb") = 0.0;
    return result;
  }

  std::vector<double> data_row_major;
  std::vector<double> points_row_major;
  const bool same_matrix = data.begin() == points.begin() && n_data == n_points;
  const double input_copy_bytes =
    static_cast<double>(n_features) *
    static_cast<double>(n_data + (same_matrix ? 0 : n_points)) *
    static_cast<double>(sizeof(double));
  const bool use_row_major = use_row_major_distance_layout(input_copy_bytes);
  const double* data_ptr = data.begin();
  const double* points_ptr = points.begin();
  if (use_row_major) {
    copy_row_major(data.begin(), data_row_major, n_data, n_features);
    data_ptr = data_row_major.data();
    points_ptr = data_ptr;
    if (!same_matrix) {
      copy_row_major(points.begin(), points_row_major, n_points, n_features);
      points_ptr = points_row_major.data();
    }
  }
  const bool use_fixed_topk = k * 8 < n_data;

  const auto write_result = [&](const int query_start, const int query_end) {
    if (distance_kind == DistanceKind::Euclidean) {
      if (!use_row_major) {
        write_knn_rows<DistanceKind::Euclidean, false>(
          data_ptr, points_ptr, n_data, n_points, n_features, k, square, sorted,
          p, use_fixed_topk, exclude_self, indices_ptr, distances_ptr, query_start, query_end
        );
      } else {
        write_knn_rows<DistanceKind::Euclidean, true>(
          data_ptr, points_ptr, n_data, n_points, n_features, k, square, sorted,
          p, use_fixed_topk, exclude_self, indices_ptr, distances_ptr, query_start, query_end
        );
      }
    } else if (distance_kind == DistanceKind::Manhattan) {
      if (!use_row_major) {
        write_knn_rows<DistanceKind::Manhattan, false>(
          data_ptr, points_ptr, n_data, n_points, n_features, k, square, sorted,
          p, use_fixed_topk, exclude_self, indices_ptr, distances_ptr, query_start, query_end
        );
      } else {
        write_knn_rows<DistanceKind::Manhattan, true>(
          data_ptr, points_ptr, n_data, n_points, n_features, k, square, sorted,
          p, use_fixed_topk, exclude_self, indices_ptr, distances_ptr, query_start, query_end
        );
      }
    } else if (distance_kind == DistanceKind::Cosine) {
      if (!use_row_major) {
        write_knn_rows<DistanceKind::Cosine, false>(
          data_ptr, points_ptr, n_data, n_points, n_features, k, square, sorted,
          p, use_fixed_topk, exclude_self, indices_ptr, distances_ptr, query_start, query_end
        );
      } else {
        write_knn_rows<DistanceKind::Cosine, true>(
          data_ptr, points_ptr, n_data, n_points, n_features, k, square, sorted,
          p, use_fixed_topk, exclude_self, indices_ptr, distances_ptr, query_start, query_end
        );
      }
    } else if (distance_kind == DistanceKind::Correlation) {
      if (!use_row_major) {
        write_knn_rows<DistanceKind::Correlation, false>(
          data_ptr, points_ptr, n_data, n_points, n_features, k, square, sorted,
          p, use_fixed_topk, exclude_self, indices_ptr, distances_ptr, query_start, query_end
        );
      } else {
        write_knn_rows<DistanceKind::Correlation, true>(
          data_ptr, points_ptr, n_data, n_points, n_features, k, square, sorted,
          p, use_fixed_topk, exclude_self, indices_ptr, distances_ptr, query_start, query_end
        );
      }
    } else {
      if (!use_row_major) {
        write_knn_rows<DistanceKind::Minkowski, false>(
          data_ptr, points_ptr, n_data, n_points, n_features, k, square, sorted,
          p, use_fixed_topk, exclude_self, indices_ptr, distances_ptr, query_start, query_end
        );
      } else {
        write_knn_rows<DistanceKind::Minkowski, true>(
          data_ptr, points_ptr, n_data, n_points, n_features, k, square, sorted,
          p, use_fixed_topk, exclude_self, indices_ptr, distances_ptr, query_start, query_end
        );
      }
    }
  };

  if (n_threads == 1) {
    write_result(0, n_points);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
      const int start = (n_points * t) / n_threads;
      const int end = (n_points * (t + 1)) / n_threads;
      workers.emplace_back(write_result, start, end);
    }
    for (auto& worker : workers) worker.join();
  }

  List result = List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances
  );
  result.attr("memory_layout") = use_row_major
    ? "row_major_contiguous"
    : "r_column_major";
  result.attr("row_major_copy") = use_row_major;
  result.attr("row_major_copy_mb") = use_row_major
    ? input_copy_bytes / (1024.0 * 1024.0)
    : 0.0;
  return result;
}

// [[Rcpp::export]]
IntegerMatrix nndescent_candidate_matrix_cpp(IntegerMatrix indices,
                                             int n_sources,
                                             int n_neighbors) {
  const int n = indices.nrow();
  const int k = indices.ncol();
  if (n < 1 || k < 1) Rcpp::stop("indices must be a non-empty matrix");
  n_sources = std::max(1, std::min(n_sources, k));
  n_neighbors = std::max(1, std::min(n_neighbors, k));
  std::vector<int> seen(static_cast<std::size_t>(n) + 1, 0);
  std::vector<int> unique_counts(static_cast<std::size_t>(n), 0);
  int max_unique = 0;

  for (int row = 0; row < n; ++row) {
    const int stamp = row + 1;
    int count = 0;
    const auto add_candidate = [&](const int candidate) {
      if (candidate < 1 || candidate > n || candidate == row + 1) return;
      if (seen[static_cast<std::size_t>(candidate)] == stamp) return;
      seen[static_cast<std::size_t>(candidate)] = stamp;
      ++count;
    };

    for (int col = 0; col < k; ++col) add_candidate(indices(row, col));
    for (int source_col = 0; source_col < n_sources; ++source_col) {
      const int source = indices(row, source_col) - 1;
      if (source < 0 || source >= n) continue;
      for (int neighbor_col = 0; neighbor_col < n_neighbors; ++neighbor_col) {
        add_candidate(indices(source, neighbor_col));
      }
    }
    unique_counts[static_cast<std::size_t>(row)] = count;
    if (count > max_unique) max_unique = count;
  }

  max_unique = std::max(k, max_unique);
  IntegerMatrix out(n, max_unique);
  std::fill(seen.begin(), seen.end(), 0);

  for (int row = 0; row < n; ++row) {
    const int stamp = row + 1;
    int out_col = 0;
    const auto add_candidate = [&](const int candidate) {
      if (candidate < 1 || candidate > n || candidate == row + 1) return;
      if (seen[static_cast<std::size_t>(candidate)] == stamp) return;
      seen[static_cast<std::size_t>(candidate)] = stamp;
      if (out_col < max_unique) {
        out(row, out_col) = candidate;
        ++out_col;
      }
    };

    for (int col = 0; col < k; ++col) add_candidate(indices(row, col));
    for (int source_col = 0; source_col < n_sources; ++source_col) {
      const int source = indices(row, source_col) - 1;
      if (source < 0 || source >= n) continue;
      for (int neighbor_col = 0; neighbor_col < n_neighbors; ++neighbor_col) {
        add_candidate(indices(source, neighbor_col));
      }
    }
  }
  double mean_unique = 0.0;
  for (const int count : unique_counts) mean_unique += static_cast<double>(count);
  mean_unique /= static_cast<double>(n);
  out.attr("mean_unique_candidates") = mean_unique;
  out.attr("max_unique_candidates") = max_unique;
  out.attr("raw_candidate_columns") = k + n_sources * n_neighbors;
  return out;
}

// [[Rcpp::export]]
IntegerMatrix nndescent_candidate_matrix_mlx_cpp(IntegerMatrix indices,
                                                 LogicalMatrix flags,
                                                 int n_sources,
                                                 int n_neighbors,
                                                 bool use_reverse,
                                                 bool active_only) {
  const int n = indices.nrow();
  const int k = indices.ncol();
  if (n < 1 || k < 1) Rcpp::stop("indices must be a non-empty matrix");
  if (flags.nrow() != n || flags.ncol() != k) {
    Rcpp::stop("flags must have the same dimensions as indices");
  }
  n_sources = std::max(1, std::min(n_sources, k));
  n_neighbors = std::max(1, std::min(n_neighbors, k));

  // Adapted from the Apache-2.0 mlx-vis NNDescent schedule
  // (https://github.com/hanxiao/mlx-vis, mlx_vis/_nndescent/nndescent.py):
  // keep NEW-neighbour expansion when converging and skip reverse candidates
  // once the graph is nearly stable. The implementation below is native C++
  // and feeds the package's Metal row-candidate refinement kernel.
  const int reverse_limit = use_reverse ? std::max(1, std::min(k, n_sources)) : 0;
  std::vector<std::vector<int> > reverse_lists;
  if (use_reverse) {
    reverse_lists.resize(static_cast<std::size_t>(n));
    for (int row = 0; row < n; ++row) {
      for (int col = 0; col < k; ++col) {
        const int neighbor = indices(row, col);
        if (neighbor < 1 || neighbor > n || neighbor == row + 1) continue;
        std::vector<int>& bucket = reverse_lists[static_cast<std::size_t>(neighbor - 1)];
        if (static_cast<int>(bucket.size()) < reverse_limit) {
          bucket.push_back(row + 1);
        }
      }
    }
  }

  std::vector<int> seen(static_cast<std::size_t>(n) + 1, 0);
  std::vector<int> unique_counts(static_cast<std::size_t>(n), 0);
  int max_unique = 0;
  int active_rows = 0;

  const auto row_has_new = [&](const int row) {
    for (int col = 0; col < k; ++col) {
      if (flags(row, col) == TRUE) return true;
    }
    return false;
  };

  const auto source_col_at = [&](const int row, const int source_pos) {
    int seen_sources = 0;
    for (int col = 0; col < k; ++col) {
      if (flags(row, col) == TRUE) {
        if (seen_sources == source_pos) return col;
        ++seen_sources;
      }
    }
    for (int col = 0; col < k; ++col) {
      if (flags(row, col) != TRUE) {
        if (seen_sources == source_pos) return col;
        ++seen_sources;
      }
    }
    return -1;
  };

  for (int row = 0; row < n; ++row) {
    const bool active = row_has_new(row);
    if (active) ++active_rows;
    const int stamp = row + 1;
    int count = 0;
    const auto add_candidate = [&](const int candidate) {
      if (candidate < 1 || candidate > n || candidate == row + 1) return;
      if (seen[static_cast<std::size_t>(candidate)] == stamp) return;
      seen[static_cast<std::size_t>(candidate)] = stamp;
      ++count;
    };

    for (int col = 0; col < k; ++col) add_candidate(indices(row, col));

    if (!active_only || active) {
      for (int source_pos = 0; source_pos < n_sources; ++source_pos) {
        const int source_col = active_only ? source_col_at(row, source_pos) : source_pos;
        if (source_col < 0) continue;
        const int source = indices(row, source_col) - 1;
        if (source < 0 || source >= n) continue;
        for (int neighbor_col = 0; neighbor_col < n_neighbors; ++neighbor_col) {
          add_candidate(indices(source, neighbor_col));
        }
      }

      if (use_reverse) {
        const std::vector<int>& rev = reverse_lists[static_cast<std::size_t>(row)];
        for (const int source_one_based : rev) {
          add_candidate(source_one_based);
          const int source = source_one_based - 1;
          for (int neighbor_col = 0; neighbor_col < n_neighbors; ++neighbor_col) {
            add_candidate(indices(source, neighbor_col));
          }
        }
      }
    }

    unique_counts[static_cast<std::size_t>(row)] = count;
    if (count > max_unique) max_unique = count;
  }

  max_unique = std::max(k, max_unique);
  IntegerMatrix out(n, max_unique);
  std::fill(seen.begin(), seen.end(), 0);

  for (int row = 0; row < n; ++row) {
    const bool active = row_has_new(row);
    const int stamp = row + 1;
    int out_col = 0;
    const auto add_candidate = [&](const int candidate) {
      if (candidate < 1 || candidate > n || candidate == row + 1) return;
      if (seen[static_cast<std::size_t>(candidate)] == stamp) return;
      seen[static_cast<std::size_t>(candidate)] = stamp;
      if (out_col < max_unique) {
        out(row, out_col) = candidate;
        ++out_col;
      }
    };

    for (int col = 0; col < k; ++col) add_candidate(indices(row, col));

    if (!active_only || active) {
      for (int source_pos = 0; source_pos < n_sources; ++source_pos) {
        const int source_col = active_only ? source_col_at(row, source_pos) : source_pos;
        if (source_col < 0) continue;
        const int source = indices(row, source_col) - 1;
        if (source < 0 || source >= n) continue;
        for (int neighbor_col = 0; neighbor_col < n_neighbors; ++neighbor_col) {
          add_candidate(indices(source, neighbor_col));
        }
      }

      if (use_reverse) {
        const std::vector<int>& rev = reverse_lists[static_cast<std::size_t>(row)];
        for (const int source_one_based : rev) {
          add_candidate(source_one_based);
          const int source = source_one_based - 1;
          for (int neighbor_col = 0; neighbor_col < n_neighbors; ++neighbor_col) {
            add_candidate(indices(source, neighbor_col));
          }
        }
      }
    }
  }

  double mean_unique = 0.0;
  for (const int count : unique_counts) mean_unique += static_cast<double>(count);
  mean_unique /= static_cast<double>(n);
  out.attr("mean_unique_candidates") = mean_unique;
  out.attr("max_unique_candidates") = max_unique;
  out.attr("raw_candidate_columns") =
    k + n_sources * n_neighbors + reverse_limit * (1 + n_neighbors);
  out.attr("active_rows") = active_only ? active_rows : n;
  out.attr("use_reverse") = use_reverse;
  out.attr("active_only") = active_only;
  out.attr("sources") = n_sources;
  out.attr("neighbors") = n_neighbors;
  return out;
}

// [[Rcpp::export]]
List nndescent_candidate_matrix_mlx_subset_cpp(IntegerMatrix indices,
                                               LogicalMatrix flags,
                                               int n_sources,
                                               int n_neighbors,
                                               bool use_reverse) {
  const int n = indices.nrow();
  const int k = indices.ncol();
  if (n < 1 || k < 1) Rcpp::stop("indices must be a non-empty matrix");
  if (flags.nrow() != n || flags.ncol() != k) {
    Rcpp::stop("flags must have the same dimensions as indices");
  }
  n_sources = std::max(1, std::min(n_sources, k));
  n_neighbors = std::max(1, std::min(n_neighbors, k));

  const int reverse_limit = use_reverse ? std::max(1, std::min(k, n_sources)) : 0;
  std::vector<std::vector<int> > reverse_lists;
  if (use_reverse) {
    reverse_lists.resize(static_cast<std::size_t>(n));
    for (int row = 0; row < n; ++row) {
      for (int col = 0; col < k; ++col) {
        const int neighbor = indices(row, col);
        if (neighbor < 1 || neighbor > n || neighbor == row + 1) continue;
        std::vector<int>& bucket = reverse_lists[static_cast<std::size_t>(neighbor - 1)];
        if (static_cast<int>(bucket.size()) < reverse_limit) {
          bucket.push_back(row + 1);
        }
      }
    }
  }

  const auto row_has_new = [&](const int row) {
    for (int col = 0; col < k; ++col) {
      if (flags(row, col) == TRUE) return true;
    }
    return false;
  };

  const auto source_col_at = [&](const int row, const int source_pos) {
    int seen_sources = 0;
    for (int col = 0; col < k; ++col) {
      if (flags(row, col) == TRUE) {
        if (seen_sources == source_pos) return col;
        ++seen_sources;
      }
    }
    for (int col = 0; col < k; ++col) {
      if (flags(row, col) != TRUE) {
        if (seen_sources == source_pos) return col;
        ++seen_sources;
      }
    }
    return -1;
  };

  std::vector<int> active_rows;
  active_rows.reserve(static_cast<std::size_t>(n));
  for (int row = 0; row < n; ++row) {
    if (row_has_new(row)) active_rows.push_back(row);
  }

  std::vector<int> seen(static_cast<std::size_t>(n) + 1, 0);
  std::vector<int> unique_counts(active_rows.size(), 0);
  int max_unique = 0;

  for (std::size_t active_pos = 0; active_pos < active_rows.size(); ++active_pos) {
    const int row = active_rows[active_pos];
    const int stamp = row + 1;
    int count = 0;
    const auto add_candidate = [&](const int candidate) {
      if (candidate < 1 || candidate > n || candidate == row + 1) return;
      if (seen[static_cast<std::size_t>(candidate)] == stamp) return;
      seen[static_cast<std::size_t>(candidate)] = stamp;
      ++count;
    };

    for (int col = 0; col < k; ++col) add_candidate(indices(row, col));
    for (int source_pos = 0; source_pos < n_sources; ++source_pos) {
      const int source_col = source_col_at(row, source_pos);
      if (source_col < 0) continue;
      const int source = indices(row, source_col) - 1;
      if (source < 0 || source >= n) continue;
      for (int neighbor_col = 0; neighbor_col < n_neighbors; ++neighbor_col) {
        add_candidate(indices(source, neighbor_col));
      }
    }

    if (use_reverse) {
      const std::vector<int>& rev = reverse_lists[static_cast<std::size_t>(row)];
      for (const int source_one_based : rev) {
        add_candidate(source_one_based);
        const int source = source_one_based - 1;
        for (int neighbor_col = 0; neighbor_col < n_neighbors; ++neighbor_col) {
          add_candidate(indices(source, neighbor_col));
        }
      }
    }

    unique_counts[active_pos] = count;
    if (count > max_unique) max_unique = count;
  }

  max_unique = std::max(k, max_unique);
  IntegerMatrix out(static_cast<int>(active_rows.size()), max_unique);
  std::fill(seen.begin(), seen.end(), 0);

  for (std::size_t active_pos = 0; active_pos < active_rows.size(); ++active_pos) {
    const int row = active_rows[active_pos];
    const int stamp = row + 1;
    int out_col = 0;
    const auto add_candidate = [&](const int candidate) {
      if (candidate < 1 || candidate > n || candidate == row + 1) return;
      if (seen[static_cast<std::size_t>(candidate)] == stamp) return;
      seen[static_cast<std::size_t>(candidate)] = stamp;
      if (out_col < max_unique) {
        out(static_cast<int>(active_pos), out_col) = candidate;
        ++out_col;
      }
    };

    for (int col = 0; col < k; ++col) add_candidate(indices(row, col));
    for (int source_pos = 0; source_pos < n_sources; ++source_pos) {
      const int source_col = source_col_at(row, source_pos);
      if (source_col < 0) continue;
      const int source = indices(row, source_col) - 1;
      if (source < 0 || source >= n) continue;
      for (int neighbor_col = 0; neighbor_col < n_neighbors; ++neighbor_col) {
        add_candidate(indices(source, neighbor_col));
      }
    }

    if (use_reverse) {
      const std::vector<int>& rev = reverse_lists[static_cast<std::size_t>(row)];
      for (const int source_one_based : rev) {
        add_candidate(source_one_based);
        const int source = source_one_based - 1;
        for (int neighbor_col = 0; neighbor_col < n_neighbors; ++neighbor_col) {
          add_candidate(indices(source, neighbor_col));
        }
      }
    }
  }

  IntegerVector query_rows(static_cast<int>(active_rows.size()));
  for (std::size_t i = 0; i < active_rows.size(); ++i) {
    query_rows[static_cast<int>(i)] = active_rows[i] + 1;
  }

  double mean_unique = 0.0;
  for (const int count : unique_counts) mean_unique += static_cast<double>(count);
  if (!unique_counts.empty()) mean_unique /= static_cast<double>(unique_counts.size());
  out.attr("mean_unique_candidates") = mean_unique;
  out.attr("max_unique_candidates") = max_unique;
  out.attr("raw_candidate_columns") =
    k + n_sources * n_neighbors + reverse_limit * (1 + n_neighbors);
  out.attr("active_rows") = static_cast<int>(active_rows.size());
  out.attr("use_reverse") = use_reverse;
  out.attr("active_only") = true;
  out.attr("sources") = n_sources;
  out.attr("neighbors") = n_neighbors;

  return List::create(
    Rcpp::Named("candidates") = out,
    Rcpp::Named("query_rows") = query_rows
  );
}

// [[Rcpp::export]]
List landmark_candidate_knn_cpp(NumericMatrix data,
                                IntegerMatrix projection_indices,
                                int k,
                                int bucket_cols,
                                int query_cols,
                                bool parallel,
                                int cores) {
  const int n = data.nrow();
  const int n_features = data.ncol();
  const int projection_k = projection_indices.ncol();
  if (n < 2) Rcpp::stop("data must have at least two rows");
  if (projection_indices.nrow() != n) {
    Rcpp::stop("projection_indices row count must match data");
  }
  if (projection_k < 1) Rcpp::stop("projection_indices must have at least one column");
  if (k < 1 || k >= n) Rcpp::stop("k must be in [1, nrow(data) - 1]");
  bucket_cols = std::max(1, std::min(bucket_cols, projection_k));
  query_cols = std::max(1, std::min(query_cols, projection_k));

  int n_landmarks = 0;
  for (int i = 0; i < n; ++i) {
    for (int c = 0; c < projection_k; ++c) {
      const int idx = projection_indices(i, c);
      if (idx < 1) Rcpp::stop("projection_indices must be 1-based positive integers");
      n_landmarks = std::max(n_landmarks, idx);
    }
  }

  std::vector<double> data_row_major;
  copy_row_major(data.begin(), data_row_major, n, n_features);

  std::vector<std::vector<int>> buckets(static_cast<std::size_t>(n_landmarks));
  for (int i = 0; i < n; ++i) {
    std::vector<int> used;
    used.reserve(bucket_cols);
    for (int c = 0; c < bucket_cols; ++c) {
      const int landmark = projection_indices(i, c) - 1;
      if (landmark < 0 || landmark >= n_landmarks) continue;
      if (std::find(used.begin(), used.end(), landmark) != used.end()) continue;
      used.push_back(landmark);
      buckets[static_cast<std::size_t>(landmark)].push_back(i);
    }
  }

  IntegerMatrix indices(n, k);
  NumericMatrix distances(n, k);
  int* indices_ptr = indices.begin();
  double* distances_ptr = distances.begin();
  const int n_threads = requested_threads(parallel, cores, n);

  auto write_rows = [&](const int row_start, const int row_end) {
    std::vector<int> seen(static_cast<std::size_t>(n), 0);
    int stamp = 1;
    std::vector<Neighbor> top;
    top.reserve(k);
    std::vector<int> used_landmarks;
    used_landmarks.reserve(query_cols);

    for (int q = row_start; q < row_end; ++q) {
      top.clear();
      used_landmarks.clear();
      if (stamp == std::numeric_limits<int>::max()) {
        std::fill(seen.begin(), seen.end(), 0);
        stamp = 1;
      }
      ++stamp;
      seen[static_cast<std::size_t>(q)] = stamp;

      for (int c = 0; c < query_cols; ++c) {
        const int landmark = projection_indices(q, c) - 1;
        if (landmark < 0 || landmark >= n_landmarks) continue;
        if (std::find(used_landmarks.begin(), used_landmarks.end(), landmark) != used_landmarks.end()) {
          continue;
        }
        used_landmarks.push_back(landmark);
        const std::vector<int>& bucket = buckets[static_cast<std::size_t>(landmark)];
        for (const int candidate : bucket) {
          if (seen[static_cast<std::size_t>(candidate)] == stamp) continue;
          seen[static_cast<std::size_t>(candidate)] = stamp;
          add_candidate_topk(data_row_major.data(), q, candidate, n_features, k, top);
        }
      }

      if (static_cast<int>(top.size()) < k) {
        for (int candidate = 0; candidate < n; ++candidate) {
          if (seen[static_cast<std::size_t>(candidate)] == stamp) continue;
          seen[static_cast<std::size_t>(candidate)] = stamp;
          add_candidate_topk(data_row_major.data(), q, candidate, n_features, k, top);
        }
      }

      if (static_cast<int>(top.size()) == k) {
        std::sort_heap(top.begin(), top.end(), neighbor_less);
      } else {
        std::sort(top.begin(), top.end(), neighbor_less);
      }
      for (int j = 0; j < k; ++j) {
        indices_ptr[static_cast<std::size_t>(j) * n + q] =
          top[static_cast<std::size_t>(j)].index + 1;
        distances_ptr[static_cast<std::size_t>(j) * n + q] =
          top[static_cast<std::size_t>(j)].distance;
      }
    }
  };

  if (n_threads == 1) {
    write_rows(0, n);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
      const int start = (n * t) / n_threads;
      const int end = (n * (t + 1)) / n_threads;
      workers.emplace_back(write_rows, start, end);
    }
    for (auto& worker : workers) worker.join();
  }

  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances
  );
}

// [[Rcpp::export]]
List landmark_candidate_knn_subset_cpp(NumericMatrix data,
                                       IntegerMatrix projection_indices,
                                       IntegerVector query_rows,
                                       int k,
                                       int bucket_cols,
                                       int query_cols,
                                       bool parallel,
                                       int cores) {
  const int n = data.nrow();
  const int n_features = data.ncol();
  const int projection_k = projection_indices.ncol();
  const int n_query = query_rows.size();
  if (n < 2) Rcpp::stop("data must have at least two rows");
  if (projection_indices.nrow() != n) {
    Rcpp::stop("projection_indices row count must match data");
  }
  if (projection_k < 1) Rcpp::stop("projection_indices must have at least one column");
  if (n_query < 1) Rcpp::stop("query_rows must contain at least one row");
  if (k < 1 || k >= n) Rcpp::stop("k must be in [1, nrow(data) - 1]");
  bucket_cols = std::max(1, std::min(bucket_cols, projection_k));
  query_cols = std::max(1, std::min(query_cols, projection_k));

  std::vector<int> rows(static_cast<std::size_t>(n_query));
  for (int i = 0; i < n_query; ++i) {
    const int row = query_rows[i] - 1;
    if (row < 0 || row >= n) Rcpp::stop("query_rows must contain 1-based row indices");
    rows[static_cast<std::size_t>(i)] = row;
  }

  int n_landmarks = 0;
  for (int i = 0; i < n; ++i) {
    for (int c = 0; c < projection_k; ++c) {
      const int idx = projection_indices(i, c);
      if (idx < 1) Rcpp::stop("projection_indices must be 1-based positive integers");
      n_landmarks = std::max(n_landmarks, idx);
    }
  }

  std::vector<double> data_row_major;
  copy_row_major(data.begin(), data_row_major, n, n_features);

  std::vector<std::vector<int>> buckets(static_cast<std::size_t>(n_landmarks));
  for (int i = 0; i < n; ++i) {
    std::vector<int> used;
    used.reserve(bucket_cols);
    for (int c = 0; c < bucket_cols; ++c) {
      const int landmark = projection_indices(i, c) - 1;
      if (landmark < 0 || landmark >= n_landmarks) continue;
      if (std::find(used.begin(), used.end(), landmark) != used.end()) continue;
      used.push_back(landmark);
      buckets[static_cast<std::size_t>(landmark)].push_back(i);
    }
  }

  IntegerMatrix indices(n_query, k);
  NumericMatrix distances(n_query, k);
  int* indices_ptr = indices.begin();
  double* distances_ptr = distances.begin();
  const int n_threads = requested_threads(parallel, cores, n_query);

  auto write_rows = [&](const int row_start, const int row_end) {
    std::vector<int> seen(static_cast<std::size_t>(n), 0);
    int stamp = 1;
    std::vector<Neighbor> top;
    top.reserve(k);
    std::vector<int> used_landmarks;
    used_landmarks.reserve(query_cols);

    for (int local_q = row_start; local_q < row_end; ++local_q) {
      const int q = rows[static_cast<std::size_t>(local_q)];
      top.clear();
      used_landmarks.clear();
      if (stamp == std::numeric_limits<int>::max()) {
        std::fill(seen.begin(), seen.end(), 0);
        stamp = 1;
      }
      ++stamp;
      seen[static_cast<std::size_t>(q)] = stamp;

      for (int c = 0; c < query_cols; ++c) {
        const int landmark = projection_indices(q, c) - 1;
        if (landmark < 0 || landmark >= n_landmarks) continue;
        if (std::find(used_landmarks.begin(), used_landmarks.end(), landmark) != used_landmarks.end()) {
          continue;
        }
        used_landmarks.push_back(landmark);
        const std::vector<int>& bucket = buckets[static_cast<std::size_t>(landmark)];
        for (const int candidate : bucket) {
          if (seen[static_cast<std::size_t>(candidate)] == stamp) continue;
          seen[static_cast<std::size_t>(candidate)] = stamp;
          add_candidate_topk(data_row_major.data(), q, candidate, n_features, k, top);
        }
      }

      if (static_cast<int>(top.size()) < k) {
        for (int candidate = 0; candidate < n; ++candidate) {
          if (seen[static_cast<std::size_t>(candidate)] == stamp) continue;
          seen[static_cast<std::size_t>(candidate)] = stamp;
          add_candidate_topk(data_row_major.data(), q, candidate, n_features, k, top);
        }
      }

      if (static_cast<int>(top.size()) == k) {
        std::sort_heap(top.begin(), top.end(), neighbor_less);
      } else {
        std::sort(top.begin(), top.end(), neighbor_less);
      }
      for (int j = 0; j < k; ++j) {
        indices_ptr[static_cast<std::size_t>(j) * n_query + local_q] =
          top[static_cast<std::size_t>(j)].index + 1;
        distances_ptr[static_cast<std::size_t>(j) * n_query + local_q] =
          top[static_cast<std::size_t>(j)].distance;
      }
    }
  };

  if (n_threads == 1) {
    write_rows(0, n_query);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
      const int start = (n_query * t) / n_threads;
      const int end = (n_query * (t + 1)) / n_threads;
      workers.emplace_back(write_rows, start, end);
    }
    for (auto& worker : workers) worker.join();
  }

  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("row_ids") = query_rows
  );
}

// [[Rcpp::export]]
List landmark_projection_knn_approx_cpp(NumericMatrix landmarks,
                                        NumericMatrix queries,
                                        int k,
                                        int n_projections,
                                        int window,
                                        int seed,
                                        bool parallel,
                                        int cores) {
  const int n_landmarks = landmarks.nrow();
  const int n_queries = queries.nrow();
  const int n_features = landmarks.ncol();
  if (n_landmarks < 1) Rcpp::stop("landmarks must have at least one row");
  if (n_queries < 1) Rcpp::stop("queries must have at least one row");
  if (n_features < 1) Rcpp::stop("landmarks must have at least one column");
  if (queries.ncol() != n_features) {
    Rcpp::stop("landmarks and queries must have the same number of columns");
  }
  if (k < 1 || k > n_landmarks) Rcpp::stop("k must be in [1, nrow(landmarks)]");
  n_projections = std::max(1, std::min(n_projections, 64));
  window = std::max(1, std::min(window, n_landmarks));
  const int n_threads = requested_threads(parallel, cores, n_queries);

  std::vector<float> landmark_data;
  std::vector<float> query_data;
  copy_row_major_float(landmarks.begin(), landmark_data, n_landmarks, n_features);
  copy_row_major_float(queries.begin(), query_data, n_queries, n_features);

  std::vector<float> projection_vectors(
    static_cast<std::size_t>(n_projections) * n_features,
    0.0f
  );
  const int axis_projections = std::min(n_features, std::min(n_projections, 8));
  for (int pidx = 0; pidx < axis_projections; ++pidx) {
    projection_vectors[static_cast<std::size_t>(pidx) * n_features + pidx] = 1.0f;
  }
  std::mt19937 rng(static_cast<std::uint32_t>(seed));
  std::normal_distribution<float> normal(0.0f, 1.0f);
  for (int pidx = axis_projections; pidx < n_projections; ++pidx) {
    float norm_sq = 0.0f;
    float* vec = projection_vectors.data() + static_cast<std::size_t>(pidx) * n_features;
    for (int c = 0; c < n_features; ++c) {
      vec[c] = normal(rng);
      norm_sq += vec[c] * vec[c];
    }
    const float inv_norm = norm_sq > 0.0f ? 1.0f / std::sqrt(norm_sq) : 1.0f;
    for (int c = 0; c < n_features; ++c) vec[c] *= inv_norm;
  }

  std::vector<std::vector<ProjectionScore>> landmark_scores(
    static_cast<std::size_t>(n_projections)
  );
  std::vector<float> query_scores(
    static_cast<std::size_t>(n_projections) * n_queries,
    0.0f
  );

  auto score_projection_range = [&](const int projection_start, const int projection_end) {
    for (int pidx = projection_start; pidx < projection_end; ++pidx) {
      const float* vec = projection_vectors.data() + static_cast<std::size_t>(pidx) * n_features;
      std::vector<ProjectionScore>& scores = landmark_scores[static_cast<std::size_t>(pidx)];
      scores.resize(static_cast<std::size_t>(n_landmarks));
      for (int i = 0; i < n_landmarks; ++i) {
        const float* row = landmark_data.data() + static_cast<std::size_t>(i) * n_features;
        float score = 0.0f;
        for (int c = 0; c < n_features; ++c) score += row[c] * vec[c];
        scores[static_cast<std::size_t>(i)] = ProjectionScore{score, i};
      }
      std::sort(scores.begin(), scores.end(), projection_score_less);

      float* query_score = query_scores.data() + static_cast<std::size_t>(pidx) * n_queries;
      for (int q = 0; q < n_queries; ++q) {
        const float* row = query_data.data() + static_cast<std::size_t>(q) * n_features;
        float score = 0.0f;
        for (int c = 0; c < n_features; ++c) score += row[c] * vec[c];
        query_score[q] = score;
      }
    }
  };

  const double score_work = static_cast<double>(n_projections) *
    static_cast<double>(n_landmarks + n_queries) *
    static_cast<double>(n_features);
  const int score_threads = score_work >= 10000000.0
    ? std::max(1, std::min(n_threads, n_projections))
    : 1;
  if (score_threads == 1) {
    score_projection_range(0, n_projections);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(score_threads));
    for (int t = 0; t < score_threads; ++t) {
      const int start = (n_projections * t) / score_threads;
      const int end = (n_projections * (t + 1)) / score_threads;
      workers.emplace_back(score_projection_range, start, end);
    }
    for (auto& worker : workers) worker.join();
  }

  IntegerMatrix indices(n_queries, k);
  NumericMatrix distances(n_queries, k);
  int* indices_ptr = indices.begin();
  double* distances_ptr = distances.begin();
  const double visited_stamp_mb =
    static_cast<double>(n_landmarks) * static_cast<double>(sizeof(std::uint16_t)) /
    (1024.0 * 1024.0);

  auto write_rows = [&](const int row_start, const int row_end) {
    std::vector<std::uint16_t> seen(static_cast<std::size_t>(n_landmarks), 0);
    std::uint16_t stamp = 0;
    std::vector<NeighborF> top;
    top.reserve(static_cast<std::size_t>(k));

    for (int q = row_start; q < row_end; ++q) {
      top.clear();
      if (stamp == std::numeric_limits<std::uint16_t>::max()) {
        std::fill(seen.begin(), seen.end(), 0);
        stamp = 1;
      } else {
        ++stamp;
      }

      for (int pidx = 0; pidx < n_projections; ++pidx) {
        const std::vector<ProjectionScore>& scores =
          landmark_scores[static_cast<std::size_t>(pidx)];
        const float score = query_scores[static_cast<std::size_t>(pidx) * n_queries + q];
        const auto it = std::lower_bound(
          scores.begin(),
          scores.end(),
          score,
          [](const ProjectionScore& item, const float value) {
            return item.score < value;
          }
        );
        const int pos = static_cast<int>(it - scores.begin());
        const int lo = std::max(0, pos - window);
        const int hi = std::min(n_landmarks - 1, pos + window);
        for (int s = lo; s <= hi; ++s) {
          const int candidate = scores[static_cast<std::size_t>(s)].index;
          if (seen[static_cast<std::size_t>(candidate)] == stamp) continue;
          seen[static_cast<std::size_t>(candidate)] = stamp;
          add_projection_candidate_topk(
            landmark_data.data(),
            query_data.data(),
            q,
            candidate,
            n_features,
            k,
            top
          );
        }
      }

      if (static_cast<int>(top.size()) < k) {
        for (int candidate = 0; candidate < n_landmarks; ++candidate) {
          if (seen[static_cast<std::size_t>(candidate)] == stamp) continue;
          seen[static_cast<std::size_t>(candidate)] = stamp;
          add_projection_candidate_topk(
            landmark_data.data(),
            query_data.data(),
            q,
            candidate,
            n_features,
            k,
            top
          );
        }
      }

      if (static_cast<int>(top.size()) == k) {
        std::sort_heap(top.begin(), top.end(), neighborf_less);
      } else {
        std::sort(top.begin(), top.end(), neighborf_less);
      }
      for (int j = 0; j < k; ++j) {
        const NeighborF& item = top[static_cast<std::size_t>(j)];
        indices_ptr[static_cast<std::size_t>(j) * n_queries + q] = item.index + 1;
        distances_ptr[static_cast<std::size_t>(j) * n_queries + q] =
          std::sqrt(std::max(static_cast<double>(item.distance), 0.0));
      }
    }
  };

  if (n_threads == 1) {
    write_rows(0, n_queries);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
      const int start = (n_queries * t) / n_threads;
      const int end = (n_queries * (t + 1)) / n_threads;
      workers.emplace_back(write_rows, start, end);
    }
    for (auto& worker : workers) worker.join();
  }

  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("n_projections") = n_projections,
    Rcpp::Named("window") = window,
    Rcpp::Named("n_threads") = n_threads,
    Rcpp::Named("score_threads") = score_threads,
    Rcpp::Named("visited_stamp_mb_per_thread") = visited_stamp_mb
  );
}

// [[Rcpp::export]]
List nndescent_self_knn_cpp(NumericMatrix data,
                            int k,
                            int pool_size,
                            int n_iters,
                            int max_candidates,
                            int n_random_projections,
                            int seed,
                            bool parallel,
                            int cores) {
  const int n = data.nrow();
  const int n_features = data.ncol();
  if (n < 2) Rcpp::stop("data must have at least two rows");
  if (k < 1 || k >= n) Rcpp::stop("k must be in [1, nrow(data) - 1]");
  pool_size = std::max(k, std::min(pool_size, n - 1));
  n_iters = std::max(0, n_iters);
  max_candidates = std::max(pool_size, std::min(max_candidates, n - 1));
  n_random_projections = std::max(1, n_random_projections);

  std::vector<float> data_row_major;
  copy_row_major_float(data.begin(), data_row_major, n, n_features);
  const float* x = data_row_major.data();

  std::vector<int> graph_indices(static_cast<std::size_t>(n) * pool_size, -1);
  std::vector<float> graph_distances(
    static_cast<std::size_t>(n) * pool_size,
    std::numeric_limits<float>::infinity()
  );
  std::vector<std::vector<NeighborF>> init_top(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    init_top[static_cast<std::size_t>(i)].reserve(pool_size);
  }

  std::mt19937 rng(static_cast<unsigned int>(seed));
  std::normal_distribution<float> normal(0.0f, 1.0f);
  const int projection_window = std::max(4, std::min(32, pool_size / 2));
  std::vector<float> direction(static_cast<std::size_t>(n_features));
  std::vector<std::pair<float, int>> scores(static_cast<std::size_t>(n));

  for (int proj = 0; proj < n_random_projections; ++proj) {
    for (int c = 0; c < n_features; ++c) direction[static_cast<std::size_t>(c)] = normal(rng);
    for (int i = 0; i < n; ++i) {
      const float* row = x + static_cast<std::size_t>(i) * n_features;
      float score = 0.0f;
      for (int c = 0; c < n_features; ++c) score += row[c] * direction[static_cast<std::size_t>(c)];
      scores[static_cast<std::size_t>(i)] = std::make_pair(score, i);
    }
    std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
      if (a.first == b.first) return a.second < b.second;
      return a.first < b.first;
    });
    for (int pos = 0; pos < n; ++pos) {
      const int query = scores[static_cast<std::size_t>(pos)].second;
      const int lo = std::max(0, pos - projection_window);
      const int hi = std::min(n - 1, pos + projection_window);
      std::vector<NeighborF>& top = init_top[static_cast<std::size_t>(query)];
      for (int other_pos = lo; other_pos <= hi; ++other_pos) {
        if (other_pos == pos) continue;
        const int candidate = scores[static_cast<std::size_t>(other_pos)].second;
        insert_sorted_top_float(x, query, candidate, n_features, pool_size, top);
      }
    }
  }

  for (int i = 0; i < n; ++i) {
    std::mt19937 row_rng(static_cast<unsigned int>(seed + 104729 * (i + 1)));
    std::uniform_int_distribution<int> uniform(0, n - 1);
    std::vector<NeighborF>& top = init_top[static_cast<std::size_t>(i)];
    int attempts = 0;
    while (static_cast<int>(top.size()) < pool_size && attempts < pool_size * 32) {
      insert_sorted_top_float(x, i, uniform(row_rng), n_features, pool_size, top);
      ++attempts;
    }
    for (int candidate = 0; static_cast<int>(top.size()) < pool_size && candidate < n; ++candidate) {
      insert_sorted_top_float(x, i, candidate, n_features, pool_size, top);
    }
    write_sorted_top_to_graph(top, graph_indices, graph_distances, i, pool_size);
  }
  init_top.clear();
  init_top.shrink_to_fit();

  const int n_threads = requested_threads(parallel, cores, n);
  const int reverse_limit = std::max(pool_size, std::min(max_candidates, pool_size * 2));

  for (int iter = 0; iter < n_iters; ++iter) {
    const std::vector<int> old_indices = graph_indices;
    const std::vector<float> old_distances = graph_distances;
    std::vector<std::vector<int>> reverse(static_cast<std::size_t>(n));
    for (int rank = 0; rank < pool_size; ++rank) {
      for (int i = 0; i < n; ++i) {
        const std::size_t base = static_cast<std::size_t>(i) * pool_size;
        const int nb = old_indices[base + rank];
        if (nb < 0 || nb >= n) continue;
        std::vector<int>& rev = reverse[static_cast<std::size_t>(nb)];
        if (static_cast<int>(rev.size()) < reverse_limit) rev.push_back(i);
      }
    }

    auto refine_rows = [&](const int row_start, const int row_end) {
      std::vector<int> seen(static_cast<std::size_t>(n), 0);
      std::vector<NeighborF> top;
      top.reserve(pool_size);
      int stamp = 1;

      auto consider = [&](const int query,
                          const int candidate,
                          int& candidate_count) {
        if (candidate < 0 || candidate >= n || candidate == query) return;
        if (candidate_count >= max_candidates) return;
        if (seen[static_cast<std::size_t>(candidate)] == stamp) return;
        seen[static_cast<std::size_t>(candidate)] = stamp;
        ++candidate_count;
        insert_sorted_top_float(x, query, candidate, n_features, pool_size, top);
      };

      for (int q = row_start; q < row_end; ++q) {
        if (stamp == std::numeric_limits<int>::max()) {
          std::fill(seen.begin(), seen.end(), 0);
          stamp = 1;
        }
        ++stamp;
        top.clear();
        seen[static_cast<std::size_t>(q)] = stamp;
        int candidate_count = 0;

        const std::size_t row_base = static_cast<std::size_t>(q) * pool_size;
        for (int j = 0; j < pool_size; ++j) {
          const int nb = old_indices[row_base + j];
          if (nb < 0) continue;
          seen[static_cast<std::size_t>(nb)] = stamp;
          top.push_back(NeighborF{old_distances[row_base + j], nb});
        }
        std::sort(top.begin(), top.end(), neighborf_less);

        for (int j = 0; j < pool_size && candidate_count < max_candidates; ++j) {
          const int nb = old_indices[row_base + j];
          if (nb < 0 || nb >= n) continue;
          const std::size_t nb_base = static_cast<std::size_t>(nb) * pool_size;
          for (int t = 0; t < pool_size && candidate_count < max_candidates; ++t) {
            consider(q, old_indices[nb_base + t], candidate_count);
          }
        }

        const std::vector<int>& rev = reverse[static_cast<std::size_t>(q)];
        for (const int nb : rev) {
          if (candidate_count >= max_candidates) break;
          consider(q, nb, candidate_count);
          const std::size_t nb_base = static_cast<std::size_t>(nb) * pool_size;
          for (int t = 0; t < pool_size && candidate_count < max_candidates; ++t) {
            consider(q, old_indices[nb_base + t], candidate_count);
          }
        }

        write_sorted_top_to_graph(top, graph_indices, graph_distances, q, pool_size);
      }
    };

    if (n_threads == 1) {
      refine_rows(0, n);
    } else {
      std::vector<std::thread> workers;
      workers.reserve(static_cast<std::size_t>(n_threads));
      for (int t = 0; t < n_threads; ++t) {
        const int start = (n * t) / n_threads;
        const int end = (n * (t + 1)) / n_threads;
        workers.emplace_back(refine_rows, start, end);
      }
      for (auto& worker : workers) worker.join();
    }
  }

  IntegerMatrix indices(n, k);
  NumericMatrix distances(n, k);
  int* indices_ptr = indices.begin();
  double* distances_ptr = distances.begin();
  for (int i = 0; i < n; ++i) {
    const std::size_t base = static_cast<std::size_t>(i) * pool_size;
    for (int j = 0; j < k; ++j) {
      const int idx = graph_indices[base + j];
      indices_ptr[static_cast<std::size_t>(j) * n + i] = idx + 1;
      distances_ptr[static_cast<std::size_t>(j) * n + i] =
        std::sqrt(std::max(static_cast<double>(graph_distances[base + j]), 0.0));
    }
  }

  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances
  );
}

// [[Rcpp::export]]
List ivf_self_knn_cpp(NumericMatrix data,
                      int k,
                      int nlist,
                      int nprobe,
                      int seed,
                      bool parallel,
                      int cores) {
  const int n = data.nrow();
  const int n_features = data.ncol();
  if (n < 2) Rcpp::stop("data must have at least two rows");
  if (k < 1 || k >= n) Rcpp::stop("k must be in [1, nrow(data) - 1]");
  nlist = std::max(1, std::min(nlist, n));
  nprobe = std::max(1, std::min(nprobe, nlist));

  std::vector<float> data_row_major;
  copy_row_major_float(data.begin(), data_row_major, n, n_features);
  const float* x = data_row_major.data();

  std::vector<int> centers(static_cast<std::size_t>(n));
  std::iota(centers.begin(), centers.end(), 0);
  std::mt19937 rng(static_cast<unsigned int>(seed));
  std::shuffle(centers.begin(), centers.end(), rng);
  centers.resize(static_cast<std::size_t>(nlist));
  std::sort(centers.begin(), centers.end());

  const int n_threads = requested_threads(parallel, cores, n);
  std::vector<int> assignment(static_cast<std::size_t>(n), 0);

  auto assign_rows = [&](const int row_start, const int row_end) {
    for (int i = row_start; i < row_end; ++i) {
      int best_slot = 0;
      float best_dist = squared_euclidean_row_major_float(
        x, i, centers[0], n_features
      );
      for (int c = 1; c < nlist; ++c) {
        const float dist = squared_euclidean_row_major_float(
          x, i, centers[static_cast<std::size_t>(c)], n_features
        );
        if (dist < best_dist ||
            (dist == best_dist && centers[static_cast<std::size_t>(c)] < centers[static_cast<std::size_t>(best_slot)])) {
          best_dist = dist;
          best_slot = c;
        }
      }
      assignment[static_cast<std::size_t>(i)] = best_slot;
    }
  };

  if (n_threads == 1) {
    assign_rows(0, n);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
      const int start = (n * t) / n_threads;
      const int end = (n * (t + 1)) / n_threads;
      workers.emplace_back(assign_rows, start, end);
    }
    for (auto& worker : workers) worker.join();
  }

  std::vector<std::vector<int>> buckets(static_cast<std::size_t>(nlist));
  for (int i = 0; i < n; ++i) {
    buckets[static_cast<std::size_t>(assignment[static_cast<std::size_t>(i)])].push_back(i);
  }

  IntegerMatrix indices(n, k);
  NumericMatrix distances(n, k);
  int* indices_ptr = indices.begin();
  double* distances_ptr = distances.begin();

  auto query_rows = [&](const int row_start, const int row_end) {
    std::vector<int> seen(static_cast<std::size_t>(n), 0);
    int stamp = 1;
    std::vector<NeighborF> probes;
    probes.reserve(static_cast<std::size_t>(nprobe));
    std::vector<NeighborF> top;
    top.reserve(static_cast<std::size_t>(k));

    for (int q = row_start; q < row_end; ++q) {
      if (stamp == std::numeric_limits<int>::max()) {
        std::fill(seen.begin(), seen.end(), 0);
        stamp = 1;
      }
      ++stamp;
      seen[static_cast<std::size_t>(q)] = stamp;
      probes.clear();
      top.clear();

      for (int c = 0; c < nlist; ++c) {
        add_probe_center(x, centers, q, c, n_features, nprobe, probes);
      }
      if (static_cast<int>(probes.size()) == nprobe) {
        std::sort_heap(probes.begin(), probes.end(), neighborf_less);
      } else {
        std::sort(probes.begin(), probes.end(), neighborf_less);
      }

      for (const NeighborF& probe : probes) {
        const int slot = probe.index;
        const std::vector<int>& bucket = buckets[static_cast<std::size_t>(slot)];
        for (const int candidate : bucket) {
          if (seen[static_cast<std::size_t>(candidate)] == stamp) continue;
          seen[static_cast<std::size_t>(candidate)] = stamp;
          insert_sorted_top_float(x, q, candidate, n_features, k, top);
        }
      }

      if (static_cast<int>(top.size()) < k) {
        for (int candidate = 0; candidate < n; ++candidate) {
          if (seen[static_cast<std::size_t>(candidate)] == stamp) continue;
          seen[static_cast<std::size_t>(candidate)] = stamp;
          insert_sorted_top_float(x, q, candidate, n_features, k, top);
          if (static_cast<int>(top.size()) >= k) {
            const float worst = top.back().distance;
            if (std::isfinite(worst) && candidate > q + k * 16) {
              break;
            }
          }
        }
      }
      if (static_cast<int>(top.size()) < k) {
        for (int candidate = 0; candidate < n; ++candidate) {
          insert_sorted_top_float(x, q, candidate, n_features, k, top);
          if (static_cast<int>(top.size()) == k) break;
        }
      }

      for (int j = 0; j < k; ++j) {
        const NeighborF& item = top[static_cast<std::size_t>(j)];
        indices_ptr[static_cast<std::size_t>(j) * n + q] = item.index + 1;
        distances_ptr[static_cast<std::size_t>(j) * n + q] =
          std::sqrt(std::max(static_cast<double>(item.distance), 0.0));
      }
    }
  };

  if (n_threads == 1) {
    query_rows(0, n);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
      const int start = (n * t) / n_threads;
      const int end = (n * (t + 1)) / n_threads;
      workers.emplace_back(query_rows, start, end);
    }
    for (auto& worker : workers) worker.join();
  }

  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("nlist") = nlist,
    Rcpp::Named("nprobe") = nprobe
  );
}

// [[Rcpp::export]]
List annoy_self_knn_cpp(NumericMatrix data,
                        int k,
                        int n_trees,
                        int leaf_size,
                        int search_k,
                        int seed,
                        bool parallel,
                        int cores) {
  const int n = data.nrow();
  const int n_features = data.ncol();
  if (n < 2) Rcpp::stop("data must have at least two rows");
  if (k < 1 || k >= n) Rcpp::stop("k must be in [1, nrow(data) - 1]");
  n_trees = std::max(1, n_trees);
  leaf_size = std::max(k + 1, std::min(std::max(leaf_size, 2), n));
  search_k = search_k <= 0 ? n_trees * leaf_size : std::max(k, search_k);

  std::vector<float> data_row_major;
  copy_row_major_float(data.begin(), data_row_major, n, n_features);
  const float* x = data_row_major.data();

  std::vector<std::vector<int>> leaves;
  leaves.reserve(static_cast<std::size_t>(n_trees) * std::max(1, n / leaf_size));
  std::vector<int> row_leaf(static_cast<std::size_t>(n_trees) * n, -1);
  std::vector<int> base_items(static_cast<std::size_t>(n));
  std::iota(base_items.begin(), base_items.end(), 0);
  std::normal_distribution<float> normal(0.0f, 1.0f);

  for (int tree = 0; tree < n_trees; ++tree) {
    std::vector<int> items = base_items;
    std::mt19937 rng(static_cast<unsigned int>(seed + 104729 * (tree + 1)));
    std::vector<int> one_tree_leaf(static_cast<std::size_t>(n), -1);
    const int leaf_base = static_cast<int>(leaves.size());
    build_annoy_leaves_recursive(
      x, n_features, leaf_size, items, leaves, one_tree_leaf, rng, normal
    );
    for (int i = 0; i < n; ++i) {
      const int leaf_id = one_tree_leaf[static_cast<std::size_t>(i)];
      row_leaf[static_cast<std::size_t>(tree) * n + i] =
        leaf_id < 0 ? -1 : leaf_id;
    }
    if (static_cast<int>(leaves.size()) == leaf_base) {
      Rcpp::stop("failed to build Annoy-style tree");
    }
  }

  IntegerMatrix indices(n, k);
  NumericMatrix distances(n, k);
  int* indices_ptr = indices.begin();
  double* distances_ptr = distances.begin();
  const int n_threads = requested_threads(parallel, cores, n);

  auto query_rows = [&](const int row_start, const int row_end) {
    std::vector<int> seen(static_cast<std::size_t>(n), 0);
    int stamp = 1;
    std::vector<NeighborF> top;
    top.reserve(static_cast<std::size_t>(k));

    for (int q = row_start; q < row_end; ++q) {
      if (stamp == std::numeric_limits<int>::max()) {
        std::fill(seen.begin(), seen.end(), 0);
        stamp = 1;
      }
      ++stamp;
      seen[static_cast<std::size_t>(q)] = stamp;
      top.clear();
      int inspected = 0;

      for (int tree = 0; tree < n_trees && inspected < search_k; ++tree) {
        const int leaf_id = row_leaf[static_cast<std::size_t>(tree) * n + q];
        if (leaf_id < 0 || leaf_id >= static_cast<int>(leaves.size())) continue;
        const std::vector<int>& leaf = leaves[static_cast<std::size_t>(leaf_id)];
        for (const int candidate : leaf) {
          if (inspected >= search_k) break;
          if (candidate == q) continue;
          if (seen[static_cast<std::size_t>(candidate)] == stamp) continue;
          seen[static_cast<std::size_t>(candidate)] = stamp;
          ++inspected;
          insert_sorted_top_float(x, q, candidate, n_features, k, top);
        }
      }

      if (static_cast<int>(top.size()) < k) {
        std::mt19937 row_rng(static_cast<unsigned int>(seed + 4099 * (q + 1)));
        std::uniform_int_distribution<int> uniform(0, n - 1);
        int attempts = 0;
        while (static_cast<int>(top.size()) < k && attempts < search_k * 4) {
          const int candidate = uniform(row_rng);
          if (candidate != q && seen[static_cast<std::size_t>(candidate)] != stamp) {
            seen[static_cast<std::size_t>(candidate)] = stamp;
            insert_sorted_top_float(x, q, candidate, n_features, k, top);
          }
          ++attempts;
        }
      }
      for (int candidate = 0; static_cast<int>(top.size()) < k && candidate < n; ++candidate) {
        insert_sorted_top_float(x, q, candidate, n_features, k, top);
      }

      for (int j = 0; j < k; ++j) {
        const NeighborF& item = top[static_cast<std::size_t>(j)];
        indices_ptr[static_cast<std::size_t>(j) * n + q] = item.index + 1;
        distances_ptr[static_cast<std::size_t>(j) * n + q] =
          std::sqrt(std::max(static_cast<double>(item.distance), 0.0));
      }
    }
  };

  if (n_threads == 1) {
    query_rows(0, n);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
      const int start = (n * t) / n_threads;
      const int end = (n * (t + 1)) / n_threads;
      workers.emplace_back(query_rows, start, end);
    }
    for (auto& worker : workers) worker.join();
  }

  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("n_trees") = n_trees,
    Rcpp::Named("leaf_size") = leaf_size,
    Rcpp::Named("search_k") = search_k,
    Rcpp::Named("n_leaves") = static_cast<int>(leaves.size())
  );
}

// [[Rcpp::export]]
List vptree_self_knn_cpp(NumericMatrix data,
                         int k,
                         bool parallel,
                         int cores) {
  const int n = data.nrow();
  const int n_features = data.ncol();
  if (n < 2) Rcpp::stop("data must have at least two rows");
  if (k < 1 || k >= n) Rcpp::stop("k must be in [1, nrow(data) - 1]");

  std::vector<float> data_row_major;
  copy_row_major_float(data.begin(), data_row_major, n, n_features);
  const float* x = data_row_major.data();

  std::vector<int> items(static_cast<std::size_t>(n));
  std::iota(items.begin(), items.end(), 0);
  std::vector<VpNode> nodes;
  nodes.reserve(static_cast<std::size_t>(n));
  const int root = build_vptree_recursive(x, n_features, items, nodes);

  IntegerMatrix indices(n, k);
  NumericMatrix distances(n, k);
  int* indices_ptr = indices.begin();
  double* distances_ptr = distances.begin();
  const int n_threads = requested_threads(parallel, cores, n);

  auto query_rows = [&](const int row_start, const int row_end) {
    std::vector<Neighbor> top;
    top.reserve(static_cast<std::size_t>(k));
    for (int q = row_start; q < row_end; ++q) {
      top.clear();
      search_vptree(x, n_features, nodes, root, q, k, top);
      if (static_cast<int>(top.size()) < k) {
        for (int candidate = 0; candidate < n; ++candidate) {
          if (candidate == q) continue;
          const double dist = static_cast<double>(
            euclidean_row_major_float(x, q, candidate, n_features)
          );
          insert_heap_top_double(candidate, dist, k, top);
        }
      }
      write_heap_top_double(top, indices_ptr, distances_ptr, q, n, k);
    }
  };

  if (n_threads == 1) {
    query_rows(0, n);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
      const int start = (n * t) / n_threads;
      const int end = (n * (t + 1)) / n_threads;
      workers.emplace_back(query_rows, start, end);
    }
    for (auto& worker : workers) worker.join();
  }

  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("nodes") = static_cast<int>(nodes.size())
  );
}

// [[Rcpp::export]]
List vptree_query_knn_cpp(NumericMatrix data,
                          NumericMatrix points,
                          int k,
                          bool parallel,
                          int cores) {
  const int n = data.nrow();
  const int n_points = points.nrow();
  const int n_features = data.ncol();
  if (points.ncol() != n_features) {
    Rcpp::stop("data and points must have the same number of columns");
  }
  if (n < 1) Rcpp::stop("data must have at least one row");
  if (n_points < 1) Rcpp::stop("points must have at least one row");
  if (k < 1 || k > n) Rcpp::stop("k must be in [1, nrow(data)]");

  std::vector<float> data_row_major;
  std::vector<float> point_row_major;
  copy_row_major_float(data.begin(), data_row_major, n, n_features);
  copy_row_major_float(points.begin(), point_row_major, n_points, n_features);
  const float* x = data_row_major.data();
  const float* qx = point_row_major.data();

  std::vector<int> items(static_cast<std::size_t>(n));
  std::iota(items.begin(), items.end(), 0);
  std::vector<VpNode> nodes;
  nodes.reserve(static_cast<std::size_t>(n));
  const int root = build_vptree_recursive(x, n_features, items, nodes);

  IntegerMatrix indices(n_points, k);
  NumericMatrix distances(n_points, k);
  int* indices_ptr = indices.begin();
  double* distances_ptr = distances.begin();
  const int n_threads = requested_threads(parallel, cores, n_points);

  auto query_rows = [&](const int row_start, const int row_end) {
    std::vector<Neighbor> top;
    top.reserve(static_cast<std::size_t>(k));
    for (int q = row_start; q < row_end; ++q) {
      top.clear();
      const float* query = qx + static_cast<std::size_t>(q) * n_features;
      search_vptree_query(x, n_features, nodes, root, query, k, top);
      if (static_cast<int>(top.size()) < k) {
        for (int candidate = 0; candidate < n; ++candidate) {
          const double dist = static_cast<double>(
            euclidean_query_to_row_major_float(x, query, candidate, n_features)
          );
          insert_heap_top_double(candidate, dist, k, top);
        }
      }
      write_heap_top_double(top, indices_ptr, distances_ptr, q, n_points, k);
    }
  };

  if (n_threads == 1) {
    query_rows(0, n_points);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
      const int start = (n_points * t) / n_threads;
      const int end = (n_points * (t + 1)) / n_threads;
      workers.emplace_back(query_rows, start, end);
    }
    for (auto& worker : workers) worker.join();
  }

  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("nodes") = static_cast<int>(nodes.size())
  );
}

// [[Rcpp::export]]
List grid2d_self_knn_cpp(NumericMatrix data,
                         int k,
                         bool parallel,
                         int cores,
                         int bins_per_dim) {
  const int n = data.nrow();
  const int n_features = data.ncol();
  if (n_features != 2) Rcpp::stop("grid2d_self_knn_cpp requires exactly two columns");
  if (n < 2) Rcpp::stop("data must have at least two rows");
  if (k < 1 || k >= n) Rcpp::stop("k must be in [1, nrow(data) - 1]");
  if (bins_per_dim < 1) Rcpp::stop("bins_per_dim must be positive");

  std::vector<double> x(static_cast<std::size_t>(n));
  std::vector<double> y(static_cast<std::size_t>(n));
  const double* col0 = data.begin();
  const double* col1 = data.begin() + static_cast<std::size_t>(n);
  for (int i = 0; i < n; ++i) {
    x[static_cast<std::size_t>(i)] = col0[i];
    y[static_cast<std::size_t>(i)] = col1[i];
  }

  Grid2DIndex grid = build_grid2d_index(x, y, bins_per_dim);
  IntegerMatrix indices(n, k);
  NumericMatrix distances(n, k);
  int* indices_ptr = indices.begin();
  double* distances_ptr = distances.begin();
  const int n_threads = requested_threads(parallel, cores, n);

  auto query_rows = [&](const int row_start, const int row_end) {
    std::vector<Neighbor> top;
    top.reserve(static_cast<std::size_t>(k));
    for (int q = row_start; q < row_end; ++q) {
      search_grid2d_exact(x, y, grid, q, k, top);
      if (static_cast<int>(top.size()) < k) {
        for (int candidate = 0; candidate < n; ++candidate) {
          if (candidate == q) continue;
          const double dx = x[static_cast<std::size_t>(q)] - x[static_cast<std::size_t>(candidate)];
          const double dy = y[static_cast<std::size_t>(q)] - y[static_cast<std::size_t>(candidate)];
          insert_heap_top_double(candidate, dx * dx + dy * dy, k, top);
        }
      }
      if (static_cast<int>(top.size()) == k) {
        std::sort_heap(top.begin(), top.end(), neighbor_less);
      } else {
        std::sort(top.begin(), top.end(), neighbor_less);
      }
      for (int j = 0; j < k; ++j) {
        indices_ptr[static_cast<std::size_t>(j) * n + q] =
          top[static_cast<std::size_t>(j)].index + 1;
        distances_ptr[static_cast<std::size_t>(j) * n + q] =
          std::sqrt(std::max(top[static_cast<std::size_t>(j)].distance, 0.0));
      }
    }
  };

  if (n_threads == 1) {
    query_rows(0, n);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
      const int start = (n * t) / n_threads;
      const int end = (n * (t + 1)) / n_threads;
      workers.emplace_back(query_rows, start, end);
    }
    for (auto& worker : workers) worker.join();
  }

  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("bins_per_dim") = bins_per_dim,
    Rcpp::Named("n_cells") = grid.bins_x * grid.bins_y,
    Rcpp::Named("n_threads") = n_threads
  );
}

// [[Rcpp::export]]
List grid3d_self_knn_cpp(NumericMatrix data,
                         int k,
                         bool parallel,
                         int cores,
                         int bins_per_dim) {
  const int n = data.nrow();
  const int n_features = data.ncol();
  if (n_features != 3) Rcpp::stop("grid3d_self_knn_cpp requires exactly three columns");
  if (n < 2) Rcpp::stop("data must have at least two rows");
  if (k < 1 || k >= n) Rcpp::stop("k must be in [1, nrow(data) - 1]");
  if (bins_per_dim < 1) Rcpp::stop("bins_per_dim must be positive");

  std::vector<double> x(static_cast<std::size_t>(n));
  std::vector<double> y(static_cast<std::size_t>(n));
  std::vector<double> z(static_cast<std::size_t>(n));
  const double* col0 = data.begin();
  const double* col1 = data.begin() + static_cast<std::size_t>(n);
  const double* col2 = data.begin() + static_cast<std::size_t>(2) * n;
  for (int i = 0; i < n; ++i) {
    x[static_cast<std::size_t>(i)] = col0[i];
    y[static_cast<std::size_t>(i)] = col1[i];
    z[static_cast<std::size_t>(i)] = col2[i];
  }

  Grid3DIndex grid = build_grid3d_index(x, y, z, bins_per_dim);
  IntegerMatrix indices(n, k);
  NumericMatrix distances(n, k);
  int* indices_ptr = indices.begin();
  double* distances_ptr = distances.begin();
  const int n_threads = requested_threads(parallel, cores, n);

  auto query_rows = [&](const int row_start, const int row_end) {
    std::vector<Neighbor> top;
    top.reserve(static_cast<std::size_t>(k));
    for (int q = row_start; q < row_end; ++q) {
      search_grid3d_exact(x, y, z, grid, q, k, top);
      if (static_cast<int>(top.size()) < k) {
        for (int candidate = 0; candidate < n; ++candidate) {
          if (candidate == q) continue;
          const double dx = x[static_cast<std::size_t>(q)] - x[static_cast<std::size_t>(candidate)];
          const double dy = y[static_cast<std::size_t>(q)] - y[static_cast<std::size_t>(candidate)];
          const double dz = z[static_cast<std::size_t>(q)] - z[static_cast<std::size_t>(candidate)];
          insert_heap_top_double(candidate, dx * dx + dy * dy + dz * dz, k, top);
        }
      }
      if (static_cast<int>(top.size()) == k) {
        std::sort_heap(top.begin(), top.end(), neighbor_less);
      } else {
        std::sort(top.begin(), top.end(), neighbor_less);
      }
      for (int j = 0; j < k; ++j) {
        indices_ptr[static_cast<std::size_t>(j) * n + q] =
          top[static_cast<std::size_t>(j)].index + 1;
        distances_ptr[static_cast<std::size_t>(j) * n + q] =
          std::sqrt(std::max(top[static_cast<std::size_t>(j)].distance, 0.0));
      }
    }
  };

  if (n_threads == 1) {
    query_rows(0, n);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
      const int start = (n * t) / n_threads;
      const int end = (n * (t + 1)) / n_threads;
      workers.emplace_back(query_rows, start, end);
    }
    for (auto& worker : workers) worker.join();
  }

  return List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("bins_per_dim") = bins_per_dim,
    Rcpp::Named("n_cells") = grid.bins_x * grid.bins_y * grid.bins_z,
    Rcpp::Named("n_threads") = n_threads
  );
}
