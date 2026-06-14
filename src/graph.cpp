#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

using Rcpp::IntegerMatrix;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;
using Rcpp::IntegerVector;

namespace {

std::uint64_t edge_key(const int a, const int b) {
  const std::uint32_t u = static_cast<std::uint32_t>(std::min(a, b));
  const std::uint32_t v = static_cast<std::uint32_t>(std::max(a, b));
  return (static_cast<std::uint64_t>(u) << 32) | static_cast<std::uint64_t>(v);
}

int edge_from_key(const std::uint64_t key) {
  return static_cast<int>(static_cast<std::uint32_t>(key >> 32));
}

int edge_to_key(const std::uint64_t key) {
  return static_cast<int>(static_cast<std::uint32_t>(key & 0xffffffffULL));
}

struct Edge {
  std::uint64_t key;
  double weight;
};

void push_edge(std::vector<Edge>& edges,
               const int i,
               const int j,
               const double weight) {
  if (i == j || !std::isfinite(weight) || weight <= 0.0) return;
  edges.push_back(Edge{edge_key(i, j), weight});
}

bool contains_neighbor(const IntegerMatrix& indices,
                       const int row,
                       const int target) {
  const int k = indices.ncol();
  for (int col = 0; col < k; ++col) {
    if (indices(row, col) == target) return true;
  }
  return false;
}

std::vector<Edge> build_full_snn_edges(const IntegerMatrix& indices,
                                       const double prune) {
  // Inspired by bluster::neighborsToSNNGraph() / scran_graph_cluster:
  // build an inverted neighbour index and count shared-neighbour
  // co-occurrences directly. This creates the standard full SNN graph
  // between all observations sharing at least one neighbour, not only
  // edges already present in the directed KNN graph.
  const int n = indices.nrow();
  const int k = indices.ncol();

  std::vector<int> valid_count(static_cast<std::size_t>(n), 0);
  std::vector<int> reverse_count(static_cast<std::size_t>(n) + 1U, 0);
  for (int row = 0; row < n; ++row) {
    const int self = row + 1;
    for (int col = 0; col < k; ++col) {
      const int idx = indices(row, col);
      if (idx >= 1 && idx <= n && idx != self) {
        ++valid_count[static_cast<std::size_t>(row)];
        ++reverse_count[static_cast<std::size_t>(idx)];
      }
    }
  }

  std::vector<int> reverse_ptr(static_cast<std::size_t>(n) + 2U, 0);
  for (int i = 1; i <= n; ++i) {
    reverse_ptr[static_cast<std::size_t>(i + 1)] =
      reverse_ptr[static_cast<std::size_t>(i)] +
      reverse_count[static_cast<std::size_t>(i)];
  }
  std::vector<int> reverse_rows(static_cast<std::size_t>(reverse_ptr[static_cast<std::size_t>(n + 1)]));
  std::fill(reverse_count.begin(), reverse_count.end(), 0);
  for (int row = 0; row < n; ++row) {
    const int self = row + 1;
    for (int col = 0; col < k; ++col) {
      const int idx = indices(row, col);
      if (idx >= 1 && idx <= n && idx != self) {
        const std::size_t offset =
          static_cast<std::size_t>(reverse_ptr[static_cast<std::size_t>(idx)] +
                                   reverse_count[static_cast<std::size_t>(idx)]++);
        reverse_rows[offset] = row;
      }
    }
  }

  std::vector<Edge> edges;
  edges.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
  std::vector<int> shared_counts(static_cast<std::size_t>(n), 0);
  std::vector<int> touched;
  touched.reserve(static_cast<std::size_t>(k) * static_cast<std::size_t>(k));

  for (int row = 0; row < n; ++row) {
    touched.clear();
    const int self = row + 1;
    for (int col = 0; col < k; ++col) {
      const int idx = indices(row, col);
      if (idx < 1 || idx > n || idx == self) continue;
      const int begin = reverse_ptr[static_cast<std::size_t>(idx)];
      const int end = reverse_ptr[static_cast<std::size_t>(idx + 1)];
      for (int pos = begin; pos < end; ++pos) {
        const int other = reverse_rows[static_cast<std::size_t>(pos)];
        if (other <= row) continue;
        int& counter = shared_counts[static_cast<std::size_t>(other)];
        if (counter == 0) touched.push_back(other);
        ++counter;
      }
    }

    for (const int other : touched) {
      const int shared = shared_counts[static_cast<std::size_t>(other)];
      shared_counts[static_cast<std::size_t>(other)] = 0;
      const int denom_int = valid_count[static_cast<std::size_t>(row)] +
        valid_count[static_cast<std::size_t>(other)] - shared;
      if (shared > 0 && denom_int > 0) {
        const double weight =
          static_cast<double>(shared) / static_cast<double>(denom_int);
        if (weight > prune) push_edge(edges, row + 1, other + 1, weight);
      }
    }
  }

  return edges;
}

std::vector<double> local_sigmas(const NumericMatrix& distances) {
  const int n = distances.nrow();
  const int k = distances.ncol();
  std::vector<double> sigma(static_cast<std::size_t>(n), 1.0);
  for (int row = 0; row < n; ++row) {
    double last = 0.0;
    double sum = 0.0;
    int count = 0;
    for (int col = 0; col < k; ++col) {
      const double d = distances(row, col);
      if (std::isfinite(d) && d > 0.0) {
        last = d;
        sum += d;
        ++count;
      }
    }
    if (last > 0.0) {
      sigma[static_cast<std::size_t>(row)] = last;
    } else if (count > 0 && sum > 0.0) {
      sigma[static_cast<std::size_t>(row)] = sum / static_cast<double>(count);
    }
  }
  return sigma;
}

} // namespace

// [[Rcpp::export]]
List knn_graph_edges_cpp(IntegerMatrix indices,
                         NumericMatrix distances,
                         std::string weight_type,
                         double prune,
                         bool mutual) {
  const int n = indices.nrow();
  const int k = indices.ncol();
  if (distances.nrow() != n || distances.ncol() != k) {
    Rcpp::stop("KNN indices and distances must have the same dimensions");
  }
  if (n < 2 || k < 1) {
    Rcpp::stop("KNN input must have at least two rows and one neighbor column");
  }
  if (weight_type != "snn" &&
      weight_type != "distance" &&
      weight_type != "adaptive" &&
      weight_type != "binary") {
    Rcpp::stop("unsupported graph weight type");
  }
  if (!std::isfinite(prune) || prune < 0.0) prune = 0.0;

  std::vector<Edge> edges;

  if (weight_type == "snn" && !mutual) {
    edges = build_full_snn_edges(indices, prune);
  } else if (weight_type == "snn") {
    edges.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
    std::vector<int> mark(static_cast<std::size_t>(n) + 1U, 0);
    int stamp = 1;
    for (int row = 0; row < n; ++row) {
      if (stamp == std::numeric_limits<int>::max()) {
        std::fill(mark.begin(), mark.end(), 0);
        stamp = 1;
      }
      const int i = row + 1;
      for (int col = 0; col < k; ++col) {
        const int idx = indices(row, col);
        if (idx >= 1 && idx <= n && idx != i) {
          mark[static_cast<std::size_t>(idx)] = stamp;
        }
      }

      for (int col = 0; col < k; ++col) {
        const int j = indices(row, col);
        if (j < 1 || j > n || j == i) continue;
        if (mutual && !contains_neighbor(indices, j - 1, i)) continue;
        int shared = 0;
        const int jrow = j - 1;
        for (int jcol = 0; jcol < k; ++jcol) {
          const int jj = indices(jrow, jcol);
          if (jj >= 1 && jj <= n && mark[static_cast<std::size_t>(jj)] == stamp) {
            ++shared;
          }
        }
        if (shared <= 0) continue;
        const double denom = static_cast<double>(2 * k - shared);
        const double weight = denom > 0.0 ? static_cast<double>(shared) / denom : 0.0;
        if (weight > prune) push_edge(edges, i, j, weight);
      }
      ++stamp;
    }
  } else {
    edges.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
    std::vector<double> sigma;
    if (weight_type == "adaptive") {
      sigma = local_sigmas(distances);
    }
    for (int row = 0; row < n; ++row) {
      const int i = row + 1;
      for (int col = 0; col < k; ++col) {
        const int j = indices(row, col);
        if (j < 1 || j > n || j == i) continue;
        if (mutual && !contains_neighbor(indices, j - 1, i)) continue;
        double weight = 1.0;
        if (weight_type == "distance") {
          const double d = distances(row, col);
          if (!std::isfinite(d) || d < 0.0) continue;
          weight = 1.0 / (1.0 + d);
        } else if (weight_type == "adaptive") {
          const double d = distances(row, col);
          if (!std::isfinite(d) || d < 0.0) continue;
          const double si = sigma[static_cast<std::size_t>(row)];
          const double sj = sigma[static_cast<std::size_t>(j - 1)];
          const double scale = std::max(si * sj, 1e-12);
          weight = std::exp(-(d * d) / scale);
        }
        if (weight > prune) push_edge(edges, i, j, weight);
      }
    }
  }

  std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
    return a.key < b.key;
  });
  std::size_t unique_count = 0;
  for (std::size_t pos = 0; pos < edges.size();) {
    const std::uint64_t key = edges[pos].key;
    double max_weight = edges[pos].weight;
    ++pos;
    while (pos < edges.size() && edges[pos].key == key) {
      if (edges[pos].weight > max_weight) max_weight = edges[pos].weight;
      ++pos;
    }
    edges[unique_count++] = Edge{key, max_weight};
  }

  const int m = static_cast<int>(unique_count);
  IntegerVector from(m);
  IntegerVector to(m);
  NumericVector weight(m);
  for (int pos = 0; pos < m; ++pos) {
    from[pos] = edge_from_key(edges[static_cast<std::size_t>(pos)].key);
    to[pos] = edge_to_key(edges[static_cast<std::size_t>(pos)].key);
    weight[pos] = edges[static_cast<std::size_t>(pos)].weight;
  }

  return List::create(
    Rcpp::Named("from") = from,
    Rcpp::Named("to") = to,
    Rcpp::Named("weight") = weight,
    Rcpp::Named("n_vertices") = n,
    Rcpp::Named("n_edges") = m,
    Rcpp::Named("weight_type") = weight_type,
    Rcpp::Named("prune") = prune,
    Rcpp::Named("mutual") = mutual
  );
}
