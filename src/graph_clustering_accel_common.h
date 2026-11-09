#ifndef FASTEMBEDR_GRAPH_CLUSTERING_ACCEL_COMMON_H
#define FASTEMBEDR_GRAPH_CLUSTERING_ACCEL_COMMON_H

#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fastembedr_graph_accel {

struct Edge {
  int u;
  int v;
  float weight;
};

struct CandidateEdge {
  std::uint64_t key;
  float weight;
};

struct Graph {
  int n = 0;
  std::vector<Edge> edges;
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<float> weight;
  std::vector<float> degree;
  double total_edge_weight = 0.0;
  float volume = 0.0f;
};

struct MoveResult {
  std::vector<int> membership;
  int moves = 0;
  int passes = 0;
};

using MoveFunction = std::function<MoveResult(
  const Graph&,
  const std::vector<int>&,
  const std::vector<int>*,
  int,
  float,
  std::uint64_t,
  bool
)>;

inline std::uint64_t edge_key(int u, int v) {
  if (u > v) std::swap(u, v);
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(u)) << 32U) |
    static_cast<std::uint32_t>(v);
}

inline int edge_left(std::uint64_t key) {
  return static_cast<int>(key >> 32U);
}

inline int edge_right(std::uint64_t key) {
  return static_cast<int>(key & 0xffffffffULL);
}

inline std::vector<int> compact_membership(const std::vector<int>& membership) {
  std::unordered_map<int, int> labels;
  labels.reserve(membership.size());
  std::vector<int> out(membership.size());
  int next = 0;
  for (std::size_t i = 0; i < membership.size(); ++i) {
    auto inserted = labels.emplace(membership[i], next);
    if (inserted.second) ++next;
    out[i] = inserted.first->second;
  }
  return out;
}

inline int community_count(const std::vector<int>& membership) {
  if (membership.empty()) return 0;
  return *std::max_element(membership.begin(), membership.end()) + 1;
}

inline Graph graph_from_edges(int n, std::vector<Edge> edges) {
  std::uint64_t positions = 0;
  for (const Edge& edge : edges) {
    positions += edge.u == edge.v ? 1ULL : 2ULL;
    if (positions > static_cast<std::uint64_t>(
        std::numeric_limits<int>::max())) {
      Rcpp::stop("Resident GPU clustering exceeds signed-32-bit CSR "
                 "capacity; use a bounded file-backed route.");
    }
  }
  Graph graph;
  graph.n = n;
  graph.edges = std::move(edges);
  graph.degree.assign(static_cast<std::size_t>(n), 0.0f);
  graph.row_ptr.assign(static_cast<std::size_t>(n) + 1U, 0);

  for (const Edge& edge : graph.edges) {
    graph.total_edge_weight += edge.weight;
    if (edge.u == edge.v) {
      graph.degree[static_cast<std::size_t>(edge.u)] += 2.0f * edge.weight;
      ++graph.row_ptr[static_cast<std::size_t>(edge.u + 1)];
    } else {
      graph.degree[static_cast<std::size_t>(edge.u)] += edge.weight;
      graph.degree[static_cast<std::size_t>(edge.v)] += edge.weight;
      ++graph.row_ptr[static_cast<std::size_t>(edge.u + 1)];
      ++graph.row_ptr[static_cast<std::size_t>(edge.v + 1)];
    }
  }
  graph.volume = static_cast<float>(2.0 * graph.total_edge_weight);
  std::partial_sum(graph.row_ptr.begin(), graph.row_ptr.end(), graph.row_ptr.begin());
  graph.col_idx.resize(static_cast<std::size_t>(graph.row_ptr.back()));
  graph.weight.resize(static_cast<std::size_t>(graph.row_ptr.back()));
  std::vector<int> cursor = graph.row_ptr;
  for (const Edge& edge : graph.edges) {
    if (edge.u == edge.v) {
      const int position = cursor[static_cast<std::size_t>(edge.u)]++;
      graph.col_idx[static_cast<std::size_t>(position)] = edge.u;
      graph.weight[static_cast<std::size_t>(position)] = 2.0f * edge.weight;
    } else {
      int position = cursor[static_cast<std::size_t>(edge.u)]++;
      graph.col_idx[static_cast<std::size_t>(position)] = edge.v;
      graph.weight[static_cast<std::size_t>(position)] = edge.weight;
      position = cursor[static_cast<std::size_t>(edge.v)]++;
      graph.col_idx[static_cast<std::size_t>(position)] = edge.u;
      graph.weight[static_cast<std::size_t>(position)] = edge.weight;
    }
  }
  return graph;
}

inline Graph graph_from_r(const Rcpp::IntegerVector& from,
                          const Rcpp::IntegerVector& to,
                          const Rcpp::NumericVector& weight,
                          int n_vertices) {
  if (n_vertices < 1) Rcpp::stop("`n_vertices` must be positive.");
  if (from.size() != to.size() || from.size() != weight.size()) {
    Rcpp::stop("Graph edge vectors must have equal lengths.");
  }
  // A non-self input edge can occupy two CSR positions. Reject the
  // worst case before allocating the raw and deduplicated edge copies.
  if (static_cast<std::uint64_t>(from.size()) >
      static_cast<std::uint64_t>(std::numeric_limits<int>::max()) / 2ULL) {
    Rcpp::stop("Resident GPU clustering exceeds signed-32-bit input "
               "capacity; use a bounded file-backed route.");
  }
  std::vector<CandidateEdge> raw;
  raw.reserve(static_cast<std::size_t>(from.size()));
  for (R_xlen_t i = 0; i < from.size(); ++i) {
    const int u = from[i] - 1;
    const int v = to[i] - 1;
    const double value = weight[i];
    if (u < 0 || u >= n_vertices || v < 0 || v >= n_vertices) {
      Rcpp::stop("Graph vertices must be between 1 and `n_vertices`.");
    }
    if (!std::isfinite(value) || value < 0.0) {
      Rcpp::stop("Graph weights must be finite and non-negative.");
    }
    if (value > 0.0) {
      raw.push_back(CandidateEdge{
        edge_key(u, v),
        static_cast<float>(value)
      });
    }
  }
  std::sort(raw.begin(), raw.end(), [](const CandidateEdge& left,
                                       const CandidateEdge& right) {
    if (left.key != right.key) return left.key < right.key;
    return left.weight > right.weight;
  });
  std::vector<Edge> edges;
  edges.reserve(raw.size());
  for (std::size_t begin = 0; begin < raw.size();) {
    std::size_t end = begin + 1U;
    float maximum = raw[begin].weight;
    while (end < raw.size() && raw[end].key == raw[begin].key) {
      maximum = std::max(maximum, raw[end].weight);
      ++end;
    }
    edges.push_back(Edge{
      edge_left(raw[begin].key),
      edge_right(raw[begin].key),
      maximum
    });
    begin = end;
  }
  return graph_from_edges(n_vertices, std::move(edges));
}

inline Graph aggregate_graph(const Graph& graph,
                             const std::vector<int>& membership0) {
  const std::vector<int> membership = compact_membership(membership0);
  const int n = community_count(membership);
  std::vector<CandidateEdge> candidates;
  candidates.reserve(graph.edges.size());
  for (const Edge& edge : graph.edges) {
    candidates.push_back(CandidateEdge{
      edge_key(
        membership[static_cast<std::size_t>(edge.u)],
        membership[static_cast<std::size_t>(edge.v)]
      ),
      edge.weight
    });
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const CandidateEdge& left, const CandidateEdge& right) {
    return left.key < right.key;
  });
  std::vector<Edge> edges;
  edges.reserve(candidates.size());
  for (std::size_t begin = 0; begin < candidates.size();) {
    std::size_t end = begin + 1U;
    double sum = candidates[begin].weight;
    while (end < candidates.size() && candidates[end].key == candidates[begin].key) {
      sum += candidates[end].weight;
      ++end;
    }
    edges.push_back(Edge{
      edge_left(candidates[begin].key),
      edge_right(candidates[begin].key),
      static_cast<float>(sum)
    });
    begin = end;
  }
  return graph_from_edges(n, std::move(edges));
}

inline double modularity(const Graph& graph,
                         const std::vector<int>& membership0,
                         double resolution) {
  if (!(graph.total_edge_weight > 0.0)) return NA_REAL;
  const std::vector<int> membership = compact_membership(membership0);
  const int count = community_count(membership);
  std::vector<double> internal(static_cast<std::size_t>(count), 0.0);
  std::vector<double> volume(static_cast<std::size_t>(count), 0.0);
  for (int u = 0; u < graph.n; ++u) {
    volume[static_cast<std::size_t>(membership[static_cast<std::size_t>(u)])] +=
      graph.degree[static_cast<std::size_t>(u)];
  }
  for (const Edge& edge : graph.edges) {
    const int left = membership[static_cast<std::size_t>(edge.u)];
    const int right = membership[static_cast<std::size_t>(edge.v)];
    if (left == right) internal[static_cast<std::size_t>(left)] += edge.weight;
  }
  double score = 0.0;
  for (int community = 0; community < count; ++community) {
    const double fraction =
      volume[static_cast<std::size_t>(community)] / graph.volume;
    score += internal[static_cast<std::size_t>(community)] /
      graph.total_edge_weight - resolution * fraction * fraction;
  }
  return score;
}

inline bool communities_are_connected(const Graph& graph,
                                      const std::vector<int>& membership0) {
  const std::vector<int> membership = compact_membership(membership0);
  const int count = community_count(membership);
  std::vector<int> expected(static_cast<std::size_t>(count), 0);
  for (const int value : membership) ++expected[static_cast<std::size_t>(value)];
  std::vector<unsigned char> seen(static_cast<std::size_t>(graph.n), 0);
  std::vector<int> stack;
  for (int community = 0; community < count; ++community) {
    int start = -1;
    for (int u = 0; u < graph.n; ++u) {
      if (membership[static_cast<std::size_t>(u)] == community) {
        start = u;
        break;
      }
    }
    if (start < 0) continue;
    stack.assign(1, start);
    seen[static_cast<std::size_t>(start)] = 1;
    int reached = 0;
    while (!stack.empty()) {
      const int u = stack.back();
      stack.pop_back();
      ++reached;
      for (int position = graph.row_ptr[static_cast<std::size_t>(u)];
           position < graph.row_ptr[static_cast<std::size_t>(u + 1)]; ++position) {
        const int v = graph.col_idx[static_cast<std::size_t>(position)];
        if (!seen[static_cast<std::size_t>(v)] &&
            membership[static_cast<std::size_t>(v)] == community) {
          seen[static_cast<std::size_t>(v)] = 1;
          stack.push_back(v);
        }
      }
    }
    if (reached != expected[static_cast<std::size_t>(community)]) return false;
  }
  return true;
}

inline std::vector<int> split_disconnected(
    const Graph& graph,
    const std::vector<int>& membership0) {
  const std::vector<int> membership = compact_membership(membership0);
  std::vector<int> out(static_cast<std::size_t>(graph.n), -1);
  std::vector<int> stack;
  int next = 0;
  for (int start = 0; start < graph.n; ++start) {
    if (out[static_cast<std::size_t>(start)] >= 0) continue;
    const int parent = membership[static_cast<std::size_t>(start)];
    out[static_cast<std::size_t>(start)] = next;
    stack.assign(1, start);
    while (!stack.empty()) {
      const int u = stack.back();
      stack.pop_back();
      for (int position = graph.row_ptr[static_cast<std::size_t>(u)];
           position < graph.row_ptr[static_cast<std::size_t>(u + 1)]; ++position) {
        const int v = graph.col_idx[static_cast<std::size_t>(position)];
        if (out[static_cast<std::size_t>(v)] < 0 &&
            membership[static_cast<std::size_t>(v)] == parent) {
          out[static_cast<std::size_t>(v)] = next;
          stack.push_back(v);
        }
      }
    }
    ++next;
  }
  return out;
}

inline std::vector<int> louvain_partition(
    const Graph& original,
    int max_passes,
    float resolution,
    std::uint64_t seed,
    const MoveFunction& move) {
  Graph graph = original;
  std::vector<int> original_to_current(static_cast<std::size_t>(original.n));
  std::iota(original_to_current.begin(), original_to_current.end(), 0);
  for (int level = 0; level < 64; ++level) {
    std::vector<int> singleton(static_cast<std::size_t>(graph.n));
    std::iota(singleton.begin(), singleton.end(), 0);
    MoveResult moved = move(
      graph, singleton, nullptr, max_passes, resolution,
      seed + static_cast<std::uint64_t>(level) * 0x9e3779b97f4a7c15ULL,
      false
    );
    moved.membership = compact_membership(moved.membership);
    const int count = community_count(moved.membership);
    for (int& node : original_to_current) {
      node = moved.membership[static_cast<std::size_t>(node)];
    }
    if (count == graph.n) break;
    graph = aggregate_graph(graph, moved.membership);
  }
  return compact_membership(original_to_current);
}

inline std::vector<int> leiden_hierarchy(
    const Graph& original,
    const std::vector<int>& initial,
    int max_passes,
    float resolution,
    std::uint64_t seed,
    const MoveFunction& move) {
  Graph graph = original;
  std::vector<int> original_to_current(static_cast<std::size_t>(original.n));
  std::iota(original_to_current.begin(), original_to_current.end(), 0);
  std::vector<int> partition = compact_membership(initial);

  for (int level = 0; level < 64; ++level) {
    MoveResult moved = move(
      graph, partition, nullptr, max_passes, resolution,
      seed + static_cast<std::uint64_t>(level) * 0x9e3779b97f4a7c15ULL,
      false
    );
    partition = compact_membership(moved.membership);
    if (community_count(partition) == graph.n) break;

    std::vector<int> singleton(static_cast<std::size_t>(graph.n));
    std::iota(singleton.begin(), singleton.end(), 0);
    MoveResult refinement = move(
      graph, singleton, &partition, 1, resolution,
      seed ^ (static_cast<std::uint64_t>(level + 1) * 0xbf58476d1ce4e5b9ULL),
      true
    );
    std::vector<int> refined =
      split_disconnected(graph, compact_membership(refinement.membership));
    const int refined_count = community_count(refined);
    if (refined_count == graph.n) break;

    for (int& node : original_to_current) {
      node = refined[static_cast<std::size_t>(node)];
    }
    std::vector<int> coarse_partition(static_cast<std::size_t>(refined_count), -1);
    for (int u = 0; u < graph.n; ++u) {
      const int child = refined[static_cast<std::size_t>(u)];
      if (coarse_partition[static_cast<std::size_t>(child)] < 0) {
        coarse_partition[static_cast<std::size_t>(child)] =
          partition[static_cast<std::size_t>(u)];
      }
    }
    graph = aggregate_graph(graph, refined);
    partition = compact_membership(coarse_partition);
  }

  std::vector<int> out(static_cast<std::size_t>(original.n));
  for (int u = 0; u < original.n; ++u) {
    out[static_cast<std::size_t>(u)] =
      partition[static_cast<std::size_t>(
        original_to_current[static_cast<std::size_t>(u)]
      )];
  }
  return compact_membership(out);
}

inline std::vector<int> leiden_partition(
    const Graph& graph,
    int outer_iterations,
    int max_passes,
    float resolution,
    std::uint64_t seed,
    const MoveFunction& move) {
  std::vector<int> membership(static_cast<std::size_t>(graph.n));
  std::iota(membership.begin(), membership.end(), 0);
  for (int iteration = 0; iteration < outer_iterations; ++iteration) {
    std::vector<int> next = leiden_hierarchy(
      graph, membership, max_passes, resolution,
      seed + static_cast<std::uint64_t>(iteration) * 0x94d049bb133111ebULL,
      move
    );
    if (next == membership) break;
    membership.swap(next);
  }
  return compact_membership(membership);
}

inline Rcpp::List cluster(
    const Rcpp::IntegerVector& from,
    const Rcpp::IntegerVector& to,
    const Rcpp::NumericVector& weight,
    int n_vertices,
    const std::string& method,
    double resolution,
    int n_iterations,
    int n_runs,
    double seed,
    const std::string& backend,
    const MoveFunction& move) {
  if (method != "louvain" && method != "leiden") {
    Rcpp::stop("GPU graph clustering supports `louvain` and `leiden`.");
  }
  if (!(resolution > 0.0) || !std::isfinite(resolution)) {
    Rcpp::stop("`resolution` must be positive.");
  }
  if (n_iterations < 1 || n_runs < 1) {
    Rcpp::stop("`n_iterations` and `n_runs` must be positive.");
  }
  const Graph graph = graph_from_r(from, to, weight, n_vertices);
  std::vector<int> best(static_cast<std::size_t>(n_vertices));
  std::iota(best.begin(), best.end(), 0);
  double best_modularity = modularity(graph, best, resolution);
  const std::uint64_t base_seed =
    static_cast<std::uint64_t>(std::llround(seed));

  for (int run = 0; run < n_runs; ++run) {
    const std::uint64_t run_seed = base_seed +
      static_cast<std::uint64_t>(run) * 0x9e3779b97f4a7c15ULL;
    std::vector<int> membership = method == "louvain" ?
      louvain_partition(
        graph, n_iterations, static_cast<float>(resolution), run_seed, move
      ) :
      leiden_partition(
        graph, n_iterations, n_iterations,
        static_cast<float>(resolution), run_seed, move
      );
    const double quality = modularity(graph, membership, resolution);
    if (run == 0 || quality > best_modularity + 1e-12) {
      best = std::move(membership);
      best_modularity = quality;
    }
  }

  Rcpp::IntegerVector membership(best.size());
  for (std::size_t i = 0; i < best.size(); ++i) {
    membership[static_cast<R_xlen_t>(i)] = best[i] + 1;
  }
  return Rcpp::List::create(
    Rcpp::Named("membership") = membership,
    Rcpp::Named("modularity") = best_modularity,
    Rcpp::Named("n_communities") = community_count(best),
    Rcpp::Named("method") = method,
    Rcpp::Named("backend") = backend,
    Rcpp::Named("connected_communities") =
      communities_are_connected(graph, best),
    Rcpp::Named("implementation") = method == "louvain" ?
      "native_parallel_multilevel_louvain" :
      "native_parallel_leiden_local_move_refine_aggregate",
    Rcpp::Named("graph_storage") = "float32_csr",
    Rcpp::Named("coarsening_backend") = "native_cpp"
  );
}

}  // namespace fastembedr_graph_accel

#endif
