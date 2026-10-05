/*
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT
 *
 * Independent package-native graph construction, Louvain, and Leiden code.
 * The Leiden phase organization was reviewed against NetworKit commit
 * 7b74f6af90bc0865c6c0937a206df63df331b712, specifically
 * include/networkit/community/ParallelLeiden.hpp and
 * networkit/cpp/community/ParallelLeiden.cpp. No NetworKit source is copied,
 * linked, or vendored. See inst/NOTICE for the design-reference boundary.
 */

#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "louvain_objective.h"

// Native graph storage and modularity optimization following the published
// Louvain and Leiden algorithms.

namespace {

struct Edge {
  int u;
  int v;
  double weight;
};

struct CandidateEdge {
  std::uint64_t key;
  double weight;
};

struct Graph {
  int n = 0;
  std::vector<Edge> edges;
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<double> adjacency_weight;
  std::vector<double> degree;
  double total_edge_weight = 0.0;
  double volume = 0.0;
};

std::uint64_t edge_key(int u, int v) {
  if (u > v) std::swap(u, v);
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(u)) << 32U) |
    static_cast<std::uint32_t>(v);
}

int edge_left(std::uint64_t key) {
  return static_cast<int>(key >> 32U);
}

int edge_right(std::uint64_t key) {
  return static_cast<int>(key & 0xffffffffULL);
}

std::vector<int> compact_membership(const std::vector<int>& membership) {
  std::unordered_map<int, int> labels;
  labels.reserve(membership.size());
  std::vector<int> out(membership.size());
  int next = 0;
  for (std::size_t i = 0; i < membership.size(); ++i) {
    const int label = membership[i];
    auto inserted = labels.emplace(label, next);
    if (inserted.second) ++next;
    out[i] = inserted.first->second;
  }
  return out;
}

int community_count(const std::vector<int>& membership) {
  if (membership.empty()) return 0;
  return *std::max_element(membership.begin(), membership.end()) + 1;
}

Graph graph_from_edges(int n, std::vector<Edge> edges) {
  Graph graph;
  graph.n = n;
  graph.edges = std::move(edges);
  graph.degree.assign(static_cast<std::size_t>(n), 0.0);
  graph.row_ptr.assign(static_cast<std::size_t>(n + 1), 0);

  for (const Edge& edge : graph.edges) {
    graph.total_edge_weight += edge.weight;
    if (edge.u == edge.v) {
      graph.degree[static_cast<std::size_t>(edge.u)] += 2.0 * edge.weight;
      ++graph.row_ptr[static_cast<std::size_t>(edge.u + 1)];
    } else {
      graph.degree[static_cast<std::size_t>(edge.u)] += edge.weight;
      graph.degree[static_cast<std::size_t>(edge.v)] += edge.weight;
      ++graph.row_ptr[static_cast<std::size_t>(edge.u + 1)];
      ++graph.row_ptr[static_cast<std::size_t>(edge.v + 1)];
    }
  }
  graph.volume = 2.0 * graph.total_edge_weight;
  std::partial_sum(graph.row_ptr.begin(), graph.row_ptr.end(), graph.row_ptr.begin());
  graph.col_idx.resize(static_cast<std::size_t>(graph.row_ptr.back()));
  graph.adjacency_weight.resize(static_cast<std::size_t>(graph.row_ptr.back()));
  std::vector<int> cursor = graph.row_ptr;
  for (const Edge& edge : graph.edges) {
    if (edge.u == edge.v) {
      const int pos = cursor[static_cast<std::size_t>(edge.u)]++;
      graph.col_idx[static_cast<std::size_t>(pos)] = edge.u;
      graph.adjacency_weight[static_cast<std::size_t>(pos)] = 2.0 * edge.weight;
    } else {
      int pos = cursor[static_cast<std::size_t>(edge.u)]++;
      graph.col_idx[static_cast<std::size_t>(pos)] = edge.v;
      graph.adjacency_weight[static_cast<std::size_t>(pos)] = edge.weight;
      pos = cursor[static_cast<std::size_t>(edge.v)]++;
      graph.col_idx[static_cast<std::size_t>(pos)] = edge.u;
      graph.adjacency_weight[static_cast<std::size_t>(pos)] = edge.weight;
    }
  }
  return graph;
}

Graph graph_from_r(const Rcpp::IntegerVector& from,
                   const Rcpp::IntegerVector& to,
                   const Rcpp::NumericVector& weight,
                   int n_vertices) {
  if (n_vertices < 1) Rcpp::stop("`n_vertices` must be positive.");
  if (from.size() != to.size() || from.size() != weight.size()) {
    Rcpp::stop("Graph `from`, `to`, and `weight` vectors must have equal lengths.");
  }
  std::vector<CandidateEdge> raw;
  raw.reserve(static_cast<std::size_t>(from.size()));
  for (R_xlen_t i = 0; i < from.size(); ++i) {
    const int u = from[i] - 1;
    const int v = to[i] - 1;
    const double w = weight[i];
    if (u < 0 || u >= n_vertices || v < 0 || v >= n_vertices) {
      Rcpp::stop("Graph vertex indices must be between 1 and `n_vertices`.");
    }
    if (!std::isfinite(w) || w < 0.0) {
      Rcpp::stop("Graph weights must be finite and non-negative.");
    }
    if (w > 0.0) raw.push_back(CandidateEdge{edge_key(u, v), w});
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
    double maximum = raw[begin].weight;
    while (end < raw.size() && raw[end].key == raw[begin].key) {
      maximum = std::max(maximum, raw[end].weight);
      ++end;
    }
    edges.push_back(Edge{edge_left(raw[begin].key), edge_right(raw[begin].key), maximum});
    begin = end;
  }
  return graph_from_edges(n_vertices, std::move(edges));
}

double modularity_score(const Graph& graph,
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
  double quality = 0.0;
  for (int c = 0; c < count; ++c) {
    const double fraction = volume[static_cast<std::size_t>(c)] / graph.volume;
    quality += internal[static_cast<std::size_t>(c)] / graph.total_edge_weight -
      resolution * fraction * fraction;
  }
  return quality;
}

struct LocalMoveResult {
  std::vector<int> membership;
  int moves = 0;
  int passes = 0;
};

LocalMoveResult local_move(const Graph& graph,
                           const std::vector<int>& initial,
                           int max_passes,
                           double resolution,
                           std::uint64_t seed) {
  LocalMoveResult result;
  result.membership = compact_membership(initial);
  if (!(graph.total_edge_weight > 0.0) || graph.n < 2) return result;

  const int n = graph.n;
  std::vector<int> counts(static_cast<std::size_t>(n), 0);
  std::vector<double> volumes(static_cast<std::size_t>(n), 0.0);
  for (int u = 0; u < n; ++u) {
    const int c = result.membership[static_cast<std::size_t>(u)];
    ++counts[static_cast<std::size_t>(c)];
    volumes[static_cast<std::size_t>(c)] += graph.degree[static_cast<std::size_t>(u)];
  }
  std::set<int> free_labels;
  for (int c = 0; c < n; ++c) {
    if (counts[static_cast<std::size_t>(c)] == 0) free_labels.insert(c);
  }

  std::vector<int> order(static_cast<std::size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::vector<double> community_weight(static_cast<std::size_t>(n), 0.0);
  std::vector<int> touched;
  touched.reserve(128);
  std::mt19937_64 generator(seed);
  const double tolerance = louvain_move_tolerance(
    graph.total_edge_weight);

  for (int pass = 0; pass < max_passes; ++pass) {
    std::shuffle(order.begin(), order.end(), generator);
    int pass_moves = 0;
    for (const int u : order) {
      const double node_degree = graph.degree[static_cast<std::size_t>(u)];
      if (!(node_degree > 0.0)) continue;
      touched.clear();
      for (int pos = graph.row_ptr[static_cast<std::size_t>(u)];
           pos < graph.row_ptr[static_cast<std::size_t>(u + 1)]; ++pos) {
        const int v = graph.col_idx[static_cast<std::size_t>(pos)];
        if (v == u) continue;
        const int c = result.membership[static_cast<std::size_t>(v)];
        if (community_weight[static_cast<std::size_t>(c)] == 0.0) touched.push_back(c);
        community_weight[static_cast<std::size_t>(c)] +=
          graph.adjacency_weight[static_cast<std::size_t>(pos)];
      }

      const int old = result.membership[static_cast<std::size_t>(u)];
      --counts[static_cast<std::size_t>(old)];
      volumes[static_cast<std::size_t>(old)] -= node_degree;
      if (counts[static_cast<std::size_t>(old)] == 0) free_labels.insert(old);

      const double old_inside = community_weight[static_cast<std::size_t>(old)];
      const double old_score = louvain_move_score(
        old_inside, node_degree, volumes[static_cast<std::size_t>(old)],
        graph.volume, resolution);
      int best = old;
      double best_score = old_score;
      bool choose_empty = false;
      if (best_score < -tolerance && !free_labels.empty()) {
        best_score = 0.0;
        choose_empty = true;
      }
      for (const int candidate : touched) {
        if (candidate == old || counts[static_cast<std::size_t>(candidate)] == 0) continue;
        const double score = louvain_move_score(
          community_weight[static_cast<std::size_t>(candidate)],
          node_degree, volumes[static_cast<std::size_t>(candidate)],
          graph.volume, resolution);
        if (score > best_score + tolerance) {
          best = candidate;
          best_score = score;
          choose_empty = false;
        }
      }

      if (best_score <= old_score + tolerance) {
        best = old;
        choose_empty = false;
      } else if (choose_empty) {
        best = *free_labels.begin();
      }

      free_labels.erase(best);
      ++counts[static_cast<std::size_t>(best)];
      volumes[static_cast<std::size_t>(best)] += node_degree;
      if (best != old) {
        result.membership[static_cast<std::size_t>(u)] = best;
        ++pass_moves;
        ++result.moves;
      }
      for (const int c : touched) community_weight[static_cast<std::size_t>(c)] = 0.0;
    }
    ++result.passes;
    if (pass_moves == 0) break;
  }
  result.membership = compact_membership(result.membership);
  return result;
}

Graph aggregate_graph(const Graph& graph, const std::vector<int>& membership0) {
  const std::vector<int> membership = compact_membership(membership0);
  const int n = community_count(membership);
  std::vector<CandidateEdge> candidates;
  candidates.reserve(graph.edges.size());
  for (const Edge& edge : graph.edges) {
    const int u = membership[static_cast<std::size_t>(edge.u)];
    const int v = membership[static_cast<std::size_t>(edge.v)];
    candidates.push_back(CandidateEdge{edge_key(u, v), edge.weight});
  }
  std::sort(candidates.begin(), candidates.end(), [](const CandidateEdge& left,
                                                     const CandidateEdge& right) {
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
    edges.push_back(Edge{edge_left(candidates[begin].key),
                         edge_right(candidates[begin].key), sum});
    begin = end;
  }
  return graph_from_edges(n, std::move(edges));
}

std::vector<int> louvain_partition(const Graph& original,
                                   int max_passes,
                                   double resolution,
                                   std::uint64_t seed) {
  Graph graph = original;
  std::vector<int> original_to_current(static_cast<std::size_t>(original.n));
  std::iota(original_to_current.begin(), original_to_current.end(), 0);
  for (int level = 0; level < 64; ++level) {
    std::vector<int> singleton(static_cast<std::size_t>(graph.n));
    std::iota(singleton.begin(), singleton.end(), 0);
    LocalMoveResult moved = local_move(
      graph, singleton, max_passes, resolution,
      seed + static_cast<std::uint64_t>(level) * 0x9e3779b97f4a7c15ULL);
    const int count = community_count(moved.membership);
    for (int& node : original_to_current) {
      node = moved.membership[static_cast<std::size_t>(node)];
    }
    if (count == graph.n) break;
    graph = aggregate_graph(graph, moved.membership);
  }
  return compact_membership(original_to_current);
}

std::vector<int> refine_partition(const Graph& graph,
                                  const std::vector<int>& parent0,
                                  double resolution,
                                  std::uint64_t seed) {
  const std::vector<int> parent = compact_membership(parent0);
  const int n = graph.n;
  const int parent_count = community_count(parent);
  std::vector<double> parent_volume(static_cast<std::size_t>(parent_count), 0.0);
  for (int u = 0; u < n; ++u) {
    parent_volume[static_cast<std::size_t>(parent[static_cast<std::size_t>(u)])] +=
      graph.degree[static_cast<std::size_t>(u)];
  }

  std::vector<int> refined(static_cast<std::size_t>(n));
  std::iota(refined.begin(), refined.end(), 0);
  std::vector<int> size(static_cast<std::size_t>(n), 1);
  std::vector<double> volume = graph.degree;
  std::vector<double> cut_to_parent(static_cast<std::size_t>(n), 0.0);
  for (int u = 0; u < n; ++u) {
    for (int pos = graph.row_ptr[static_cast<std::size_t>(u)];
         pos < graph.row_ptr[static_cast<std::size_t>(u + 1)]; ++pos) {
      const int v = graph.col_idx[static_cast<std::size_t>(pos)];
      if (v != u && parent[static_cast<std::size_t>(v)] ==
                      parent[static_cast<std::size_t>(u)]) {
        cut_to_parent[static_cast<std::size_t>(u)] +=
          graph.adjacency_weight[static_cast<std::size_t>(pos)];
      }
    }
  }

  std::vector<int> order(static_cast<std::size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::mt19937_64 generator(seed);
  std::shuffle(order.begin(), order.end(), generator);
  std::vector<double> community_weight(static_cast<std::size_t>(n), 0.0);
  std::vector<int> touched;
  touched.reserve(128);
  const double tolerance = 1e-12 * std::max(1.0, graph.total_edge_weight);

  for (const int u : order) {
    const int own = refined[static_cast<std::size_t>(u)];
    if (size[static_cast<std::size_t>(own)] != 1) continue;
    const int parent_id = parent[static_cast<std::size_t>(u)];
    const double node_degree = graph.degree[static_cast<std::size_t>(u)];
    const double r_threshold = resolution * node_degree *
      (parent_volume[static_cast<std::size_t>(parent_id)] - node_degree) /
      graph.volume;
    if (cut_to_parent[static_cast<std::size_t>(own)] + tolerance < r_threshold) continue;

    touched.clear();
    for (int pos = graph.row_ptr[static_cast<std::size_t>(u)];
         pos < graph.row_ptr[static_cast<std::size_t>(u + 1)]; ++pos) {
      const int v = graph.col_idx[static_cast<std::size_t>(pos)];
      if (v == u || parent[static_cast<std::size_t>(v)] != parent_id) continue;
      const int candidate = refined[static_cast<std::size_t>(v)];
      if (candidate == own) continue;
      if (community_weight[static_cast<std::size_t>(candidate)] == 0.0) {
        touched.push_back(candidate);
      }
      community_weight[static_cast<std::size_t>(candidate)] +=
        graph.adjacency_weight[static_cast<std::size_t>(pos)];
    }

    int best = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    for (const int candidate : touched) {
      const double candidate_volume = volume[static_cast<std::size_t>(candidate)];
      const double t_threshold = resolution * candidate_volume *
        (parent_volume[static_cast<std::size_t>(parent_id)] - candidate_volume) /
        graph.volume;
      if (cut_to_parent[static_cast<std::size_t>(candidate)] + tolerance < t_threshold) {
        continue;
      }
      const double score = community_weight[static_cast<std::size_t>(candidate)] -
        resolution * node_degree * candidate_volume / graph.volume;
      if (score >= -tolerance && score > best_score + tolerance) {
        best = candidate;
        best_score = score;
      }
    }
    if (best >= 0) {
      const double between = community_weight[static_cast<std::size_t>(best)];
      refined[static_cast<std::size_t>(u)] = best;
      size[static_cast<std::size_t>(best)] += 1;
      size[static_cast<std::size_t>(own)] = 0;
      volume[static_cast<std::size_t>(best)] += node_degree;
      volume[static_cast<std::size_t>(own)] = 0.0;
      cut_to_parent[static_cast<std::size_t>(best)] +=
        cut_to_parent[static_cast<std::size_t>(own)] - 2.0 * between;
      cut_to_parent[static_cast<std::size_t>(own)] = 0.0;
    }
    for (const int c : touched) community_weight[static_cast<std::size_t>(c)] = 0.0;
  }
  return compact_membership(refined);
}

std::vector<int> leiden_hierarchy(const Graph& original,
                                  const std::vector<int>& initial,
                                  int max_passes,
                                  double resolution,
                                  std::uint64_t seed) {
  Graph graph = original;
  std::vector<int> original_to_current(static_cast<std::size_t>(original.n));
  std::iota(original_to_current.begin(), original_to_current.end(), 0);
  std::vector<int> partition = compact_membership(initial);

  for (int level = 0; level < 64; ++level) {
    LocalMoveResult moved = local_move(
      graph, partition, max_passes, resolution,
      seed + static_cast<std::uint64_t>(level) * 0x9e3779b97f4a7c15ULL);
    partition = std::move(moved.membership);
    if (community_count(partition) == graph.n) break;

    std::vector<int> refined = refine_partition(
      graph, partition, resolution,
      seed ^ (static_cast<std::uint64_t>(level + 1) * 0xbf58476d1ce4e5b9ULL));
    const int refined_count = community_count(refined);
    if (refined_count == graph.n) break;

    for (int& node : original_to_current) {
      node = refined[static_cast<std::size_t>(node)];
    }
    std::vector<int> coarse_partition(static_cast<std::size_t>(refined_count), -1);
    for (int u = 0; u < graph.n; ++u) {
      const int child = refined[static_cast<std::size_t>(u)];
      const int parent = partition[static_cast<std::size_t>(u)];
      if (coarse_partition[static_cast<std::size_t>(child)] < 0) {
        coarse_partition[static_cast<std::size_t>(child)] = parent;
      }
    }
    graph = aggregate_graph(graph, refined);
    partition = compact_membership(coarse_partition);
  }

  std::vector<int> out(static_cast<std::size_t>(original.n));
  for (int u = 0; u < original.n; ++u) {
    out[static_cast<std::size_t>(u)] =
      partition[static_cast<std::size_t>(original_to_current[static_cast<std::size_t>(u)])];
  }
  return compact_membership(out);
}

std::vector<int> leiden_partition(const Graph& graph,
                                  int outer_iterations,
                                  int max_passes,
                                  double resolution,
                                  std::uint64_t seed) {
  std::vector<int> membership(static_cast<std::size_t>(graph.n));
  std::iota(membership.begin(), membership.end(), 0);
  for (int iteration = 0; iteration < outer_iterations; ++iteration) {
    std::vector<int> next = leiden_hierarchy(
      graph, membership, max_passes, resolution,
      seed + static_cast<std::uint64_t>(iteration) * 0x94d049bb133111ebULL);
    if (next == membership) break;
    membership.swap(next);
  }
  return compact_membership(membership);
}

bool communities_are_connected(const Graph& graph,
                               const std::vector<int>& membership0) {
  const std::vector<int> membership = compact_membership(membership0);
  const int count = community_count(membership);
  std::vector<int> expected(static_cast<std::size_t>(count), 0);
  for (const int c : membership) ++expected[static_cast<std::size_t>(c)];
  std::vector<unsigned char> seen(static_cast<std::size_t>(graph.n), 0);
  std::vector<int> stack;
  for (int c = 0; c < count; ++c) {
    int start = -1;
    for (int u = 0; u < graph.n; ++u) {
      if (membership[static_cast<std::size_t>(u)] == c) {
        start = u;
        break;
      }
    }
    if (start < 0) continue;
    stack.clear();
    stack.push_back(start);
    seen[static_cast<std::size_t>(start)] = 1;
    int reached = 0;
    while (!stack.empty()) {
      const int u = stack.back();
      stack.pop_back();
      ++reached;
      for (int pos = graph.row_ptr[static_cast<std::size_t>(u)];
           pos < graph.row_ptr[static_cast<std::size_t>(u + 1)]; ++pos) {
        const int v = graph.col_idx[static_cast<std::size_t>(pos)];
        if (!seen[static_cast<std::size_t>(v)] &&
            membership[static_cast<std::size_t>(v)] == c) {
          seen[static_cast<std::size_t>(v)] = 1;
          stack.push_back(v);
        }
      }
    }
    if (reached != expected[static_cast<std::size_t>(c)]) return false;
  }
  return true;
}

int shared_neighbor_count(const std::vector<int>& left,
                          const std::vector<int>& right) {
  std::size_t i = 0;
  std::size_t j = 0;
  int shared = 0;
  while (i < left.size() && j < right.size()) {
    if (left[i] == right[j]) {
      ++shared;
      ++i;
      ++j;
    } else if (left[i] < right[j]) {
      ++i;
    } else {
      ++j;
    }
  }
  return shared;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List fastembedr_graph_from_knn_cpp(Rcpp::IntegerMatrix indices,
                                         Rcpp::NumericMatrix distances,
                                         std::string weight_type = "snn",
                                         bool mutual = false,
                                         double prune = 0.0,
                                         int n_threads = 1) {
  const int n = indices.nrow();
  const int k = indices.ncol();
  if (n < 2 || k < 1 || distances.nrow() != n || distances.ncol() != k) {
    Rcpp::stop("KNN indices and distances must be equally sized non-empty matrices.");
  }
  if (weight_type != "snn" && weight_type != "distance" && weight_type != "binary") {
    Rcpp::stop("`weight_type` must be `snn`, `distance`, or `binary`.");
  }
  if (!std::isfinite(prune) || prune < 0.0) Rcpp::stop("`prune` must be non-negative.");
  n_threads = std::max(1, std::min(n_threads, n));

  const int* index_data = INTEGER(indices);
  const double* distance_data = REAL(distances);
  int minimum = std::numeric_limits<int>::max();
  int maximum = std::numeric_limits<int>::min();
  for (R_xlen_t i = 0; i < indices.size(); ++i) {
    const int value = index_data[i];
    if (value == NA_INTEGER) continue;
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
  }
  const bool one_based = minimum >= 1 && maximum <= n;

  std::vector<std::vector<int>> neighbors(static_cast<std::size_t>(n));
  auto fill_neighbors = [&](int begin, int end) {
    for (int u = begin; u < end; ++u) {
      std::vector<int>& row = neighbors[static_cast<std::size_t>(u)];
      row.reserve(static_cast<std::size_t>(k));
      for (int column = 0; column < k; ++column) {
        const int raw = index_data[u + n * column];
        if (raw == NA_INTEGER) continue;
        const int v = raw - (one_based ? 1 : 0);
        if (v >= 0 && v < n && v != u) row.push_back(v);
      }
      std::sort(row.begin(), row.end());
      row.erase(std::unique(row.begin(), row.end()), row.end());
    }
  };
  std::vector<std::thread> threads;
  for (int thread = 0; thread < n_threads; ++thread) {
    const int begin = static_cast<int>((static_cast<long long>(thread) * n) / n_threads);
    const int end = static_cast<int>((static_cast<long long>(thread + 1) * n) / n_threads);
    threads.emplace_back(fill_neighbors, begin, end);
  }
  for (std::thread& thread : threads) thread.join();
  threads.clear();

  std::vector<std::vector<CandidateEdge>> local(static_cast<std::size_t>(n_threads));
  auto build_candidates = [&](int thread, int begin, int end) {
    std::vector<CandidateEdge>& out = local[static_cast<std::size_t>(thread)];
    out.reserve(static_cast<std::size_t>(end - begin) * static_cast<std::size_t>(k));
    for (int u = begin; u < end; ++u) {
      for (int column = 0; column < k; ++column) {
        const int raw = index_data[u + n * column];
        if (raw == NA_INTEGER) continue;
        const int v = raw - (one_based ? 1 : 0);
        if (v < 0 || v >= n || v == u) continue;
        if (mutual && !std::binary_search(
              neighbors[static_cast<std::size_t>(v)].begin(),
              neighbors[static_cast<std::size_t>(v)].end(), u)) {
          continue;
        }
        double edge_weight = 1.0;
        if (weight_type == "distance") {
          const double distance = distance_data[u + n * column];
          if (!std::isfinite(distance) || distance < 0.0) continue;
          edge_weight = 1.0 / (1.0 + distance);
        } else if (weight_type == "snn") {
          const int shared = shared_neighbor_count(
            neighbors[static_cast<std::size_t>(u)],
            neighbors[static_cast<std::size_t>(v)]);
          const int union_size = static_cast<int>(neighbors[static_cast<std::size_t>(u)].size() +
            neighbors[static_cast<std::size_t>(v)].size()) - shared;
          edge_weight = union_size > 0 ?
            static_cast<double>(shared) / static_cast<double>(union_size) : 0.0;
        }
        if (edge_weight > prune && edge_weight > 0.0) {
          out.push_back(CandidateEdge{edge_key(u, v), edge_weight});
        }
      }
    }
  };
  for (int thread = 0; thread < n_threads; ++thread) {
    const int begin = static_cast<int>((static_cast<long long>(thread) * n) / n_threads);
    const int end = static_cast<int>((static_cast<long long>(thread + 1) * n) / n_threads);
    threads.emplace_back(build_candidates, thread, begin, end);
  }
  for (std::thread& thread : threads) thread.join();

  std::size_t total = 0;
  for (const auto& part : local) total += part.size();
  std::vector<CandidateEdge> candidates;
  candidates.reserve(total);
  for (auto& part : local) {
    candidates.insert(candidates.end(),
                      std::make_move_iterator(part.begin()),
                      std::make_move_iterator(part.end()));
  }
  std::sort(candidates.begin(), candidates.end(), [](const CandidateEdge& left,
                                                     const CandidateEdge& right) {
    if (left.key != right.key) return left.key < right.key;
    return left.weight > right.weight;
  });

  std::vector<Edge> edges;
  edges.reserve(candidates.size());
  for (std::size_t begin = 0; begin < candidates.size();) {
    std::size_t end = begin + 1U;
    double maximum = candidates[begin].weight;
    while (end < candidates.size() && candidates[end].key == candidates[begin].key) {
      maximum = std::max(maximum, candidates[end].weight);
      ++end;
    }
    edges.push_back(Edge{edge_left(candidates[begin].key),
                         edge_right(candidates[begin].key), maximum});
    begin = end;
  }

  Rcpp::IntegerVector from(edges.size());
  Rcpp::IntegerVector to(edges.size());
  Rcpp::NumericVector weight(edges.size());
  for (std::size_t i = 0; i < edges.size(); ++i) {
    from[static_cast<R_xlen_t>(i)] = edges[i].u + 1;
    to[static_cast<R_xlen_t>(i)] = edges[i].v + 1;
    weight[static_cast<R_xlen_t>(i)] = edges[i].weight;
  }
  return Rcpp::List::create(
    Rcpp::Named("from") = from,
    Rcpp::Named("to") = to,
    Rcpp::Named("weight") = weight,
    Rcpp::Named("n_vertices") = n,
    Rcpp::Named("n_edges") = static_cast<double>(edges.size()),
    Rcpp::Named("weight_type") = weight_type,
    Rcpp::Named("mutual") = mutual,
    Rcpp::Named("prune") = prune,
    Rcpp::Named("n_threads") = n_threads
  );
}

// [[Rcpp::export]]
Rcpp::List fastembedr_graph_cluster_cpp(Rcpp::IntegerVector from,
                                        Rcpp::IntegerVector to,
                                        Rcpp::NumericVector weight,
                                        int n_vertices,
                                        std::string method = "leiden",
                                        double resolution = 1.0,
                                        int n_iterations = 10,
                                        int n_runs = 1,
                                        double seed = 1) {
  if (method != "louvain" && method != "leiden") {
    Rcpp::stop("Native modularity clustering supports `louvain` and `leiden`.");
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
  double best_modularity = modularity_score(graph, best, resolution);
  const std::uint64_t base_seed = static_cast<std::uint64_t>(std::llround(seed));

  for (int run = 0; run < n_runs; ++run) {
    const std::uint64_t run_seed = base_seed +
      static_cast<std::uint64_t>(run) * 0x9e3779b97f4a7c15ULL;
    std::vector<int> membership = method == "louvain" ?
      louvain_partition(graph, n_iterations, resolution, run_seed) :
      leiden_partition(graph, n_iterations, n_iterations, resolution, run_seed);
    const double quality = modularity_score(graph, membership, resolution);
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
    Rcpp::Named("backend") = "cpu",
    Rcpp::Named("connected_communities") = communities_are_connected(graph, best),
    Rcpp::Named("implementation") = method == "louvain" ?
      "native_multilevel_louvain" : "native_leiden_local_move_refine_aggregate"
  );
}

// [[Rcpp::export]]
double fastembedr_graph_modularity_cpp(Rcpp::IntegerVector from,
                                       Rcpp::IntegerVector to,
                                       Rcpp::NumericVector weight,
                                       int n_vertices,
                                       Rcpp::IntegerVector membership,
                                       double resolution = 1.0) {
  if (membership.size() != n_vertices) {
    Rcpp::stop("`membership` length must equal `n_vertices`.");
  }
  std::vector<int> labels(static_cast<std::size_t>(n_vertices));
  for (int i = 0; i < n_vertices; ++i) {
    if (membership[i] == NA_INTEGER) Rcpp::stop("`membership` cannot contain NA.");
    labels[static_cast<std::size_t>(i)] = membership[i];
  }
  return modularity_score(graph_from_r(from, to, weight, n_vertices), labels, resolution);
}
