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
#include <climits>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <random>
#include <queue>
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
  std::vector<std::size_t> row_ptr;
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
  graph.row_ptr.assign(static_cast<std::size_t>(n) + 1u, 0);

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
  std::vector<std::size_t> cursor = graph.row_ptr;
  for (const Edge& edge : graph.edges) {
    if (edge.u == edge.v) {
      const auto pos = cursor[static_cast<std::size_t>(edge.u)]++;
      graph.col_idx[static_cast<std::size_t>(pos)] = edge.u;
      graph.adjacency_weight[static_cast<std::size_t>(pos)] = 2.0 * edge.weight;
    } else {
      auto pos = cursor[static_cast<std::size_t>(edge.u)]++;
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
      for (auto pos = graph.row_ptr[static_cast<std::size_t>(u)];
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

int parallel_vertex_color(int vertex, std::uint64_t seed) {
  std::uint32_t value = static_cast<std::uint32_t>(vertex) ^
    static_cast<std::uint32_t>(seed ^ (seed >> 32U));
  value ^= value >> 16U;
  value *= 0x7feb352dU;
  value ^= value >> 15U;
  value *= 0x846ca68bU;
  value ^= value >> 16U;
  return static_cast<int>(value & 7U);
}

LocalMoveResult local_move_parallel(const Graph& graph,
    const std::vector<int>& initial, int max_passes,
    double resolution, std::uint64_t seed, int threads) {
  LocalMoveResult result;
  result.membership = compact_membership(initial);
  if (!(graph.volume > 0.0) || graph.n < 2) return result;
  const int n = graph.n;
  threads = std::min(threads, n);
  const double tolerance = louvain_move_tolerance(
    graph.total_edge_weight);
  std::vector<int> proposal(static_cast<std::size_t>(n));
  std::vector<double> volume(static_cast<std::size_t>(n));
  std::vector<int> count(static_cast<std::size_t>(n));
  for (int pass = 0; pass < max_passes; ++pass) {
    Rcpp::checkUserInterrupt();
    const std::uint64_t pass_seed = seed ^
      (0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(pass + 1));
    int pass_moves = 0;
    for (int color = 0; color < 8; ++color) {
      std::fill(volume.begin(), volume.end(), 0.0);
      std::fill(count.begin(), count.end(), 0);
      for (int u = 0; u < n; ++u) {
        const int label = result.membership[static_cast<std::size_t>(u)];
        volume[static_cast<std::size_t>(label)] +=
          graph.degree[static_cast<std::size_t>(u)];
        ++count[static_cast<std::size_t>(label)];
      }
      proposal = result.membership;
      std::vector<std::exception_ptr> errors(
        static_cast<std::size_t>(threads));
      std::vector<std::thread> workers;
      workers.reserve(static_cast<std::size_t>(threads));
      auto propose_range = [&](int worker, int begin, int end) {
        try {
          for (int u = begin; u < end; ++u) {
            if (parallel_vertex_color(u, pass_seed) != color) continue;
            const double degree = graph.degree[static_cast<std::size_t>(u)];
            if (!(degree > 0.0)) continue;
            const int old = result.membership[static_cast<std::size_t>(u)];
            std::unordered_map<int, double> neighbor_weight;
            for (auto pos = graph.row_ptr[static_cast<std::size_t>(u)];
                 pos < graph.row_ptr[static_cast<std::size_t>(u + 1)];
                 ++pos) {
              const int v = graph.col_idx[static_cast<std::size_t>(pos)];
              if (v == u) continue;
              const int label =
                result.membership[static_cast<std::size_t>(v)];
              neighbor_weight[label] +=
                graph.adjacency_weight[static_cast<std::size_t>(pos)];
            }
            const double old_inside = neighbor_weight[old];
            const double old_score = louvain_move_score(old_inside,
              degree, volume[static_cast<std::size_t>(old)] - degree,
              graph.volume, resolution);
            int best = old;
            double best_score = old_score;
            for (const auto& candidate : neighbor_weight) {
              const int label = candidate.first;
              if (label == old ||
                  count[static_cast<std::size_t>(label)] == 0) continue;
              const double score = louvain_move_score(
                candidate.second, degree,
                volume[static_cast<std::size_t>(label)],
                graph.volume, resolution);
              if (score > best_score + tolerance ||
                  (std::abs(score - best_score) <= tolerance &&
                   score > old_score + tolerance && label < best)) {
                best = label;
                best_score = score;
              }
            }
            proposal[static_cast<std::size_t>(u)] = best;
          }
        } catch (...) {
          errors[static_cast<std::size_t>(worker)] =
            std::current_exception();
        }
      };
      try {
        for (int worker = 0; worker < threads; ++worker) {
          const int begin = static_cast<int>(
            static_cast<long long>(n) * worker / threads);
          const int end = static_cast<int>(
            static_cast<long long>(n) * (worker + 1) / threads);
          workers.emplace_back(propose_range, worker, begin, end);
        }
      } catch (...) {
        for (auto& worker : workers) worker.join();
        throw;
      }
      for (auto& worker : workers) worker.join();
      for (const auto& error : errors) {
        if (error) std::rethrow_exception(error);
      }
      int changes = 0;
      for (int u = 0; u < n; ++u) {
        changes += proposal[static_cast<std::size_t>(u)] !=
          result.membership[static_cast<std::size_t>(u)];
      }
      if (changes == 0) continue;
      const double before = modularity_score(
        graph, result.membership, resolution);
      const double after = modularity_score(graph, proposal, resolution);
      if (after + tolerance >= before) {
        result.membership.swap(proposal);
        pass_moves += changes;
      }
    }
    result.moves += pass_moves;
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
                                   std::uint64_t seed,
                                   int fit_threads = 1) {
  Graph graph = original;
  std::vector<int> original_to_current(static_cast<std::size_t>(original.n));
  std::iota(original_to_current.begin(), original_to_current.end(), 0);
  for (int level = 0; level < 64; ++level) {
    std::vector<int> singleton(static_cast<std::size_t>(graph.n));
    std::iota(singleton.begin(), singleton.end(), 0);
    const std::uint64_t level_seed = seed +
      static_cast<std::uint64_t>(level) * 0x9e3779b97f4a7c15ULL;
    LocalMoveResult moved = fit_threads == 1 ?
      local_move(graph, singleton, max_passes, resolution,
        level_seed) :
      local_move_parallel(graph, singleton, max_passes, resolution,
        level_seed, fit_threads);
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
    for (auto pos = graph.row_ptr[static_cast<std::size_t>(u)];
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
    for (auto pos = graph.row_ptr[static_cast<std::size_t>(u)];
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
                                  std::uint64_t seed,
                                  int fit_threads = 1) {
  Graph graph = original;
  std::vector<int> original_to_current(static_cast<std::size_t>(original.n));
  std::iota(original_to_current.begin(), original_to_current.end(), 0);
  std::vector<int> partition = compact_membership(initial);

  for (int level = 0; level < 64; ++level) {
    const std::uint64_t level_seed = seed +
      static_cast<std::uint64_t>(level) * 0x9e3779b97f4a7c15ULL;
    LocalMoveResult moved = fit_threads == 1 ?
      local_move(graph, partition, max_passes, resolution,
        level_seed) :
      local_move_parallel(graph, partition, max_passes, resolution,
        level_seed, fit_threads);
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
                                  std::uint64_t seed,
                                  int fit_threads = 1) {
  std::vector<int> membership(static_cast<std::size_t>(graph.n));
  std::iota(membership.begin(), membership.end(), 0);
  for (int iteration = 0; iteration < outer_iterations; ++iteration) {
    std::vector<int> next = leiden_hierarchy(
      graph, membership, max_passes, resolution,
      seed + static_cast<std::uint64_t>(iteration) *
        0x94d049bb133111ebULL, fit_threads);
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
      for (auto pos = graph.row_ptr[static_cast<std::size_t>(u)];
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

#include "graph_clustering_options.h"

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
Rcpp::List fastembedr_graph_canonical_cpp(Rcpp::IntegerVector from,
                                         Rcpp::IntegerVector to,
                                         Rcpp::NumericVector weight,
                                         int n_vertices) {
  return graph_edges_r(graph_from_r(from, to, weight, n_vertices));
}

// [[Rcpp::export]]
Rcpp::List fastembedr_graph_projection_cpp(Rcpp::IntegerVector from,
                                          Rcpp::IntegerVector to,
                                          Rcpp::NumericVector weight,
                                          int n_vertices,
                                          SEXP projection) {
  Rcpp::List metadata;
  Graph graph = projection_graph(
    graph_from_r(from, to, weight, n_vertices), projection, metadata);
  return Rcpp::List::create(Rcpp::Named("graph") = graph_edges_r(graph),
                            Rcpp::Named("metadata") = metadata);
}

// [[Rcpp::export]]
Rcpp::List fastembedr_graph_small_groups_cpp(Rcpp::IntegerVector from,
                                            Rcpp::IntegerVector to,
                                            Rcpp::NumericVector weight,
                                            int n_vertices,
                                            Rcpp::IntegerVector membership,
                                            double resolution,
                                            SEXP small_groups) {
  if (membership.size() != n_vertices) {
    Rcpp::stop("CUDA membership length must equal graph vertices.");
  }
  if (!std::isfinite(resolution) || resolution <= 0) {
    Rcpp::stop("Resolution must be positive and finite.");
  }
  Graph graph = graph_from_r(from, to, weight, n_vertices);
  std::vector<int> raw(static_cast<std::size_t>(n_vertices));
  for (int i = 0; i < n_vertices; ++i) {
    if (membership[i] < 1 || membership[i] > n_vertices) {
      Rcpp::stop("CUDA membership must contain valid one-based labels.");
    }
    raw[static_cast<std::size_t>(i)] = membership[i] - 1;
  }
  raw = compact_membership(raw);
  Rcpp::List metadata;
  Rcpp::IntegerVector reassigned, unassigned;
  const auto labels = resolve_graph_groups(raw, small_groups, metadata,
    reassigned, unassigned);
  return Rcpp::List::create(
    Rcpp::Named("membership") = graph_membership_r(labels),
    Rcpp::Named("modularity") = unassigned.size() ? NA_REAL :
      modularity_score(graph, labels, resolution),
    Rcpp::Named("n_communities") = community_count(labels),
    Rcpp::Named("connected_communities") =
      Rcpp::LogicalVector::create(unassigned.size() ? NA_LOGICAL :
        communities_are_connected(graph, labels)),
    Rcpp::Named("membership_original") = membership,
    Rcpp::Named("modularity_original") =
      modularity_score(graph, raw, resolution),
    Rcpp::Named("reassigned") = reassigned,
    Rcpp::Named("unassigned") = unassigned,
    Rcpp::Named("metadata") = metadata);
}

// [[Rcpp::export]]
Rcpp::List fastembedr_graph_cluster_cpp(Rcpp::IntegerVector from,
                                        Rcpp::IntegerVector to,
                                        Rcpp::NumericVector weight,
                                        int n_vertices,
                                        std::string method = "leiden",
                                        Rcpp::NumericVector resolution =
                                          Rcpp::NumericVector::create(1.0),
                                        int n_iterations = 10,
                                        int n_runs = 1,
                                        Rcpp::NumericVector seed =
                                          Rcpp::NumericVector::create(1.0),
                                        SEXP projection = R_NilValue,
                                        SEXP small_groups = R_NilValue,
                                        int n_threads = 1,
                                        bool parallel_fit = false) {
  if (method != "louvain" && method != "leiden") {
    Rcpp::stop("Native modularity clustering supports `louvain` and `leiden`.");
  }
  if (!resolution.size() || !seed.size()) {
    Rcpp::stop("Resolution and seed must be nonempty.");
  }
  for (double value : resolution) {
    if (!std::isfinite(value) || value <= 0) {
      Rcpp::stop("Resolution must contain positive finite values.");
    }
  }
  for (double value : seed) {
    if (!std::isfinite(value) || value != std::floor(value) ||
        std::abs(value) > INT_MAX) Rcpp::stop("Seeds must be finite integers.");
  }
  if (n_iterations < 1 || n_runs < 1) {
    Rcpp::stop("`n_iterations` and `n_runs` must be positive.");
  }
  if (n_threads < 1 || (!parallel_fit && n_threads > n_runs)) {
    Rcpp::stop("CPU restart workers must be between 1 and `n_runs`.");
  }
  if (parallel_fit && n_threads > 32) {
    Rcpp::stop("CPU fit mode supports at most 32 workers.");
  }
  const bool consensus = resolution.size() > 1 || seed.size() > 1;
  if (consensus && n_runs != 1) {
    Rcpp::stop("With vector controls use n_runs = 1; seed defines repeats.");
  }
  if (consensus && n_threads != 1) {
    Rcpp::stop("CPU restart workers require scalar controls.");
  }
  const double final_resolution = resolution[0];
  Graph graph = graph_from_r(from, to, weight, n_vertices);
  Rcpp::List consensus_meta, projection_meta, small_meta;
  Rcpp::CharacterVector order;
  if (consensus) {
    graph = consensus_graph(graph, resolution, seed, method, n_iterations,
      consensus_meta);
    order.push_back("consensus");
  }
  if (projection != R_NilValue) {
    graph = projection_graph(graph, projection, projection_meta);
    order.push_back("projection");
  }
  if ((consensus || projection != R_NilValue) &&
      graph.edges.empty()) Rcpp::stop("Reweighting removed all graph edges.");
  const std::uint64_t base_seed =
    static_cast<std::uint64_t>(std::llround(seed[0]));
  std::vector<int> best = cluster_partition(graph, method, final_resolution,
    n_iterations, n_runs, base_seed,
    parallel_fit ? 1 : n_threads, parallel_fit ? n_threads : 1);
  order.push_back("clustering");
  const std::vector<int> original = best;
  Rcpp::IntegerVector reassigned, unassigned;
  if (small_groups != R_NilValue) {
    best = resolve_graph_groups(best, small_groups, small_meta,
      reassigned, unassigned);
    order.push_back("small_groups");
  }
  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("membership") = graph_membership_r(best),
    Rcpp::Named("modularity") = unassigned.size() ? NA_REAL :
      modularity_score(graph, best, final_resolution),
    Rcpp::Named("n_communities") = community_count(best),
    Rcpp::Named("method") = method,
    Rcpp::Named("backend") = "cpu",
    Rcpp::Named("connected_communities") = Rcpp::LogicalVector::create(
      unassigned.size() ? NA_LOGICAL : communities_are_connected(graph, best)),
    Rcpp::Named("implementation") = method == "louvain" ?
      "native_multilevel_louvain" : "native_leiden_local_move_refine_aggregate",
    Rcpp::Named("effective_cpu_workers") = parallel_fit ?
      std::min(n_threads, n_vertices) : n_threads,
    Rcpp::Named("parallel_mode") = parallel_fit ? "fit" :
      (n_threads > 1 ? "restarts" : "single")
  );
  if (consensus || projection != R_NilValue) {
    result["reweighted_graph"] = graph_edges_r(graph);
  }
  if (small_groups != R_NilValue) {
    result["membership_original"] = graph_membership_r(original);
    result["modularity_original"] =
      modularity_score(graph, original, final_resolution);
    result["reassigned"] = reassigned;
    result["unassigned"] = unassigned;
  }
  if (consensus || projection != R_NilValue ||
      small_groups != R_NilValue) {
    result["parameters"] = Rcpp::List::create(
      Rcpp::Named("consensus") = !consensus ? R_NilValue :
        static_cast<SEXP>(consensus_meta),
      Rcpp::Named("projection") = projection == R_NilValue ? R_NilValue :
        static_cast<SEXP>(projection_meta),
      Rcpp::Named("small_groups") = small_groups == R_NilValue ? R_NilValue :
        static_cast<SEXP>(small_meta),
      Rcpp::Named("weighted_edges") =
        static_cast<double>(graph.edges.size()),
      Rcpp::Named("final_resolution") = final_resolution,
      Rcpp::Named("final_seed") = seed[0],
      Rcpp::Named("processing_order") = order);
  }
  return result;
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
