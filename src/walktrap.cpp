/*
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT
 *
 * Independent implementation of the Pons-Latapy random-walk community
 * algorithm. No igraph Walktrap source is copied, linked, or vendored; igraph
 * is used only as an optional validation reference.
 */

#include <Rcpp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct WalkEdge {
  int u;
  int v;
  double weight;
};

struct WalkNeighbor {
  double delta = 0.0;
  double edge_weight = 0.0;
};

struct WalkCommunity {
  bool active = false;
  int size = 0;
  int probability_slot = -1;
  double degree = 0.0;
  std::unordered_map<int, WalkNeighbor> neighbors;
};

struct WalkCandidate {
  double delta;
  int left;
  int right;
};

struct WalkCandidateGreater {
  bool operator()(const WalkCandidate& left, const WalkCandidate& right) const {
    if (left.delta != right.delta) return left.delta > right.delta;
    if (left.left != right.left) return left.left > right.left;
    return left.right > right.right;
  }
};

struct DisjointSet {
  std::vector<int> parent;
  std::vector<int> rank;

  explicit DisjointSet(int n)
      : parent(static_cast<std::size_t>(n)),
        rank(static_cast<std::size_t>(n), 0) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  int find(int value) {
    int root = value;
    while (parent[static_cast<std::size_t>(root)] != root) {
      root = parent[static_cast<std::size_t>(root)];
    }
    while (parent[static_cast<std::size_t>(value)] != value) {
      const int next = parent[static_cast<std::size_t>(value)];
      parent[static_cast<std::size_t>(value)] = root;
      value = next;
    }
    return root;
  }

  int unite(int left, int right) {
    left = find(left);
    right = find(right);
    if (left == right) return left;
    if (rank[static_cast<std::size_t>(left)] < rank[static_cast<std::size_t>(right)]) {
      std::swap(left, right);
    }
    parent[static_cast<std::size_t>(right)] = left;
    if (rank[static_cast<std::size_t>(left)] == rank[static_cast<std::size_t>(right)]) {
      ++rank[static_cast<std::size_t>(left)];
    }
    return left;
  }
};

double walk_distance_squared(const std::vector<double>& probability,
                             int n,
                             int left_slot,
                             int right_slot,
                             const std::vector<double>& inverse_degree,
                             const std::vector<std::vector<int>>& support) {
  const std::size_t left_offset = static_cast<std::size_t>(left_slot) *
    static_cast<std::size_t>(n);
  const std::size_t right_offset = static_cast<std::size_t>(right_slot) *
    static_cast<std::size_t>(n);
  double distance = 0.0;
  const auto& left = support[static_cast<std::size_t>(left_slot)];
  const auto& right = support[static_cast<std::size_t>(right_slot)];
  std::size_t i = 0;
  std::size_t j = 0;
  while (i < left.size() || j < right.size()) {
    int k;
    if (j == right.size() || (i < left.size() && left[i] < right[j])) {
      k = left[i++];
    } else if (i == left.size() || right[j] < left[i]) {
      k = right[j++];
    } else {
      k = left[i++];
      ++j;
    }
    const std::size_t offset = static_cast<std::size_t>(k);
    const double difference = probability[left_offset + offset] -
      probability[right_offset + offset];
    distance += difference * difference * inverse_degree[offset];
  }
  return distance;
}

std::vector<int> replay_partition(
    int n,
    const std::vector<std::pair<int, int>>& merges,
    int merge_count) {
  DisjointSet sets(n);
  std::vector<int> representative(static_cast<std::size_t>(2 * n - 1), -1);
  for (int i = 0; i < n; ++i) representative[static_cast<std::size_t>(i)] = i;
  for (int step = 0; step < merge_count; ++step) {
    const int left = merges[static_cast<std::size_t>(step)].first;
    const int right = merges[static_cast<std::size_t>(step)].second;
    const int root = sets.unite(
      representative[static_cast<std::size_t>(left)],
      representative[static_cast<std::size_t>(right)]);
    representative[static_cast<std::size_t>(n + step)] = root;
  }

  std::unordered_map<int, int> labels;
  std::vector<int> membership(static_cast<std::size_t>(n));
  int next = 0;
  for (int i = 0; i < n; ++i) {
    const int root = sets.find(i);
    auto inserted = labels.emplace(root, next);
    if (inserted.second) ++next;
    membership[static_cast<std::size_t>(i)] = inserted.first->second;
  }
  return membership;
}

}  // namespace

// Independent implementation of the equations and adjacent-community
// agglomeration in Pons and Latapy (2005), validated against igraph in tests.
// [[Rcpp::export]]
Rcpp::List fastembedr_walktrap_cpp(Rcpp::IntegerVector from,
                                   Rcpp::IntegerVector to,
                                   Rcpp::NumericVector weight,
                                   int n_vertices,
                                   int steps = 4) {
  if (n_vertices < 1) Rcpp::stop("`n_vertices` must be positive.");
  if (from.size() != to.size() || from.size() != weight.size()) {
    Rcpp::stop("Graph `from`, `to`, and `weight` vectors must have equal lengths.");
  }
  if (steps < 1) Rcpp::stop("`steps` must be positive.");

  const int n = n_vertices;
  const std::size_t dense_entries = static_cast<std::size_t>(n) *
    static_cast<std::size_t>(n);
  if (dense_entries > 16000000ULL) {
    Rcpp::stop(
      "Exact Walktrap would exceed its 16-million-entry transition-matrix limit. "
      "Use Leiden or Louvain for larger graphs.");
  }

  std::vector<std::unordered_map<int, double>> adjacency(static_cast<std::size_t>(n));
  for (R_xlen_t edge = 0; edge < from.size(); ++edge) {
    const int u = from[edge] - 1;
    const int v = to[edge] - 1;
    const double w = weight[edge];
    if (u < 0 || u >= n || v < 0 || v >= n) {
      Rcpp::stop("Graph vertex indices must be between 1 and `n_vertices`.");
    }
    if (!std::isfinite(w) || w < 0.0) {
      Rcpp::stop("Graph weights must be finite and non-negative.");
    }
    if (u != v && w > 0.0) {
      adjacency[static_cast<std::size_t>(u)][v] += w;
      adjacency[static_cast<std::size_t>(v)][u] += w;
    }
  }

  std::vector<WalkEdge> edges;
  std::vector<double> graph_degree(static_cast<std::size_t>(n), 0.0);
  std::vector<int> graph_edge_count(static_cast<std::size_t>(n), 0);
  double total_edge_weight = 0.0;
  for (int u = 0; u < n; ++u) {
    for (const auto& item : adjacency[static_cast<std::size_t>(u)]) {
      graph_degree[static_cast<std::size_t>(u)] += item.second;
      ++graph_edge_count[static_cast<std::size_t>(u)];
      if (u < item.first) {
        edges.push_back(WalkEdge{u, item.first, item.second});
        total_edge_weight += item.second;
      }
    }
  }

  if (!(total_edge_weight > 0.0)) {
    Rcpp::IntegerVector singleton(n);
    for (int i = 0; i < n; ++i) singleton[i] = i + 1;
    return Rcpp::List::create(
      Rcpp::Named("membership") = singleton,
      Rcpp::Named("modularity") = NA_REAL,
      Rcpp::Named("n_communities") = n,
      Rcpp::Named("best_merge_count") = 0,
      Rcpp::Named("merges") = Rcpp::IntegerMatrix(0, 2),
      Rcpp::Named("modularity_trace") = Rcpp::NumericVector::create(NA_REAL),
      Rcpp::Named("implementation") = "native_pons_latapy_walktrap"
    );
  }

  std::vector<std::vector<std::pair<int, double>>> walk_neighbors(
    static_cast<std::size_t>(n));
  for (int u = 0; u < n; ++u) {
    const auto& source = adjacency[static_cast<std::size_t>(u)];
    auto& target = walk_neighbors[static_cast<std::size_t>(u)];
    target.assign(source.begin(), source.end());
    std::sort(target.begin(), target.end());
  }

  // Walktrap adds a self-loop whose weight is the mean incident edge weight.
  std::vector<double> self_weight(static_cast<std::size_t>(n), 1.0);
  std::vector<double> walk_degree(static_cast<std::size_t>(n), 1.0);
  for (int u = 0; u < n; ++u) {
    if (graph_edge_count[static_cast<std::size_t>(u)] > 0) {
      self_weight[static_cast<std::size_t>(u)] =
        graph_degree[static_cast<std::size_t>(u)] /
        static_cast<double>(graph_edge_count[static_cast<std::size_t>(u)]);
    }
    walk_degree[static_cast<std::size_t>(u)] =
      graph_degree[static_cast<std::size_t>(u)] + self_weight[static_cast<std::size_t>(u)];
  }
  std::vector<double> inverse_walk_degree(static_cast<std::size_t>(n));
  for (int u = 0; u < n; ++u) {
    inverse_walk_degree[static_cast<std::size_t>(u)] =
      1.0 / walk_degree[static_cast<std::size_t>(u)];
  }

  std::vector<double> probability(dense_entries, 0.0);
  std::vector<double> next(dense_entries, 0.0);
  for (int i = 0; i < n; ++i) {
    probability[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) +
      static_cast<std::size_t>(i)] = 1.0;
  }
  for (int step = 0; step < steps; ++step) {
    std::fill(next.begin(), next.end(), 0.0);
    for (int source = 0; source < n; ++source) {
      const std::size_t row = static_cast<std::size_t>(source) *
        static_cast<std::size_t>(n);
      for (int state = 0; state < n; ++state) {
        const double mass = probability[row + static_cast<std::size_t>(state)];
        if (mass == 0.0) continue;
        const double normalized = mass / walk_degree[static_cast<std::size_t>(state)];
        next[row + static_cast<std::size_t>(state)] +=
          normalized * self_weight[static_cast<std::size_t>(state)];
        for (const auto& item : walk_neighbors[static_cast<std::size_t>(state)]) {
          next[row + static_cast<std::size_t>(item.first)] += normalized * item.second;
        }
      }
    }
    probability.swap(next);
  }

  std::vector<std::vector<int>> support(static_cast<std::size_t>(n));
  for (int source = 0; source < n; ++source) {
    const std::size_t row = static_cast<std::size_t>(source) *
      static_cast<std::size_t>(n);
    for (int state = 0; state < n; ++state) {
      if (probability[row + static_cast<std::size_t>(state)] != 0.0) {
        support[static_cast<std::size_t>(source)].push_back(state);
      }
    }
  }

  const int max_communities = 2 * n - 1;
  std::vector<WalkCommunity> communities(static_cast<std::size_t>(max_communities));
  for (int i = 0; i < n; ++i) {
    WalkCommunity& community = communities[static_cast<std::size_t>(i)];
    community.active = true;
    community.size = 1;
    community.probability_slot = i;
    community.degree = graph_degree[static_cast<std::size_t>(i)];
    community.neighbors.reserve(adjacency[static_cast<std::size_t>(i)].size() * 2U + 1U);
  }

  std::priority_queue<WalkCandidate,
                      std::vector<WalkCandidate>,
                      WalkCandidateGreater> heap;
  for (const WalkEdge& edge : edges) {
    const double distance = walk_distance_squared(
      probability, n, edge.u, edge.v, inverse_walk_degree, support);
    const double delta = distance / (2.0 * static_cast<double>(n));
    communities[static_cast<std::size_t>(edge.u)].neighbors.emplace(
      edge.v, WalkNeighbor{delta, edge.weight});
    communities[static_cast<std::size_t>(edge.v)].neighbors.emplace(
      edge.u, WalkNeighbor{delta, edge.weight});
    heap.push(WalkCandidate{delta, edge.u, edge.v});
  }

  const double two_m = 2.0 * total_edge_weight;
  double modularity = 0.0;
  for (const double degree : graph_degree) {
    modularity -= degree * degree / (two_m * two_m);
  }
  double best_modularity = modularity;
  int best_merge_count = 0;
  std::vector<double> modularity_trace;
  modularity_trace.reserve(static_cast<std::size_t>(n));
  modularity_trace.push_back(modularity);
  std::vector<std::pair<int, int>> merges;
  merges.reserve(static_cast<std::size_t>(n - 1));

  int next_id = n;
  while (!heap.empty()) {
    const WalkCandidate candidate = heap.top();
    heap.pop();
    const int left = candidate.left;
    const int right = candidate.right;
    if (left < 0 || right < 0 || left >= next_id || right >= next_id) continue;
    WalkCommunity& left_community = communities[static_cast<std::size_t>(left)];
    WalkCommunity& right_community = communities[static_cast<std::size_t>(right)];
    if (!left_community.active || !right_community.active) continue;
    const auto current = left_community.neighbors.find(right);
    if (current == left_community.neighbors.end()) continue;
    const double tolerance = 1e-12 * std::max(1.0, std::abs(current->second.delta));
    if (std::abs(current->second.delta - candidate.delta) > tolerance) continue;

    const WalkNeighbor between = current->second;
    const int merged = next_id++;
    WalkCommunity& merged_community = communities[static_cast<std::size_t>(merged)];
    merged_community.active = true;
    merged_community.size = left_community.size + right_community.size;
    merged_community.degree = left_community.degree + right_community.degree;
    merged_community.probability_slot = left_community.probability_slot;

    const int left_size = left_community.size;
    const int right_size = right_community.size;
    const int left_slot = left_community.probability_slot;
    const int right_slot = right_community.probability_slot;
    const std::size_t left_offset = static_cast<std::size_t>(left_slot) *
      static_cast<std::size_t>(n);
    const std::size_t right_offset = static_cast<std::size_t>(right_slot) *
      static_cast<std::size_t>(n);
    const double inverse_size = 1.0 / static_cast<double>(left_size + right_size);
    std::vector<int> merged_support;
    const auto& left_support = support[static_cast<std::size_t>(left_slot)];
    const auto& right_support = support[static_cast<std::size_t>(right_slot)];
    merged_support.reserve(left_support.size() + right_support.size());
    std::set_union(left_support.begin(), left_support.end(),
                   right_support.begin(), right_support.end(),
                   std::back_inserter(merged_support));
    for (const int k : merged_support) {
      probability[left_offset + static_cast<std::size_t>(k)] =
        (static_cast<double>(left_size) *
           probability[left_offset + static_cast<std::size_t>(k)] +
         static_cast<double>(right_size) *
           probability[right_offset + static_cast<std::size_t>(k)]) * inverse_size;
    }
    support[static_cast<std::size_t>(left_slot)] = std::move(merged_support);

    std::vector<int> adjacent;
    adjacent.reserve(left_community.neighbors.size() + right_community.neighbors.size());
    for (const auto& item : left_community.neighbors) {
      if (item.first != right) adjacent.push_back(item.first);
    }
    for (const auto& item : right_community.neighbors) {
      if (item.first != left) adjacent.push_back(item.first);
    }
    std::sort(adjacent.begin(), adjacent.end());
    adjacent.erase(std::unique(adjacent.begin(), adjacent.end()), adjacent.end());
    merged_community.neighbors.reserve(adjacent.size() * 2U + 1U);

    for (const int other : adjacent) {
      WalkCommunity& other_community = communities[static_cast<std::size_t>(other)];
      if (!other_community.active) continue;
      const auto left_it = left_community.neighbors.find(other);
      const auto right_it = right_community.neighbors.find(other);
      const bool has_left = left_it != left_community.neighbors.end();
      const bool has_right = right_it != right_community.neighbors.end();
      double delta = 0.0;
      if (has_left && has_right) {
        const double other_size = static_cast<double>(other_community.size);
        delta =
          ((static_cast<double>(left_size) + other_size) * left_it->second.delta +
           (static_cast<double>(right_size) + other_size) * right_it->second.delta -
           other_size * between.delta) /
          (static_cast<double>(left_size + right_size) + other_size);
      } else {
        const double distance = walk_distance_squared(
          probability, n, merged_community.probability_slot,
          other_community.probability_slot, inverse_walk_degree, support);
        delta = (1.0 / static_cast<double>(n)) *
          (static_cast<double>(merged_community.size) *
           static_cast<double>(other_community.size) /
           static_cast<double>(merged_community.size + other_community.size)) * distance;
      }
      if (delta < 0.0 && delta > -1e-14) delta = 0.0;
      const double edge_weight =
        (has_left ? left_it->second.edge_weight : 0.0) +
        (has_right ? right_it->second.edge_weight : 0.0);
      merged_community.neighbors.emplace(other, WalkNeighbor{delta, edge_weight});
      other_community.neighbors.erase(left);
      other_community.neighbors.erase(right);
      other_community.neighbors.emplace(merged, WalkNeighbor{delta, edge_weight});
      heap.push(WalkCandidate{delta, std::min(merged, other), std::max(merged, other)});
    }

    modularity += between.edge_weight / total_edge_weight -
      left_community.degree * right_community.degree * 2.0 / (two_m * two_m);
    merges.emplace_back(left, right);
    modularity_trace.push_back(modularity);
    if (modularity > best_modularity + 1e-12) {
      best_modularity = modularity;
      best_merge_count = static_cast<int>(merges.size());
    }

    left_community.active = false;
    right_community.active = false;
    left_community.neighbors.clear();
    right_community.neighbors.clear();
  }

  const std::vector<int> membership0 = replay_partition(n, merges, best_merge_count);
  Rcpp::IntegerVector membership(n);
  int n_communities = 0;
  for (int i = 0; i < n; ++i) {
    membership[i] = membership0[static_cast<std::size_t>(i)] + 1;
    n_communities = std::max(n_communities, membership[i]);
  }
  Rcpp::IntegerMatrix merge_matrix(static_cast<int>(merges.size()), 2);
  for (std::size_t i = 0; i < merges.size(); ++i) {
    merge_matrix(static_cast<int>(i), 0) = merges[i].first + 1;
    merge_matrix(static_cast<int>(i), 1) = merges[i].second + 1;
  }

  return Rcpp::List::create(
    Rcpp::Named("membership") = membership,
    Rcpp::Named("modularity") = best_modularity,
    Rcpp::Named("n_communities") = n_communities,
    Rcpp::Named("best_merge_count") = best_merge_count,
    Rcpp::Named("merges") = merge_matrix,
    Rcpp::Named("modularity_trace") = Rcpp::wrap(modularity_trace),
    Rcpp::Named("implementation") = "native_pons_latapy_walktrap"
  );
}
