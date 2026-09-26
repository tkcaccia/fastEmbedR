/*
 * SPDX-FileCopyrightText: 2015-2026 Meta Platforms, Inc. and affiliates
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT
 *
 * Compact HNSW implementation distilled from the algorithmic organization in
 * FAISS 1.14.3, commit 0ca9df4792b173d573044ee14ca0704780176e82.
 * Upstream files: faiss/impl/HNSW.{h,cpp} and faiss/IndexHNSW.{h,cpp}.
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) 2026 Stefano Cacciatore.
 *
 * Licensed under the MIT License. The FAISS copyright and MIT license must be
 * retained with redistributed derivatives of this file.
 */

#include <Rcpp.h>

#include "native_knn_common.h"
#include "native_knn_hnsw_tuning.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct NodeDistance {
  float distance;
  int id;
};

struct CloserFirst {
  bool operator()(const NodeDistance& a, const NodeDistance& b) const {
    if (a.distance != b.distance) return a.distance > b.distance;
    return a.id > b.id;
  }
};

struct FartherFirst {
  bool operator()(const NodeDistance& a, const NodeDistance& b) const {
    if (a.distance != b.distance) return a.distance < b.distance;
    return a.id < b.id;
  }
};

inline bool closer(const NodeDistance& a, const NodeDistance& b) {
  return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
}

class CompactHNSW {
 public:
  CompactHNSW(std::vector<float> data, int n, int p, int m, int ef_construction, int ef_search)
      : data_(std::move(data)), n_(n), p_(p), m_(m), ef_construction_(ef_construction), ef_search_(ef_search) {
    generate_levels();
    allocate_graph();
  }

  void build(int n_threads) {
    if (n_ == 0) return;
    n_threads = fastembedr::adaptive_worker_count(n_threads, n_);
    entry_point_ = 0;
    current_max_level_ = levels_[0];
    BuildScratch scratch(n_, ef_construction_, 2 * m_);
    BuildParallelState parallel(this, n_threads, 2 * m_);
    for (int point = 1; point < n_; ++point) {
      add_point(point, scratch, parallel);
    }
    // Distances accelerate reciprocal pruning only; queries need the graph IDs.
    std::vector<float>().swap(neighbor_distances_);
  }

  void search_all(int k, int n_threads, std::vector<int>& output_ids, std::vector<float>& output_distances) const {
    output_ids.assign(static_cast<std::size_t>(n_) * k, -1);
    output_distances.assign(static_cast<std::size_t>(n_) * k, std::numeric_limits<float>::infinity());
    std::atomic<int> next(0);
    n_threads = fastembedr::adaptive_worker_count(n_threads, n_);
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (int thread_id = 0; thread_id < n_threads; ++thread_id) {
      workers.emplace_back([&, thread_id]() {
        (void)thread_id;
        VisitTable visited(n_);
        while (true) {
          int query = next.fetch_add(1, std::memory_order_relaxed);
          if (query >= n_) break;
          std::vector<NodeDistance> candidates = search_query(query, std::max(k + 1, ef_search_), visited);
          int used = 0;
          for (const NodeDistance& candidate : candidates) {
            if (candidate.id == query) continue;
            std::size_t pos = static_cast<std::size_t>(query) * k + used;
            output_ids[pos] = candidate.id;
            output_distances[pos] = candidate.distance;
            if (++used == k) break;
          }
          if (used < k) exact_fill(query, k, used, output_ids, output_distances);
        }
      });
    }
    for (std::thread& worker : workers) worker.join();
  }

  void search_queries(const std::vector<float>& queries,
                      int n_queries,
                      int k,
                      int n_threads,
                      std::vector<int>& output_ids,
                      std::vector<float>& output_distances) const {
    output_ids.assign(static_cast<std::size_t>(n_queries) * k, -1);
    output_distances.assign(
      static_cast<std::size_t>(n_queries) * k,
      std::numeric_limits<float>::infinity()
    );
    std::atomic<int> next(0);
    n_threads = fastembedr::adaptive_worker_count(n_threads, n_queries);
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (int thread_id = 0; thread_id < n_threads; ++thread_id) {
      workers.emplace_back([&, thread_id]() {
        (void)thread_id;
        VisitTable visited(n_);
        while (true) {
          const int query_id = next.fetch_add(1, std::memory_order_relaxed);
          if (query_id >= n_queries) break;
          const float* query = queries.data() +
            static_cast<std::size_t>(query_id) * p_;
          std::vector<NodeDistance> candidates = search_vector(
            query, std::max(k, ef_search_), visited
          );
          if (static_cast<int>(candidates.size()) < k) {
            candidates = exact_query(query, k);
          }
          for (int rank = 0; rank < k; ++rank) {
            const std::size_t pos = static_cast<std::size_t>(query_id) * k + rank;
            output_ids[pos] = candidates[static_cast<std::size_t>(rank)].id;
            output_distances[pos] = candidates[static_cast<std::size_t>(rank)].distance;
          }
        }
      });
    }
    for (std::thread& worker : workers) worker.join();
  }

  std::size_t graph_bytes() const {
    return neighbors_.size() * sizeof(std::int32_t) + levels_.size() * sizeof(int) +
      node_offsets_.size() * sizeof(std::size_t) + level_offsets_.size() * sizeof(std::size_t) +
      counts_.size() * sizeof(std::uint16_t);
  }

 private:
  struct VisitTable {
    explicit VisitTable(int n) : marks(n, 0), generation(1) {}
    void reset() {
      if (++generation == 0) { std::fill(marks.begin(), marks.end(), 0); generation = 1; }
    }
    bool set(int id) {
      if (marks[id] == generation) return false;
      marks[id] = generation;
      return true;
    }
    std::vector<std::uint32_t> marks;
    std::uint32_t generation;
  };

  struct ReciprocalScratch {
    explicit ReciprocalScratch(int max_neighbors) {
      candidates.reserve(static_cast<std::size_t>(max_neighbors) + 1u);
      selected.reserve(static_cast<std::size_t>(max_neighbors));
      selected_distances.reserve(static_cast<std::size_t>(max_neighbors));
      rejected.reserve(static_cast<std::size_t>(max_neighbors) + 1u);
    }

    std::vector<NodeDistance> candidates;
    std::vector<int> selected;
    std::vector<float> selected_distances;
    std::vector<NodeDistance> rejected;
  };

  struct BuildScratch {
    BuildScratch(int n, int ef, int max_neighbors) : visited(n) {
      candidate_heap.reserve(static_cast<std::size_t>(ef) + max_neighbors + 1u);
      result_heap.reserve(static_cast<std::size_t>(ef) + 1u);
      layer_candidates.reserve(static_cast<std::size_t>(ef));
      selected.reserve(static_cast<std::size_t>(max_neighbors));
      selected_distances.reserve(static_cast<std::size_t>(max_neighbors));
      rejected.reserve(static_cast<std::size_t>(ef));
    }

    VisitTable visited;
    std::vector<NodeDistance> candidate_heap;
    std::vector<NodeDistance> result_heap;
    std::vector<NodeDistance> layer_candidates;
    std::vector<int> selected;
    std::vector<float> selected_distances;
    std::vector<NodeDistance> rejected;
  };

  // Reciprocal links from one insertion touch distinct graph rows, so a barrier
  // can parallelize them without changing the sequential point insertion order.
  struct BuildParallelState {
    BuildParallelState(CompactHNSW* owner,
                       int n_threads,
                       int max_neighbors)
        : owner(owner),
          n_threads(std::max(1, std::min(n_threads, owner->n_))) {
      reciprocal_scratch.reserve(static_cast<std::size_t>(this->n_threads));
      for (int i = 0; i < this->n_threads; ++i) {
        reciprocal_scratch.emplace_back(max_neighbors);
      }
      workers.reserve(static_cast<std::size_t>(this->n_threads - 1));
      for (int thread_id = 1; thread_id < this->n_threads; ++thread_id) {
        workers.emplace_back([this, thread_id]() {
          std::uint64_t observed_generation = 0;
          while (true) {
            {
              std::unique_lock<std::mutex> lock(mutex);
              work_ready.wait(lock, [&]() {
                return stopping || generation != observed_generation;
              });
              if (stopping) return;
              observed_generation = generation;
            }
            process_stripe(thread_id);
            {
              std::lock_guard<std::mutex> lock(mutex);
              --workers_remaining;
              if (workers_remaining == 0) work_done.notify_one();
            }
          }
        });
      }
    }

    ~BuildParallelState() {
      {
        std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
      }
      work_ready.notify_all();
      for (std::thread& worker : workers) worker.join();
    }

    void add(const std::vector<int>& nodes, int point_id, int level) {
      if (workers.empty() || nodes.size() < 8u) {
        ReciprocalScratch& scratch = reciprocal_scratch.front();
        for (int node : nodes) {
          owner->add_reciprocal(node, point_id, level, scratch);
        }
        return;
      }
      {
        std::lock_guard<std::mutex> lock(mutex);
        task_nodes = &nodes;
        task_point = point_id;
        task_level = level;
        workers_remaining = static_cast<int>(workers.size());
        ++generation;
      }
      work_ready.notify_all();
      process_stripe(0);
      std::unique_lock<std::mutex> lock(mutex);
      work_done.wait(lock, [&]() { return workers_remaining == 0; });
    }

   private:
    void process_stripe(int thread_id) {
      ReciprocalScratch& scratch =
        reciprocal_scratch[static_cast<std::size_t>(thread_id)];
      for (std::size_t i = static_cast<std::size_t>(thread_id);
           i < task_nodes->size();
           i += static_cast<std::size_t>(n_threads)) {
        owner->add_reciprocal(
          (*task_nodes)[i],
          task_point,
          task_level,
          scratch
        );
      }
    }

    CompactHNSW* owner;
    int n_threads;
    std::vector<ReciprocalScratch> reciprocal_scratch;
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable work_ready;
    std::condition_variable work_done;
    std::uint64_t generation = 0;
    int workers_remaining = 0;
    const std::vector<int>* task_nodes = nullptr;
    int task_point = -1;
    int task_level = 0;
    bool stopping = false;
  };

  std::vector<float> data_;
  int n_;
  int p_;
  int m_;
  int ef_construction_;
  int ef_search_;
  int entry_point_ = -1;
  int current_max_level_ = -1;
  std::vector<int> levels_;
  std::vector<std::size_t> node_offsets_;
  std::vector<std::size_t> level_offsets_;
  std::vector<std::uint16_t> counts_;
  std::vector<std::int32_t> neighbors_;
  std::vector<float> neighbor_distances_;

  inline const float* point(int id) const { return data_.data() + static_cast<std::size_t>(id) * p_; }

  inline float distance(const float* a, const float* b) const {
    return fastembedr::squared_l2_distance(a, b, p_);
  }

  inline float distance(int a, int b) const { return distance(point(a), point(b)); }

  inline bool distance_below(int a, int b, float threshold) const {
    const float* lhs = point(a);
    const float* rhs = point(b);
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    int d = 0;
    for (; d + 15 < p_; d += 16) {
      float x0 = lhs[d] - rhs[d]; float x1 = lhs[d + 1] - rhs[d + 1];
      float x2 = lhs[d + 2] - rhs[d + 2]; float x3 = lhs[d + 3] - rhs[d + 3];
      float x4 = lhs[d + 4] - rhs[d + 4]; float x5 = lhs[d + 5] - rhs[d + 5];
      float x6 = lhs[d + 6] - rhs[d + 6]; float x7 = lhs[d + 7] - rhs[d + 7];
      float x8 = lhs[d + 8] - rhs[d + 8]; float x9 = lhs[d + 9] - rhs[d + 9];
      float xa = lhs[d + 10] - rhs[d + 10]; float xb = lhs[d + 11] - rhs[d + 11];
      float xc = lhs[d + 12] - rhs[d + 12]; float xd = lhs[d + 13] - rhs[d + 13];
      float xe = lhs[d + 14] - rhs[d + 14]; float xf = lhs[d + 15] - rhs[d + 15];
      sum0 += x0*x0 + x4*x4 + x8*x8 + xc*xc;
      sum1 += x1*x1 + x5*x5 + x9*x9 + xd*xd;
      sum2 += x2*x2 + x6*x6 + xa*xa + xe*xe;
      sum3 += x3*x3 + x7*x7 + xb*xb + xf*xf;
      if ((sum0 + sum1) + (sum2 + sum3) >= threshold) return false;
    }
    float sum = (sum0 + sum1) + (sum2 + sum3);
    for (; d < p_; ++d) {
      float delta = lhs[d] - rhs[d];
      sum += delta * delta;
      if (sum >= threshold) return false;
    }
    return sum < threshold;
  }

  int capacity(int level) const { return level == 0 ? 2 * m_ : m_; }

  std::size_t level_index(int node, int level) const { return level_offsets_[node] + static_cast<std::size_t>(level); }

  std::size_t neighbor_offset(int node, int level) const {
    return node_offsets_[node] + (level == 0 ? 0u : static_cast<std::size_t>(2 * m_ + (level - 1) * m_));
  }

  void generate_levels() {
    levels_.resize(n_);
    std::mt19937 generator(12345u);
    std::uniform_real_distribution<double> uniform(std::nextafter(0.0, 1.0), 1.0);
    const double level_multiplier = 1.0 / std::log(static_cast<double>(m_));
    for (int i = 0; i < n_; ++i) levels_[i] = static_cast<int>(-std::log(uniform(generator)) * level_multiplier);
  }

  void allocate_graph() {
    node_offsets_.resize(static_cast<std::size_t>(n_) + 1u, 0u);
    level_offsets_.resize(static_cast<std::size_t>(n_) + 1u, 0u);
    for (int i = 0; i < n_; ++i) {
      node_offsets_[i + 1] = node_offsets_[i] + static_cast<std::size_t>(2 * m_ + levels_[i] * m_);
      level_offsets_[i + 1] = level_offsets_[i] + static_cast<std::size_t>(levels_[i] + 1);
    }
    neighbors_.assign(node_offsets_.back(), -1);
    neighbor_distances_.assign(
      node_offsets_.back(),
      std::numeric_limits<float>::infinity()
    );
    counts_.assign(level_offsets_.back(), 0);
  }

  std::pair<const std::int32_t*, int> neighbor_range(int node, int level) const {
    std::size_t offset = neighbor_offset(node, level);
    return {neighbors_.data() + offset, counts_[level_index(node, level)]};
  }

  std::pair<std::int32_t*, std::uint16_t*> mutable_neighbor_range(int node, int level) {
    std::size_t offset = neighbor_offset(node, level);
    return {neighbors_.data() + offset, &counts_[level_index(node, level)]};
  }

  int greedy_search(const float* query, int entry, int level, float& entry_distance) const {
    bool changed = true;
    while (changed) {
      changed = false;
      auto range = neighbor_range(entry, level);
      for (int j = 0; j < range.second; ++j) {
        int candidate = range.first[j];
        float candidate_distance = distance(query, point(candidate));
        NodeDistance candidate_pair{candidate_distance, candidate};
        NodeDistance current_pair{entry_distance, entry};
        if (closer(candidate_pair, current_pair)) {
          entry = candidate;
          entry_distance = candidate_distance;
          changed = true;
        }
      }
    }
    return entry;
  }

  std::vector<NodeDistance> search_layer(const float* query, int entry, int ef, int level, VisitTable& visited) const {
    std::priority_queue<NodeDistance, std::vector<NodeDistance>, CloserFirst> candidates;
    std::priority_queue<NodeDistance, std::vector<NodeDistance>, FartherFirst> results;
    visited.reset();
    float initial_distance = distance(query, point(entry));
    NodeDistance initial{initial_distance, entry};
    candidates.push(initial); results.push(initial); visited.set(entry);
    while (!candidates.empty()) {
      NodeDistance current = candidates.top();
      NodeDistance worst = results.top();
      if (results.size() >= static_cast<std::size_t>(ef) && closer(worst, current)) break;
      candidates.pop();
      auto range = neighbor_range(current.id, level);
      for (int j = 0; j < range.second; ++j) {
        int candidate_id = range.first[j];
        if (!visited.set(candidate_id)) continue;
        NodeDistance candidate{distance(query, point(candidate_id)), candidate_id};
        if (results.size() < static_cast<std::size_t>(ef) || closer(candidate, results.top())) {
          candidates.push(candidate);
          results.push(candidate);
          if (results.size() > static_cast<std::size_t>(ef)) results.pop();
        }
      }
    }
    std::vector<NodeDistance> output;
    output.reserve(results.size());
    while (!results.empty()) { output.push_back(results.top()); results.pop(); }
    std::sort(output.begin(), output.end(), closer);
    return output;
  }

  void search_layer_build(const float* query,
                          int entry,
                          int ef,
                          int level,
                          BuildScratch& scratch) const {
    std::vector<NodeDistance>& candidates = scratch.candidate_heap;
    std::vector<NodeDistance>& results = scratch.result_heap;
    std::vector<NodeDistance>& output = scratch.layer_candidates;
    candidates.clear();
    results.clear();
    output.clear();
    scratch.visited.reset();
    float initial_distance = distance(query, point(entry));
    NodeDistance initial{initial_distance, entry};
    candidates.push_back(initial);
    std::push_heap(candidates.begin(), candidates.end(), CloserFirst{});
    results.push_back(initial);
    std::push_heap(results.begin(), results.end(), FartherFirst{});
    scratch.visited.set(entry);
    while (!candidates.empty()) {
      NodeDistance current = candidates.front();
      NodeDistance worst = results.front();
      if (results.size() >= static_cast<std::size_t>(ef) &&
          closer(worst, current)) {
        break;
      }
      std::pop_heap(candidates.begin(), candidates.end(), CloserFirst{});
      candidates.pop_back();
      auto range = neighbor_range(current.id, level);
      for (int j = 0; j < range.second; ++j) {
        int candidate_id = range.first[j];
        if (!scratch.visited.set(candidate_id)) continue;
        NodeDistance candidate{
          distance(query, point(candidate_id)),
          candidate_id
        };
        if (results.size() < static_cast<std::size_t>(ef) ||
            closer(candidate, results.front())) {
          candidates.push_back(candidate);
          std::push_heap(candidates.begin(), candidates.end(), CloserFirst{});
          results.push_back(candidate);
          std::push_heap(results.begin(), results.end(), FartherFirst{});
          if (results.size() > static_cast<std::size_t>(ef)) {
            std::pop_heap(results.begin(), results.end(), FartherFirst{});
            results.pop_back();
          }
        }
      }
    }
    output.reserve(results.size());
    while (!results.empty()) {
      output.push_back(results.front());
      std::pop_heap(results.begin(), results.end(), FartherFirst{});
      results.pop_back();
    }
    std::sort(output.begin(), output.end(), closer);
  }

  void select_diverse_into(int query_id,
                           const std::vector<NodeDistance>& candidates,
                           int max_size,
                           std::vector<int>& selected,
                           std::vector<float>& selected_distances,
                           std::vector<NodeDistance>& rejected) const {
    selected.clear();
    selected_distances.clear();
    rejected.clear();
    for (const NodeDistance& candidate : candidates) {
      if (candidate.id == query_id) continue;
      bool good = true;
      for (int other : selected) {
        if (distance_below(candidate.id, other, candidate.distance)) {
          good = false;
          break;
        }
      }
      if (good) {
        selected.push_back(candidate.id);
        selected_distances.push_back(candidate.distance);
        if (static_cast<int>(selected.size()) == max_size) break;
      } else {
        rejected.push_back(candidate);
      }
    }
    for (const NodeDistance& candidate : rejected) {
      if (static_cast<int>(selected.size()) == max_size) break;
      selected.push_back(candidate.id);
      selected_distances.push_back(candidate.distance);
    }
  }

  void replace_neighbors(int node,
                         int level,
                         const std::vector<int>& selected,
                         const std::vector<float>& selected_distances) {
    auto range = mutable_neighbor_range(node, level);
    const std::size_t offset = neighbor_offset(node, level);
    int cap = capacity(level);
    int size = std::min(cap, static_cast<int>(selected.size()));
    for (int i = 0; i < size; ++i) {
      range.first[i] = selected[i];
      neighbor_distances_[offset + static_cast<std::size_t>(i)] =
        selected_distances[static_cast<std::size_t>(i)];
    }
    for (int i = size; i < cap; ++i) {
      range.first[i] = -1;
      neighbor_distances_[offset + static_cast<std::size_t>(i)] =
        std::numeric_limits<float>::infinity();
    }
    *range.second = static_cast<std::uint16_t>(size);
  }

  void add_reciprocal(int node,
                      int other,
                      int level,
                      ReciprocalScratch& scratch) {
    auto range = mutable_neighbor_range(node, level);
    int count = *range.second;
    for (int i = 0; i < count; ++i) if (range.first[i] == other) return;
    int cap = capacity(level);
    if (count < cap) {
      range.first[count] = other;
      neighbor_distances_[
        neighbor_offset(node, level) + static_cast<std::size_t>(count)
      ] = distance(node, other);
      *range.second = static_cast<std::uint16_t>(count + 1);
      return;
    }
    std::vector<NodeDistance>& candidates = scratch.candidates;
    candidates.clear();
    const std::size_t offset = neighbor_offset(node, level);
    for (int i = 0; i < count; ++i) {
      candidates.push_back({
        neighbor_distances_[offset + static_cast<std::size_t>(i)],
        range.first[i]
      });
    }
    candidates.push_back({distance(node, other), other});
    std::sort(candidates.begin(), candidates.end(), closer);
    select_diverse_into(
      node,
      candidates,
      cap,
      scratch.selected,
      scratch.selected_distances,
      scratch.rejected
    );
    replace_neighbors(
      node,
      level,
      scratch.selected,
      scratch.selected_distances
    );
  }

  void add_point(int point_id,
                 BuildScratch& scratch,
                 BuildParallelState& parallel) {
    int nearest = entry_point_;
    float nearest_distance = distance(point_id, nearest);
    for (int level = current_max_level_; level > levels_[point_id]; --level)
      nearest = greedy_search(point(point_id), nearest, level, nearest_distance);
    for (int level = std::min(levels_[point_id], current_max_level_); level >= 0; --level) {
      search_layer_build(
        point(point_id),
        nearest,
        ef_construction_,
        level,
        scratch
      );
      select_diverse_into(
        point_id,
        scratch.layer_candidates,
        capacity(level),
        scratch.selected,
        scratch.selected_distances,
        scratch.rejected
      );
      replace_neighbors(
        point_id,
        level,
        scratch.selected,
        scratch.selected_distances
      );
      parallel.add(scratch.selected, point_id, level);
      if (!scratch.layer_candidates.empty()) {
        nearest = scratch.layer_candidates.front().id;
        nearest_distance = scratch.layer_candidates.front().distance;
      }
    }
    if (levels_[point_id] > current_max_level_) {
      entry_point_ = point_id;
      current_max_level_ = levels_[point_id];
    }
  }

  std::vector<NodeDistance> search_query(int query_id, int ef, VisitTable& visited) const {
    return search_vector(point(query_id), ef, visited);
  }

  std::vector<NodeDistance> search_vector(const float* query,
                                          int ef,
                                          VisitTable& visited) const {
    int nearest = entry_point_;
    float nearest_distance = distance(query, point(nearest));
    for (int level = current_max_level_; level >= 1; --level)
      nearest = greedy_search(query, nearest, level, nearest_distance);
    return search_layer(query, nearest, ef, 0, visited);
  }

  std::vector<NodeDistance> exact_query(const float* query, int k) const {
    std::priority_queue<NodeDistance, std::vector<NodeDistance>, FartherFirst> heap;
    for (int candidate = 0; candidate < n_; ++candidate) {
      NodeDistance value{distance(query, point(candidate)), candidate};
      if (heap.size() < static_cast<std::size_t>(k) || closer(value, heap.top())) {
        heap.push(value);
        if (heap.size() > static_cast<std::size_t>(k)) heap.pop();
      }
    }
    std::vector<NodeDistance> exact;
    exact.reserve(static_cast<std::size_t>(k));
    while (!heap.empty()) {
      exact.push_back(heap.top());
      heap.pop();
    }
    std::sort(exact.begin(), exact.end(), closer);
    return exact;
  }

  void exact_fill(int query, int k, int used, std::vector<int>& output_ids, std::vector<float>& output_distances) const {
    std::priority_queue<NodeDistance, std::vector<NodeDistance>, FartherFirst> heap;
    for (int candidate = 0; candidate < n_; ++candidate) {
      if (candidate == query) continue;
      NodeDistance value{distance(query, candidate), candidate};
      if (heap.size() < static_cast<std::size_t>(k) || closer(value, heap.top())) {
        heap.push(value);
        if (heap.size() > static_cast<std::size_t>(k)) heap.pop();
      }
    }
    std::vector<NodeDistance> exact;
    while (!heap.empty()) { exact.push_back(heap.top()); heap.pop(); }
    std::sort(exact.begin(), exact.end(), closer);
    (void)used;
    for (int rank = 0; rank < k; ++rank) {
      std::size_t pos = static_cast<std::size_t>(query) * k + rank;
      output_ids[pos] = exact[rank].id;
      output_distances[pos] = exact[rank].distance;
    }
  }
};

} // namespace

Rcpp::List native_hnsw_knn_impl(SEXP data_sexp,
                                int k,
                                int n_threads,
                                const std::string& metric_name,
                                double target_recall) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  const fastembedr::KnnMetric metric = fastembedr::parse_knn_metric(metric_name);
  fastembedr::FloatMatrix input = fastembedr::matrix_to_row_major_float(data_sexp, metric);
  fastembedr::require_finite_matrix(input);
  const auto converted = Clock::now();
  const int n = input.nrow;
  const int p = input.ncol;
  if (n < 2 || p < 1 || k < 1 || k >= n) Rcpp::stop("invalid HNSW input");
  const fastembedr::HnswTuning tuning = fastembedr::tune_native_hnsw(
    n, p, k, metric, target_recall
  );
  const int m = tuning.m;
  const int ef_construction = tuning.ef_construction;
  const int ef_search = tuning.ef_search;
  CompactHNSW index(std::move(input.values), n, p, m, ef_construction, ef_search);
  index.build(n_threads);
  const auto built = Clock::now();
  std::vector<int> ids;
  std::vector<float> squared_distances;
  index.search_all(k, n_threads, ids, squared_distances);
  const auto searched = Clock::now();
  Rcpp::IntegerMatrix indices(n, k);
  Rcpp::NumericMatrix distances(n, k);
  for (int i = 0; i < n; ++i) for (int j = 0; j < k; ++j) {
    std::size_t pos = static_cast<std::size_t>(i) * k + j;
    indices(i, j) = ids[pos] + 1;
    distances(i, j) = fastembedr::output_distance(squared_distances[pos], metric);
  }
  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("backend") = "cpu",
    Rcpp::Named("method") = "native_hnsw",
    Rcpp::Named("metric") = metric_name,
    Rcpp::Named("requested_recall") = target_recall,
    Rcpp::Named("target_recall") = target_recall,
    Rcpp::Named("recall_audited") = false,
    Rcpp::Named("target_met") = Rcpp::LogicalVector::get_na(),
    Rcpp::Named("recall_status") =
      "calibration_informed_not_runtime_audited",
    Rcpp::Named("tuning_policy") = "shape_metric_k_recall99",
    Rcpp::Named("tuning_rule") = tuning.rule,
    Rcpp::Named("tuning_shape_group") = tuning.shape,
    Rcpp::Named("tuning_k_bucket") = tuning.k_bucket,
    Rcpp::Named("tuning_reference_target_met") =
      tuning.reference_target_met,
    Rcpp::Named("tuning_source") =
      "faissR_0.99.48_09f4c88fe8af",
    Rcpp::Named("M") = m,
    Rcpp::Named("efConstruction") = ef_construction,
    Rcpp::Named("efSearch") = ef_search,
    Rcpp::Named("graph_bytes") = static_cast<double>(index.graph_bytes()),
    Rcpp::Named("timing") = Rcpp::NumericVector::create(
      Rcpp::Named("convert") = std::chrono::duration<double>(converted - start).count(),
      Rcpp::Named("build") = std::chrono::duration<double>(built - converted).count(),
      Rcpp::Named("query") = std::chrono::duration<double>(searched - built).count()
    )
  );
  result.attr("backend") = "cpu";
  result.attr("method") = "native_hnsw";
  result.attr("recall_audited") = false;
  result.attr("metric") = metric_name;
  return result;
}

Rcpp::List native_hnsw_query_impl(SEXP data_sexp,
                                  SEXP query_sexp,
                                  int k,
                                  int n_threads,
                                  const std::string& metric_name,
                                  double target_recall) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  const fastembedr::KnnMetric metric = fastembedr::parse_knn_metric(metric_name);
  fastembedr::FloatMatrix input =
    fastembedr::matrix_to_row_major_float(data_sexp, metric);
  fastembedr::FloatMatrix query =
    fastembedr::matrix_to_row_major_float(query_sexp, metric);
  fastembedr::require_finite_matrix(input);
  fastembedr::require_finite_matrix(query);
  const auto converted = Clock::now();
  const int n = input.nrow;
  const int p = input.ncol;
  const int n_queries = query.nrow;
  if (n < 1 || n_queries < 1 || p < 1 || query.ncol != p || k < 1 || k > n) {
    Rcpp::stop("Invalid native HNSW query input.");
  }
  const fastembedr::HnswTuning tuning = fastembedr::tune_native_hnsw(
    n, p, k, metric, target_recall
  );
  const int m = tuning.m;
  const int ef_construction = tuning.ef_construction;
  const int ef_search = tuning.ef_search;
  CompactHNSW index(
    std::move(input.values), n, p, m, ef_construction, ef_search
  );
  index.build(n_threads);
  const auto built = Clock::now();
  std::vector<int> ids;
  std::vector<float> squared_distances;
  index.search_queries(
    query.values, n_queries, k, n_threads, ids, squared_distances
  );
  const auto searched = Clock::now();
  Rcpp::IntegerMatrix indices(n_queries, k);
  Rcpp::NumericMatrix distances(n_queries, k);
  for (int i = 0; i < n_queries; ++i) {
    for (int j = 0; j < k; ++j) {
      const std::size_t pos = static_cast<std::size_t>(i) * k + j;
      indices(i, j) = ids[pos] + 1;
      distances(i, j) = fastembedr::output_distance(
        squared_distances[pos], metric
      );
    }
  }
  Rcpp::List result = Rcpp::List::create(
    Rcpp::Named("indices") = indices,
    Rcpp::Named("distances") = distances,
    Rcpp::Named("backend") = "cpu",
    Rcpp::Named("method") = "native_hnsw_query",
    Rcpp::Named("metric") = metric_name,
    Rcpp::Named("requested_recall") = target_recall,
    Rcpp::Named("target_recall") = target_recall,
    Rcpp::Named("recall_audited") = false,
    Rcpp::Named("target_met") = Rcpp::LogicalVector::get_na(),
    Rcpp::Named("recall_status") =
      "calibration_informed_not_runtime_audited",
    Rcpp::Named("tuning_policy") = "shape_metric_k_recall99",
    Rcpp::Named("tuning_rule") = tuning.rule,
    Rcpp::Named("tuning_shape_group") = tuning.shape,
    Rcpp::Named("tuning_k_bucket") = tuning.k_bucket,
    Rcpp::Named("tuning_reference_target_met") =
      tuning.reference_target_met,
    Rcpp::Named("tuning_source") =
      "faissR_0.99.48_09f4c88fe8af",
    Rcpp::Named("M") = m,
    Rcpp::Named("efConstruction") = ef_construction,
    Rcpp::Named("efSearch") = ef_search,
    Rcpp::Named("graph_bytes") = static_cast<double>(index.graph_bytes()),
    Rcpp::Named("timing") = Rcpp::NumericVector::create(
      Rcpp::Named("convert") =
        std::chrono::duration<double>(converted - start).count(),
      Rcpp::Named("build") =
        std::chrono::duration<double>(built - converted).count(),
      Rcpp::Named("query") =
        std::chrono::duration<double>(searched - built).count()
    )
  );
  result.attr("backend") = "cpu";
  result.attr("method") = "native_hnsw_query";
  result.attr("recall_audited") = false;
  result.attr("metric") = metric_name;
  result.attr("exclude_self") = false;
  return result;
}
