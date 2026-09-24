/*
 * SPDX-FileCopyrightText: 2015-2026 Meta Platforms, Inc. and affiliates
 * SPDX-FileCopyrightText: 2024 Sydney Bach, The Solace Project
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT AND Apache-2.0
 *
 * Native Metal IVF-Flat search for fastEmbedR.
 *
 * Search organization is informed by:
 * - FAISS 1.14.3, commit 0ca9df4792b173d573044ee14ca0704780176e82
 *   Copyright (c) Meta Platforms, Inc. and affiliates (MIT).
 * - MLXPorts/Faiss-mlx, commit d092af559375144fc719cd88a10e414f92c625fa
 *   Copyright 2024 Sydney Bach, The Solace Project (Apache-2.0).
 *   Upstream file: python/metalfaiss/faissmlx/kernels/ivf_kernels.py.
 * - FAISS upstream files: faiss/IndexIVF.{h,cpp} and
 *   faiss/IndexIVFFlat.{h,cpp}.
 *
 * This file is a new Objective-C++/Metal implementation. FAISS-derived
 * portions remain under the MIT license; portions adapted from the Faiss-mlx
 * fused list-scan/top-k organization remain under Apache-2.0. Modifications
 * are Copyright (c) 2026 Stefano Cacciatore. Redistributed derivatives must
 * retain both upstream notices and licenses from inst/LICENSES/.
 */

#include <Rcpp.h>

#include "native_knn_common.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {

constexpr int kMaxP = 1024;
constexpr int kMaxGlobalP = 16384;
constexpr int kMaxLists = 1024;
// Keep enough local top-k storage for compact t-SNE and UMAP neighborhoods
// without routing an explicit Metal request through a host fallback.
constexpr int kMaxK = 128;
constexpr int kMaxProbe = 256;
constexpr int kMaxShortlistPerGroup = 128;
constexpr int kInitialShortlistPerGroup = 72;
constexpr int kProjectionDim = 128;
constexpr int kMetalSimdGroups = 4;

const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant uint NSG = 4;
constant uint SIMD_WIDTH = 32;
constant uint MAX_P = 1024;
constant uint MAX_Q = 128;
constant uint MAX_K = 128;
constant uint MAX_PROBE = 256;
constant uint MAX_SHORTLIST = 128;

struct ProjectParams {
  uint n;
  uint p;
  uint qdim;
};

struct IndexParams {
  uint n;
  uint p;
  uint qdim;
  uint nlist;
};

struct SearchParams {
  uint n;
  uint p;
  uint qdim;
  uint nlist;
  uint nprobe;
  uint k;
  uint query_offset;
  uint shortlist_per_group;
};

struct QuerySearchParams {
  uint n_reference;
  uint n_query;
  uint p;
  uint k;
};

struct QueryIvfParams {
  uint n_reference;
  uint n_query;
  uint p;
  uint qdim;
  uint nlist;
  uint nprobe;
  uint k;
};

inline bool better_pair(float da, int ia, float db, int ib) {
  return da < db || (da == db && ia < ib);
}

inline void insert_sorted(
    threadgroup float* values,
    threadgroup int* ids,
    uint base,
    uint count,
    float value,
    int id) {
  if (count == 0 || !better_pair(value, id, values[base + count - 1], ids[base + count - 1])) return;
  uint pos = count - 1;
  while (pos > 0 && better_pair(value, id, values[base + pos - 1], ids[base + pos - 1])) {
    values[base + pos] = values[base + pos - 1];
    ids[base + pos] = ids[base + pos - 1];
    --pos;
  }
  values[base + pos] = value;
  ids[base + pos] = id;
}

kernel void signed_hash_project(
    device const float* data [[buffer(0)]],
    device const uint* feature_offsets [[buffer(1)]],
    device const uint* feature_ids [[buffer(2)]],
    device const char* feature_signs [[buffer(3)]],
    device float* projected [[buffer(4)]],
    constant ProjectParams& params [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]) {
  if (row >= params.n || tid >= params.qdim) return;
  float value = 0.0f;
  uint begin = feature_offsets[tid];
  uint end = feature_offsets[tid + 1];
  for (uint pos = begin; pos < end; ++pos) {
    value = fma(float(feature_signs[pos]), data[row * params.p + feature_ids[pos]], value);
  }
  projected[row * params.qdim + tid] = value;
}

kernel void assign_centroid(
    device const float* projected [[buffer(0)]],
    device const float* centroids [[buffer(1)]],
    device int* assignment [[buffer(2)]],
    constant IndexParams& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float query[MAX_Q];
  threadgroup float best_dist[NSG];
  threadgroup int best_id[NSG];
  if (row >= params.n) return;
  for (uint d = tid; d < params.qdim; d += NSG * SIMD_WIDTH) query[d] = projected[row * params.qdim + d];
  if (tid < NSG) { best_dist[tid] = INFINITY; best_id[tid] = -1; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint c = sg; c < params.nlist; c += NSG) {
    float partial = 0.0f;
    for (uint d = lane; d < params.qdim; d += SIMD_WIDTH) {
      float delta = query[d] - centroids[c * params.qdim + d];
      partial = fma(delta, delta, partial);
    }
    float distance = simd_sum(partial);
    if (lane == 0 && better_pair(distance, int(c), best_dist[sg], best_id[sg])) {
      best_dist[sg] = distance;
      best_id[sg] = int(c);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float distance = best_dist[0];
    int id = best_id[0];
    for (uint g = 1; g < NSG; ++g) {
      if (better_pair(best_dist[g], best_id[g], distance, id)) {
        distance = best_dist[g];
        id = best_id[g];
      }
    }
    assignment[row] = id;
  }
}

kernel void clear_centroid_accumulators(
    device atomic_float* sums [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    constant IndexParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  uint total = params.nlist * params.qdim;
  if (gid < total) atomic_store_explicit(&sums[gid], 0.0f, memory_order_relaxed);
  if (gid < params.nlist) atomic_store_explicit(&counts[gid], 0u, memory_order_relaxed);
}

kernel void accumulate_centroids(
    device const float* projected [[buffer(0)]],
    device const int* assignment [[buffer(1)]],
    device atomic_float* sums [[buffer(2)]],
    device atomic_uint* counts [[buffer(3)]],
    constant IndexParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  uint total = params.n * params.qdim;
  if (gid >= total) return;
  uint row = gid / params.qdim;
  uint dimension = gid - row * params.qdim;
  int centroid = assignment[row];
  if (centroid < 0 || uint(centroid) >= params.nlist) return;
  atomic_fetch_add_explicit(
    &sums[uint(centroid) * params.qdim + dimension],
    projected[gid],
    memory_order_relaxed
  );
  if (dimension == 0) atomic_fetch_add_explicit(&counts[centroid], 1u, memory_order_relaxed);
}

kernel void finalize_centroids(
    device const atomic_float* sums [[buffer(0)]],
    device const atomic_uint* counts [[buffer(1)]],
    device float* centroids [[buffer(2)]],
    constant IndexParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  uint total = params.nlist * params.qdim;
  if (gid >= total) return;
  uint centroid = gid / params.qdim;
  uint count = atomic_load_explicit(&counts[centroid], memory_order_relaxed);
  if (count > 0) {
    centroids[gid] = atomic_load_explicit(&sums[gid], memory_order_relaxed) / float(count);
  }
}

kernel void exact_topk_simd(
    device const float* data [[buffer(0)]],
    device int* out_ids [[buffer(1)]],
    device float* out_dist [[buffer(2)]],
    constant SearchParams& params [[buffer(3)]],
    uint qlocal [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float query[MAX_P];
  threadgroup float local_dist[NSG * MAX_K];
  threadgroup int local_id[NSG * MAX_K];
  uint q = params.query_offset + qlocal;
  if (q >= params.n) return;
  for (uint d = tid; d < params.p; d += NSG * SIMD_WIDTH) query[d] = data[q * params.p + d];
  for (uint i = tid; i < NSG * params.k; i += NSG * SIMD_WIDTH) {
    local_dist[i] = INFINITY;
    local_id[i] = -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint base = sg * params.k;
  for (uint candidate = sg; candidate < params.n; candidate += NSG) {
    if (candidate == q) continue;
    float partial = 0.0f;
    const device float* point = data + candidate * params.p;
    for (uint d = lane; d < params.p; d += SIMD_WIDTH) {
      float delta = query[d] - point[d];
      partial = fma(delta, delta, partial);
    }
    float distance = simd_sum(partial);
    if (lane == 0) insert_sorted(local_dist, local_id, base, params.k, distance, int(candidate));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint heads[NSG];
    for (uint g = 0; g < NSG; ++g) heads[g] = 0;
    for (uint rank = 0; rank < params.k; ++rank) {
      float best_d = INFINITY;
      int best_i = -1;
      uint best_g = 0;
      for (uint g = 0; g < NSG; ++g) {
        uint h = heads[g];
        if (h < params.k) {
          uint pos = g * params.k + h;
          if (better_pair(local_dist[pos], local_id[pos], best_d, best_i)) {
            best_d = local_dist[pos]; best_i = local_id[pos]; best_g = g;
          }
        }
      }
      out_dist[qlocal * params.k + rank] = best_d;
      out_ids[qlocal * params.k + rank] = best_i;
      ++heads[best_g];
    }
  }
}

// High-dimensional exact search avoids a fixed threadgroup query cache. This
// preserves the same L2 calculation for inputs that would exceed Metal's
// threadgroup-memory budget, at the cost of rereading the query from device
// memory for each candidate.
kernel void exact_topk_simd_global_query(
    device const float* data [[buffer(0)]],
    device int* out_ids [[buffer(1)]],
    device float* out_dist [[buffer(2)]],
    constant SearchParams& params [[buffer(3)]],
    uint qlocal [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float local_dist[NSG * MAX_K];
  threadgroup int local_id[NSG * MAX_K];
  uint q = params.query_offset + qlocal;
  if (q >= params.n) return;
  for (uint i = tid; i < NSG * params.k; i += NSG * SIMD_WIDTH) {
    local_dist[i] = INFINITY;
    local_id[i] = -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint base = sg * params.k;
  const device float* query = data + q * params.p;
  for (uint candidate = sg; candidate < params.n; candidate += NSG) {
    if (candidate == q) continue;
    float partial = 0.0f;
    const device float* point = data + candidate * params.p;
    for (uint d = lane; d < params.p; d += SIMD_WIDTH) {
      float delta = query[d] - point[d];
      partial = fma(delta, delta, partial);
    }
    float distance = simd_sum(partial);
    if (lane == 0) {
      insert_sorted(
        local_dist, local_id, base, params.k, distance, int(candidate)
      );
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint heads[NSG];
    for (uint g = 0; g < NSG; ++g) heads[g] = 0;
    for (uint rank = 0; rank < params.k; ++rank) {
      float best_d = INFINITY;
      int best_i = -1;
      uint best_g = 0;
      for (uint g = 0; g < NSG; ++g) {
        uint h = heads[g];
        if (h < params.k) {
          uint pos = g * params.k + h;
          if (better_pair(local_dist[pos], local_id[pos], best_d, best_i)) {
            best_d = local_dist[pos];
            best_i = local_id[pos];
            best_g = g;
          }
        }
      }
      out_dist[qlocal * params.k + rank] = best_d;
      out_ids[qlocal * params.k + rank] = best_i;
      ++heads[best_g];
    }
  }
}

kernel void exact_query_topk_simd(
    device const float* reference [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device int* out_ids [[buffer(2)]],
    device float* out_dist [[buffer(3)]],
    constant QuerySearchParams& params [[buffer(4)]],
    uint q [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float query_row[MAX_P];
  threadgroup float local_dist[NSG * MAX_K];
  threadgroup int local_id[NSG * MAX_K];
  if (q >= params.n_query) return;
  for (uint d = tid; d < params.p; d += NSG * SIMD_WIDTH) {
    query_row[d] = query[q * params.p + d];
  }
  for (uint i = tid; i < NSG * params.k; i += NSG * SIMD_WIDTH) {
    local_dist[i] = INFINITY;
    local_id[i] = -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint base = sg * params.k;
  for (uint candidate = sg; candidate < params.n_reference; candidate += NSG) {
    float partial = 0.0f;
    const device float* point = reference + candidate * params.p;
    for (uint d = lane; d < params.p; d += SIMD_WIDTH) {
      float delta = query_row[d] - point[d];
      partial = fma(delta, delta, partial);
    }
    float distance = simd_sum(partial);
    if (lane == 0) {
      insert_sorted(
        local_dist, local_id, base, params.k, distance, int(candidate)
      );
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint heads[NSG];
    for (uint g = 0; g < NSG; ++g) heads[g] = 0;
    for (uint rank = 0; rank < params.k; ++rank) {
      float best_d = INFINITY;
      int best_i = -1;
      uint best_g = 0;
      for (uint g = 0; g < NSG; ++g) {
        uint h = heads[g];
        if (h < params.k) {
          uint pos = g * params.k + h;
          if (better_pair(local_dist[pos], local_id[pos], best_d, best_i)) {
            best_d = local_dist[pos];
            best_i = local_id[pos];
            best_g = g;
          }
        }
      }
      out_dist[q * params.k + rank] = best_d;
      out_ids[q * params.k + rank] = best_i;
      ++heads[best_g];
    }
  }
}

kernel void exact_query_topk_simd_global_query(
    device const float* reference [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device int* out_ids [[buffer(2)]],
    device float* out_dist [[buffer(3)]],
    constant QuerySearchParams& params [[buffer(4)]],
    uint q [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float local_dist[NSG * MAX_K];
  threadgroup int local_id[NSG * MAX_K];
  if (q >= params.n_query) return;
  for (uint i = tid; i < NSG * params.k; i += NSG * SIMD_WIDTH) {
    local_dist[i] = INFINITY;
    local_id[i] = -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint base = sg * params.k;
  const device float* query_row = query + q * params.p;
  for (uint candidate = sg; candidate < params.n_reference; candidate += NSG) {
    float partial = 0.0f;
    const device float* point = reference + candidate * params.p;
    for (uint d = lane; d < params.p; d += SIMD_WIDTH) {
      float delta = query_row[d] - point[d];
      partial = fma(delta, delta, partial);
    }
    float distance = simd_sum(partial);
    if (lane == 0) {
      insert_sorted(
        local_dist, local_id, base, params.k, distance, int(candidate)
      );
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint heads[NSG];
    for (uint g = 0; g < NSG; ++g) heads[g] = 0;
    for (uint rank = 0; rank < params.k; ++rank) {
      float best_d = INFINITY;
      int best_i = -1;
      uint best_g = 0;
      for (uint g = 0; g < NSG; ++g) {
        uint h = heads[g];
        if (h < params.k) {
          uint pos = g * params.k + h;
          if (better_pair(local_dist[pos], local_id[pos], best_d, best_i)) {
            best_d = local_dist[pos];
            best_i = local_id[pos];
            best_g = g;
          }
        }
      }
      out_dist[q * params.k + rank] = best_d;
      out_ids[q * params.k + rank] = best_i;
      ++heads[best_g];
    }
  }
}

kernel void ivf_query_topk_fused(
    device const float* reference [[buffer(0)]],
    device const float* query [[buffer(1)]],
    device const float* query_projected [[buffer(2)]],
    device const float* centroids [[buffer(3)]],
    device const uint* list_offsets [[buffer(4)]],
    device const int* list_ids [[buffer(5)]],
    device int* out_ids [[buffer(6)]],
    device float* out_dist [[buffer(7)]],
    constant QueryIvfParams& params [[buffer(8)]],
    uint q [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float query_row[MAX_P];
  threadgroup float query_low[MAX_Q];
  threadgroup float coarse_dist[NSG * MAX_PROBE];
  threadgroup int coarse_id[NSG * MAX_PROBE];
  threadgroup int probes[MAX_PROBE];
  threadgroup float local_dist[NSG * MAX_K];
  threadgroup int local_id[NSG * MAX_K];
  if (q >= params.n_query || params.nprobe > MAX_PROBE) return;

  for (uint d = tid; d < params.p; d += NSG * SIMD_WIDTH) {
    query_row[d] = query[q * params.p + d];
  }
  for (uint d = tid; d < params.qdim; d += NSG * SIMD_WIDTH) {
    query_low[d] = query_projected[q * params.qdim + d];
  }
  for (uint i = tid; i < NSG * params.nprobe; i += NSG * SIMD_WIDTH) {
    coarse_dist[i] = INFINITY;
    coarse_id[i] = -1;
  }
  for (uint i = tid; i < NSG * params.k; i += NSG * SIMD_WIDTH) {
    local_dist[i] = INFINITY;
    local_id[i] = -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint coarse_base = sg * params.nprobe;
  for (uint c = sg; c < params.nlist; c += NSG) {
    float partial = 0.0f;
    const device float* centroid = centroids + c * params.qdim;
    for (uint d = lane; d < params.qdim; d += SIMD_WIDTH) {
      float delta = query_low[d] - centroid[d];
      partial = fma(delta, delta, partial);
    }
    float distance = simd_sum(partial);
    if (lane == 0) {
      insert_sorted(
        coarse_dist, coarse_id, coarse_base, params.nprobe, distance, int(c)
      );
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint heads[NSG];
    for (uint g = 0; g < NSG; ++g) heads[g] = 0;
    for (uint rank = 0; rank < params.nprobe; ++rank) {
      float best_d = INFINITY;
      int best_i = -1;
      uint best_g = 0;
      for (uint g = 0; g < NSG; ++g) {
        uint h = heads[g];
        if (h < params.nprobe) {
          uint pos = g * params.nprobe + h;
          if (better_pair(coarse_dist[pos], coarse_id[pos], best_d, best_i)) {
            best_d = coarse_dist[pos];
            best_i = coarse_id[pos];
            best_g = g;
          }
        }
      }
      probes[rank] = best_i;
      ++heads[best_g];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint fine_base = sg * params.k;
  for (uint probe = 0; probe < params.nprobe; ++probe) {
    int list = probes[probe];
    if (list < 0 || uint(list) >= params.nlist) continue;
    uint begin = list_offsets[list];
    uint end = list_offsets[list + 1];
    for (uint pos = begin + sg; pos < end; pos += NSG) {
      int candidate = list_ids[pos];
      if (candidate < 0 || uint(candidate) >= params.n_reference) continue;
      float partial = 0.0f;
      const device float* point =
        reference + uint(candidate) * params.p;
      for (uint d = lane; d < params.p; d += SIMD_WIDTH) {
        float delta = query_row[d] - point[d];
        partial = fma(delta, delta, partial);
      }
      float distance = simd_sum(partial);
      if (lane == 0) {
        insert_sorted(
          local_dist, local_id, fine_base, params.k, distance, candidate
        );
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint heads[NSG];
    for (uint g = 0; g < NSG; ++g) heads[g] = 0;
    for (uint rank = 0; rank < params.k; ++rank) {
      float best_d = INFINITY;
      int best_i = -1;
      uint best_g = 0;
      for (uint g = 0; g < NSG; ++g) {
        uint h = heads[g];
        if (h < params.k) {
          uint pos = g * params.k + h;
          if (better_pair(local_dist[pos], local_id[pos], best_d, best_i)) {
            best_d = local_dist[pos];
            best_i = local_id[pos];
            best_g = g;
          }
        }
      }
      out_dist[q * params.k + rank] = best_d;
      out_ids[q * params.k + rank] = best_i;
      ++heads[best_g];
    }
  }
}

kernel void ivf_topk_fused(
    device const float* data [[buffer(0)]],
    device const float* projected [[buffer(1)]],
    device const float* centroids [[buffer(2)]],
    device const uint* list_offsets [[buffer(3)]],
    device const int* list_ids [[buffer(4)]],
    device int* out_ids [[buffer(5)]],
    device float* out_dist [[buffer(6)]],
    constant SearchParams& params [[buffer(7)]],
    uint qlocal [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float query[MAX_P];
  threadgroup float coarse_dist[NSG * MAX_PROBE];
  threadgroup int coarse_id[NSG * MAX_PROBE];
  threadgroup int probes[MAX_PROBE];
  threadgroup float local_dist[NSG * MAX_K];
  threadgroup int local_id[NSG * MAX_K];
  uint q = params.query_offset + qlocal;
  if (q >= params.n) return;

  for (uint d = tid; d < params.p; d += NSG * SIMD_WIDTH) query[d] = data[q * params.p + d];
  for (uint i = tid; i < NSG * params.nprobe; i += NSG * SIMD_WIDTH) {
    coarse_dist[i] = INFINITY;
    coarse_id[i] = -1;
  }
  for (uint i = tid; i < NSG * params.k; i += NSG * SIMD_WIDTH) {
    local_dist[i] = INFINITY;
    local_id[i] = -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint coarse_base = sg * params.nprobe;
  const device float* qproj = projected + q * params.qdim;
  for (uint c = sg; c < params.nlist; c += NSG) {
    float partial = 0.0f;
    const device float* centroid = centroids + c * params.qdim;
    for (uint d = lane; d < params.qdim; d += SIMD_WIDTH) {
      float delta = qproj[d] - centroid[d];
      partial = fma(delta, delta, partial);
    }
    float distance = simd_sum(partial);
    if (lane == 0) insert_sorted(coarse_dist, coarse_id, coarse_base, params.nprobe, distance, int(c));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint heads[NSG];
    for (uint g = 0; g < NSG; ++g) heads[g] = 0;
    for (uint rank = 0; rank < params.nprobe; ++rank) {
      float best_d = INFINITY;
      int best_i = -1;
      uint best_g = 0;
      for (uint g = 0; g < NSG; ++g) {
        uint h = heads[g];
        if (h < params.nprobe) {
          uint pos = g * params.nprobe + h;
          if (better_pair(coarse_dist[pos], coarse_id[pos], best_d, best_i)) {
            best_d = coarse_dist[pos]; best_i = coarse_id[pos]; best_g = g;
          }
        }
      }
      probes[rank] = best_i;
      ++heads[best_g];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint fine_base = sg * params.k;
  for (uint probe = 0; probe < params.nprobe; ++probe) {
    int list = probes[probe];
    if (list < 0 || uint(list) >= params.nlist) continue;
    uint begin = list_offsets[list];
    uint end = list_offsets[list + 1];
    for (uint pos = begin + sg; pos < end; pos += NSG) {
      int candidate = list_ids[pos];
      if (candidate < 0 || uint(candidate) == q) continue;
      float partial = 0.0f;
      const device float* point = data + uint(candidate) * params.p;
      for (uint d = lane; d < params.p; d += SIMD_WIDTH) {
        float delta = query[d] - point[d];
        partial = fma(delta, delta, partial);
      }
      float distance = simd_sum(partial);
      if (lane == 0) insert_sorted(local_dist, local_id, fine_base, params.k, distance, candidate);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint heads[NSG];
    for (uint g = 0; g < NSG; ++g) heads[g] = 0;
    for (uint rank = 0; rank < params.k; ++rank) {
      float best_d = INFINITY;
      int best_i = -1;
      uint best_g = 0;
      for (uint g = 0; g < NSG; ++g) {
        uint h = heads[g];
        if (h < params.k) {
          uint pos = g * params.k + h;
          if (better_pair(local_dist[pos], local_id[pos], best_d, best_i)) {
            best_d = local_dist[pos]; best_i = local_id[pos]; best_g = g;
          }
        }
      }
      out_dist[qlocal * params.k + rank] = best_d;
      out_ids[qlocal * params.k + rank] = best_i;
      ++heads[best_g];
    }
  }
}

kernel void ivf_projected_shortlist(
    device const float* projected [[buffer(0)]],
    device const float* centroids [[buffer(1)]],
    device const uint* list_offsets [[buffer(2)]],
    device const int* list_ids [[buffer(3)]],
    device int* candidate_ids [[buffer(4)]],
    constant SearchParams& params [[buffer(5)]],
    uint qlocal [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float query[MAX_Q];
  threadgroup float coarse_dist[NSG * MAX_PROBE];
  threadgroup int coarse_id[NSG * MAX_PROBE];
  threadgroup int probes[MAX_PROBE];
  threadgroup float candidate_dist[NSG * MAX_SHORTLIST];
  threadgroup int candidate_local_id[NSG * MAX_SHORTLIST];
  uint q = params.query_offset + qlocal;
  if (q >= params.n || params.shortlist_per_group > MAX_SHORTLIST) return;

  for (uint d = tid; d < params.qdim; d += NSG * SIMD_WIDTH) {
    query[d] = projected[q * params.qdim + d];
  }
  for (uint i = tid; i < NSG * params.nprobe; i += NSG * SIMD_WIDTH) {
    coarse_dist[i] = INFINITY;
    coarse_id[i] = -1;
  }
  for (uint i = tid; i < NSG * params.shortlist_per_group; i += NSG * SIMD_WIDTH) {
    candidate_dist[i] = INFINITY;
    candidate_local_id[i] = -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint coarse_base = sg * params.nprobe;
  for (uint c = sg; c < params.nlist; c += NSG) {
    float partial = 0.0f;
    const device float* centroid = centroids + c * params.qdim;
    for (uint d = lane; d < params.qdim; d += SIMD_WIDTH) {
      float delta = query[d] - centroid[d];
      partial = fma(delta, delta, partial);
    }
    float distance = simd_sum(partial);
    if (lane == 0) {
      insert_sorted(coarse_dist, coarse_id, coarse_base, params.nprobe,
                    distance, int(c));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint heads[NSG];
    for (uint g = 0; g < NSG; ++g) heads[g] = 0;
    for (uint rank = 0; rank < params.nprobe; ++rank) {
      float best_d = INFINITY;
      int best_i = -1;
      uint best_g = 0;
      for (uint g = 0; g < NSG; ++g) {
        uint h = heads[g];
        if (h < params.nprobe) {
          uint pos = g * params.nprobe + h;
          if (better_pair(coarse_dist[pos], coarse_id[pos], best_d, best_i)) {
            best_d = coarse_dist[pos];
            best_i = coarse_id[pos];
            best_g = g;
          }
        }
      }
      probes[rank] = best_i;
      ++heads[best_g];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint candidate_base = sg * params.shortlist_per_group;
  for (uint probe = 0; probe < params.nprobe; ++probe) {
    int list = probes[probe];
    if (list < 0 || uint(list) >= params.nlist) continue;
    uint begin = list_offsets[list];
    uint end = list_offsets[list + 1];
    for (uint pos = begin + sg; pos < end; pos += NSG) {
      int candidate = list_ids[pos];
      if (candidate < 0 || uint(candidate) == q) continue;
      float partial = 0.0f;
      const device float* point = projected + uint(candidate) * params.qdim;
      for (uint d = lane; d < params.qdim; d += SIMD_WIDTH) {
        float delta = query[d] - point[d];
        partial = fma(delta, delta, partial);
      }
      float distance = simd_sum(partial);
      if (lane == 0) {
        insert_sorted(candidate_dist, candidate_local_id, candidate_base,
                      params.shortlist_per_group, distance, candidate);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint shortlist_size = NSG * params.shortlist_per_group;
  uint output_base = qlocal * shortlist_size;
  for (uint i = tid; i < shortlist_size; i += NSG * SIMD_WIDTH) {
    candidate_ids[output_base + i] = candidate_local_id[i];
  }
}

kernel void ivf_exact_rerank_shortlist(
    device const float* data [[buffer(0)]],
    device const int* candidate_ids [[buffer(1)]],
    device int* out_ids [[buffer(2)]],
    device float* out_dist [[buffer(3)]],
    constant SearchParams& params [[buffer(4)]],
    uint qlocal [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float query[MAX_P];
  threadgroup float local_dist[NSG * MAX_K];
  threadgroup int local_id[NSG * MAX_K];
  uint q = params.query_offset + qlocal;
  if (q >= params.n || params.shortlist_per_group > MAX_SHORTLIST) return;

  for (uint d = tid; d < params.p; d += NSG * SIMD_WIDTH) {
    query[d] = data[q * params.p + d];
  }
  for (uint i = tid; i < NSG * params.k; i += NSG * SIMD_WIDTH) {
    local_dist[i] = INFINITY;
    local_id[i] = -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint shortlist_size = NSG * params.shortlist_per_group;
  uint candidate_base = qlocal * shortlist_size;
  uint local_base = sg * params.k;
  for (uint pos = sg; pos < shortlist_size; pos += NSG) {
    int candidate = candidate_ids[candidate_base + pos];
    if (candidate < 0 || uint(candidate) == q) continue;
    float partial = 0.0f;
    const device float* point = data + uint(candidate) * params.p;
    for (uint d = lane; d < params.p; d += SIMD_WIDTH) {
      float delta = query[d] - point[d];
      partial = fma(delta, delta, partial);
    }
    float distance = simd_sum(partial);
    if (lane == 0) {
      insert_sorted(local_dist, local_id, local_base, params.k,
                    distance, candidate);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    uint heads[NSG];
    for (uint g = 0; g < NSG; ++g) heads[g] = 0;
    for (uint rank = 0; rank < params.k; ++rank) {
      float best_d = INFINITY;
      int best_i = -1;
      uint best_g = 0;
      for (uint g = 0; g < NSG; ++g) {
        uint h = heads[g];
        if (h < params.k) {
          uint pos = g * params.k + h;
          if (better_pair(local_dist[pos], local_id[pos], best_d, best_i)) {
            best_d = local_dist[pos];
            best_i = local_id[pos];
            best_g = g;
          }
        }
      }
      out_dist[qlocal * params.k + rank] = best_d;
      out_ids[qlocal * params.k + rank] = best_i;
      ++heads[best_g];
    }
  }
}
)METAL";

struct ProjectParamsHost { std::uint32_t n, p, qdim; };
struct IndexParamsHost { std::uint32_t n, p, qdim, nlist; };
struct SearchParamsHost {
  std::uint32_t n, p, qdim, nlist, nprobe, k, query_offset, shortlist_per_group;
};
struct QuerySearchParamsHost {
  std::uint32_t n_reference, n_query, p, k;
};
struct QueryIvfParamsHost {
  std::uint32_t n_reference, n_query, p, qdim, nlist, nprobe, k;
};

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString* name) {
  id<MTLFunction> function = [library newFunctionWithName:name];
  if (!function) Rcpp::stop("missing Metal function %s", [name UTF8String]);
  NSError* error = nil;
  id<MTLComputePipelineState> result = [device newComputePipelineStateWithFunction:function error:&error];
  [function release];
  if (!result) Rcpp::stop("failed to build Metal pipeline: %s", error ? [[error localizedDescription] UTF8String] : "unknown");
  return result;
}

struct MetalKnnState {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> project_pipeline = nil;
  id<MTLComputePipelineState> assign_pipeline = nil;
  id<MTLComputePipelineState> clear_centroids_pipeline = nil;
  id<MTLComputePipelineState> accumulate_centroids_pipeline = nil;
  id<MTLComputePipelineState> finalize_centroids_pipeline = nil;
  id<MTLComputePipelineState> exact_pipeline = nil;
  id<MTLComputePipelineState> exact_global_pipeline = nil;
  id<MTLComputePipelineState> exact_query_pipeline = nil;
  id<MTLComputePipelineState> exact_query_global_pipeline = nil;
  id<MTLComputePipelineState> ivf_query_pipeline = nil;
  id<MTLComputePipelineState> ivf_pipeline = nil;
  id<MTLComputePipelineState> shortlist_pipeline = nil;
  id<MTLComputePipelineState> rerank_pipeline = nil;
};

MTLCompileOptions* metal_knn_compile_options() {
  MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
  if (@available(macOS 15.0, *)) {
    options.mathMode = MTLMathModeFast;
  }
#else
  options.fastMathEnabled = YES;
#endif
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 140000
  if (@available(macOS 14.0, *)) {
    options.languageVersion = MTLLanguageVersion3_1;
    return options;
  }
#endif
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 130000
  if (@available(macOS 13.0, *)) {
    options.languageVersion = MTLLanguageVersion3_0;
    return options;
  }
#endif
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 120000
  if (@available(macOS 12.0, *)) {
    options.languageVersion = MTLLanguageVersion2_4;
    return options;
  }
#endif
  options.languageVersion = MTLLanguageVersion2_3;
  return options;
}

MetalKnnState& metal_knn_state() {
  static MetalKnnState state;
  if (state.device != nil && state.queue != nil && state.library != nil &&
      state.project_pipeline != nil && state.assign_pipeline != nil &&
      state.clear_centroids_pipeline != nil &&
      state.accumulate_centroids_pipeline != nil &&
      state.finalize_centroids_pipeline != nil &&
      state.exact_pipeline != nil && state.exact_global_pipeline != nil &&
      state.exact_query_pipeline != nil &&
      state.exact_query_global_pipeline != nil &&
      state.ivf_query_pipeline != nil &&
      state.ivf_pipeline != nil &&
      state.shortlist_pipeline != nil && state.rerank_pipeline != nil) {
    return state;
  }
  state.device = MTLCreateSystemDefaultDevice();
  if (state.device == nil) Rcpp::stop("No Metal device is available.");
  state.queue = [state.device newCommandQueue];
  MTLCompileOptions* options = metal_knn_compile_options();
  NSError* error = nil;
  state.library = [state.device
    newLibraryWithSource:[NSString stringWithUTF8String:kMetalSource]
    options:options
    error:&error];
  [options release];
  if (state.library == nil) {
    Rcpp::stop(
      "Failed to compile native Metal KNN kernels: %s",
      error ? [[error localizedDescription] UTF8String] : "unknown"
    );
  }
  state.project_pipeline = make_pipeline(state.device, state.library, @"signed_hash_project");
  state.assign_pipeline = make_pipeline(state.device, state.library, @"assign_centroid");
  state.clear_centroids_pipeline = make_pipeline(state.device, state.library, @"clear_centroid_accumulators");
  state.accumulate_centroids_pipeline = make_pipeline(state.device, state.library, @"accumulate_centroids");
  state.finalize_centroids_pipeline = make_pipeline(state.device, state.library, @"finalize_centroids");
  state.exact_pipeline = make_pipeline(state.device, state.library, @"exact_topk_simd");
  state.exact_global_pipeline = make_pipeline(
    state.device, state.library, @"exact_topk_simd_global_query"
  );
  state.exact_query_pipeline = make_pipeline(
    state.device, state.library, @"exact_query_topk_simd"
  );
  state.exact_query_global_pipeline = make_pipeline(
    state.device, state.library, @"exact_query_topk_simd_global_query"
  );
  state.ivf_query_pipeline = make_pipeline(
    state.device, state.library, @"ivf_query_topk_fused"
  );
  state.ivf_pipeline = make_pipeline(state.device, state.library, @"ivf_topk_fused");
  state.shortlist_pipeline = make_pipeline(state.device, state.library, @"ivf_projected_shortlist");
  state.rerank_pipeline = make_pipeline(state.device, state.library, @"ivf_exact_rerank_shortlist");
  return state;
}

void run_and_wait(id<MTLCommandBuffer> command) {
  [command commit];
  [command waitUntilCompleted];
  if (command.status == MTLCommandBufferStatusError) {
    Rcpp::stop("Metal KNN command failed: %s", command.error ? [[command.error localizedDescription] UTF8String] : "unknown");
  }
}

double recall_at_k(const int* truth, const int* approx, int rows, int k) {
  double hits = 0.0;
  for (int i = 0; i < rows; ++i) {
    for (int a = 0; a < k; ++a) {
      int id = approx[static_cast<std::size_t>(i) * k + a];
      for (int t = 0; t < k; ++t) {
        if (id == truth[static_cast<std::size_t>(i) * k + t]) { hits += 1.0; break; }
      }
    }
  }
  return hits / static_cast<double>(rows * k);
}

int projection_dim(int p) {
  int q = 1;
  while (q * 2 <= p && q * 2 <= kProjectionDim) q *= 2;
  return std::max(1, q);
}

} // namespace

bool native_metal_knn_available_impl() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    const bool available = device != nil;
    if (device != nil) [device release];
    return available;
  }
}

Rcpp::List native_metal_knn_impl(SEXP data_sexp,
                                 int k,
                                 const std::string& requested_method,
                                 const std::string& metric_name,
                                 double target) {
  using Clock = std::chrono::steady_clock;
  const auto t0 = Clock::now();
  const fastembedr::KnnMetric metric = fastembedr::parse_knn_metric(metric_name);
  fastembedr::FloatMatrix input = fastembedr::matrix_to_row_major_float(data_sexp, metric);
  std::vector<float>& data = input.values;
  const int n = input.nrow;
  const int p = input.ncol;
  const auto t_convert = Clock::now();
  if (n < 2 || p < 1 || p > kMaxGlobalP ||
      k < 1 || k > kMaxK || k >= n) {
    Rcpp::stop(
      "Native Metal KNN supports 1-%d dimensions and k <= %d.",
      kMaxGlobalP, kMaxK
    );
  }
  std::string method = requested_method;
  if (method == "auto") {
    method = n < 4096 ? "exact" : "ivf";
  }
  if (method != "exact" && method != "ivf") {
    Rcpp::stop("Native Metal KNN method must be `auto`, `exact`, or `ivf`.");
  }
  if (method == "ivf" && p > kMaxP) {
    Rcpp::stop(
      "Native Metal IVF KNN supports at most %d dimensions; use exact "
      "search for this %d-dimensional input.",
      kMaxP, p
    );
  }
  const int qdim = projection_dim(p);
  const int nlist = std::max(16, std::min(kMaxLists, static_cast<int>(std::ceil(4.0 * std::sqrt(static_cast<double>(n))))));

  std::vector<std::uint32_t> feature_offsets(static_cast<std::size_t>(qdim) + 1u, 0u);
  std::vector<std::vector<std::pair<std::uint32_t, std::int8_t>>> buckets(qdim);
  for (int d = 0; d < p; ++d) {
    std::uint32_t h = static_cast<std::uint32_t>(d + 1) * 2654435761u;
    int bucket = static_cast<int>(h & static_cast<std::uint32_t>(qdim - 1));
    std::int8_t sign = ((h >> 17u) & 1u) ? std::int8_t(1) : std::int8_t(-1);
    buckets[bucket].push_back({static_cast<std::uint32_t>(d), sign});
  }
  std::vector<std::uint32_t> feature_ids;
  std::vector<std::int8_t> feature_signs;
  feature_ids.reserve(p); feature_signs.reserve(p);
  for (int b = 0; b < qdim; ++b) {
    feature_offsets[b] = static_cast<std::uint32_t>(feature_ids.size());
    for (const auto& item : buckets[b]) { feature_ids.push_back(item.first); feature_signs.push_back(item.second); }
  }
  feature_offsets[qdim] = static_cast<std::uint32_t>(feature_ids.size());

  @autoreleasepool {
    MetalKnnState& state = metal_knn_state();
    id<MTLDevice> device = state.device;
    id<MTLCommandQueue> queue = state.queue;
    id<MTLComputePipelineState> project_pipeline = state.project_pipeline;
    id<MTLComputePipelineState> assign_pipeline = state.assign_pipeline;
    id<MTLComputePipelineState> clear_centroids_pipeline = state.clear_centroids_pipeline;
    id<MTLComputePipelineState> accumulate_centroids_pipeline = state.accumulate_centroids_pipeline;
    id<MTLComputePipelineState> finalize_centroids_pipeline = state.finalize_centroids_pipeline;
    id<MTLComputePipelineState> exact_pipeline = state.exact_pipeline;
    id<MTLComputePipelineState> exact_global_pipeline =
      state.exact_global_pipeline;
    id<MTLComputePipelineState> ivf_pipeline = state.ivf_pipeline;
    id<MTLComputePipelineState> shortlist_pipeline = state.shortlist_pipeline;
    id<MTLComputePipelineState> rerank_pipeline = state.rerank_pipeline;
    const auto t_setup = Clock::now();

    id<MTLBuffer> data_buffer = [device newBufferWithBytes:data.data() length:data.size() * sizeof(float) options:MTLResourceStorageModeShared];
    if (method == "exact") {
      id<MTLBuffer> out_id_buffer = [device newBufferWithLength:static_cast<std::size_t>(n) * k * sizeof(int) options:MTLResourceStorageModeShared];
      id<MTLBuffer> out_dist_buffer = [device newBufferWithLength:static_cast<std::size_t>(n) * k * sizeof(float) options:MTLResourceStorageModeShared];
      SearchParamsHost params{static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(p), 0u, 0u, 0u, static_cast<std::uint32_t>(k), 0u, 0u};
      id<MTLBuffer> params_buffer = [device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:
        p <= kMaxP ? exact_pipeline : exact_global_pipeline];
      [encoder setBuffer:data_buffer offset:0 atIndex:0];
      [encoder setBuffer:out_id_buffer offset:0 atIndex:1];
      [encoder setBuffer:out_dist_buffer offset:0 atIndex:2];
      [encoder setBuffer:params_buffer offset:0 atIndex:3];
      [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [encoder endEncoding];
      run_and_wait(command);
      const auto searched = Clock::now();
      const int* ids = static_cast<const int*>([out_id_buffer contents]);
      const float* internal_distances = static_cast<const float*>([out_dist_buffer contents]);
      Rcpp::IntegerMatrix indices(n, k);
      Rcpp::NumericMatrix distances(n, k);
      for (int i = 0; i < n; ++i) for (int j = 0; j < k; ++j) {
        std::size_t pos = static_cast<std::size_t>(i) * k + j;
        indices(i, j) = ids[pos] + 1;
        distances(i, j) = fastembedr::output_distance(internal_distances[pos], metric);
      }
      Rcpp::List result = Rcpp::List::create(
        Rcpp::Named("indices") = indices,
        Rcpp::Named("distances") = distances,
        Rcpp::Named("backend") = "metal",
        Rcpp::Named("method") = "native_metal_exact",
        Rcpp::Named("metric") = metric_name,
        Rcpp::Named("target_recall") = target,
        Rcpp::Named("timing") = Rcpp::NumericVector::create(
          Rcpp::Named("convert") = std::chrono::duration<double>(t_convert - t0).count(),
          Rcpp::Named("metal_setup") = std::chrono::duration<double>(t_setup - t_convert).count(),
          Rcpp::Named("full_search") = std::chrono::duration<double>(searched - t_setup).count()
        )
      );
      result.attr("backend") = "metal";
      result.attr("method") = "native_metal_exact";
      result.attr("metric") = metric_name;
      result.attr("target_recall") = target;
      [params_buffer release]; [out_dist_buffer release]; [out_id_buffer release]; [data_buffer release];
      return result;
    }
    id<MTLBuffer> offset_map_buffer = [device newBufferWithBytes:feature_offsets.data() length:feature_offsets.size() * sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> feature_buffer = [device newBufferWithBytes:feature_ids.data() length:feature_ids.size() * sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> sign_buffer = [device newBufferWithBytes:feature_signs.data() length:feature_signs.size() * sizeof(std::int8_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> projected_buffer = [device newBufferWithLength:static_cast<std::size_t>(n) * qdim * sizeof(float) options:MTLResourceStorageModeShared];
    ProjectParamsHost project_params{static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(p), static_cast<std::uint32_t>(qdim)};
    id<MTLBuffer> project_params_buffer = [device newBufferWithBytes:&project_params length:sizeof(project_params) options:MTLResourceStorageModeShared];
    {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:project_pipeline];
      [encoder setBuffer:data_buffer offset:0 atIndex:0]; [encoder setBuffer:offset_map_buffer offset:0 atIndex:1];
      [encoder setBuffer:feature_buffer offset:0 atIndex:2]; [encoder setBuffer:sign_buffer offset:0 atIndex:3];
      [encoder setBuffer:projected_buffer offset:0 atIndex:4]; [encoder setBuffer:project_params_buffer offset:0 atIndex:5];
      [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(qdim, 1, 1)];
      [encoder endEncoding]; run_and_wait(command);
    }
    const auto t_project = Clock::now();

    const float* projected = static_cast<const float*>([projected_buffer contents]);
    std::mt19937 rng(4u);
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<float> centroids(static_cast<std::size_t>(nlist) * qdim);
    for (int c = 0; c < nlist; ++c) {
      std::memcpy(centroids.data() + static_cast<std::size_t>(c) * qdim,
                  projected + static_cast<std::size_t>(order[c]) * qdim,
                  static_cast<std::size_t>(qdim) * sizeof(float));
    }
    id<MTLBuffer> centroid_buffer = [device newBufferWithBytes:centroids.data() length:centroids.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> assignment_buffer = [device newBufferWithLength:static_cast<std::size_t>(n) * sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> centroid_sum_buffer = [device newBufferWithLength:static_cast<std::size_t>(nlist) * qdim * sizeof(float) options:MTLResourceStorageModePrivate];
    id<MTLBuffer> centroid_count_buffer = [device newBufferWithLength:static_cast<std::size_t>(nlist) * sizeof(std::uint32_t) options:MTLResourceStorageModePrivate];
    IndexParamsHost index_params{static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(p), static_cast<std::uint32_t>(qdim), static_cast<std::uint32_t>(nlist)};
    id<MTLBuffer> index_params_buffer = [device newBufferWithBytes:&index_params length:sizeof(index_params) options:MTLResourceStorageModeShared];

    for (int iteration = 0; iteration < 4; ++iteration) {
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:assign_pipeline];
      [encoder setBuffer:projected_buffer offset:0 atIndex:0]; [encoder setBuffer:centroid_buffer offset:0 atIndex:1];
      [encoder setBuffer:assignment_buffer offset:0 atIndex:2]; [encoder setBuffer:index_params_buffer offset:0 atIndex:3];
      [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [encoder endEncoding];
      if (iteration < 3) {
        const std::size_t centroid_items = static_cast<std::size_t>(nlist) * qdim;
        id<MTLComputeCommandEncoder> clear_encoder = [command computeCommandEncoder];
        [clear_encoder setComputePipelineState:clear_centroids_pipeline];
        [clear_encoder setBuffer:centroid_sum_buffer offset:0 atIndex:0];
        [clear_encoder setBuffer:centroid_count_buffer offset:0 atIndex:1];
        [clear_encoder setBuffer:index_params_buffer offset:0 atIndex:2];
        [clear_encoder dispatchThreads:MTLSizeMake(std::max<std::size_t>(centroid_items, nlist), 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [clear_encoder endEncoding];

        id<MTLComputeCommandEncoder> accumulate_encoder = [command computeCommandEncoder];
        [accumulate_encoder setComputePipelineState:accumulate_centroids_pipeline];
        [accumulate_encoder setBuffer:projected_buffer offset:0 atIndex:0];
        [accumulate_encoder setBuffer:assignment_buffer offset:0 atIndex:1];
        [accumulate_encoder setBuffer:centroid_sum_buffer offset:0 atIndex:2];
        [accumulate_encoder setBuffer:centroid_count_buffer offset:0 atIndex:3];
        [accumulate_encoder setBuffer:index_params_buffer offset:0 atIndex:4];
        [accumulate_encoder dispatchThreads:MTLSizeMake(static_cast<std::size_t>(n) * qdim, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [accumulate_encoder endEncoding];

        id<MTLComputeCommandEncoder> finalize_encoder = [command computeCommandEncoder];
        [finalize_encoder setComputePipelineState:finalize_centroids_pipeline];
        [finalize_encoder setBuffer:centroid_sum_buffer offset:0 atIndex:0];
        [finalize_encoder setBuffer:centroid_count_buffer offset:0 atIndex:1];
        [finalize_encoder setBuffer:centroid_buffer offset:0 atIndex:2];
        [finalize_encoder setBuffer:index_params_buffer offset:0 atIndex:3];
        [finalize_encoder dispatchThreads:MTLSizeMake(centroid_items, 1, 1)
                              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [finalize_encoder endEncoding];
      }
      run_and_wait(command);
    }
    const auto t_train = Clock::now();

    const int* assignment = static_cast<const int*>([assignment_buffer contents]);
    std::vector<std::uint32_t> list_offsets(static_cast<std::size_t>(nlist) + 1u, 0u);
    for (int i = 0; i < n; ++i) if (assignment[i] >= 0 && assignment[i] < nlist) ++list_offsets[static_cast<std::size_t>(assignment[i]) + 1u];
    for (int c = 0; c < nlist; ++c) list_offsets[c + 1] += list_offsets[c];
    std::vector<std::uint32_t> cursor = list_offsets;
    std::vector<int> list_ids(n, -1);
    for (int i = 0; i < n; ++i) {
      int c = assignment[i];
      if (c >= 0 && c < nlist) list_ids[cursor[c]++] = i;
    }
    id<MTLBuffer> list_offset_buffer = [device newBufferWithBytes:list_offsets.data() length:list_offsets.size() * sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> list_id_buffer = [device newBufferWithBytes:list_ids.data() length:list_ids.size() * sizeof(int) options:MTLResourceStorageModeShared];
    const auto t_pack = Clock::now();

    const int pilot_block_size = std::min(64, n);
    const int pilot_blocks = std::min(4, std::max(1, n / pilot_block_size));
    const int pilot_n = pilot_block_size * pilot_blocks;
    std::vector<int> pilot_offsets(pilot_blocks, 0);
    for (int block = 0; block < pilot_blocks; ++block) {
      pilot_offsets[block] = pilot_blocks == 1 ? 0 : static_cast<int>(
        std::llround(static_cast<double>(block) * (n - pilot_block_size) /
                     static_cast<double>(pilot_blocks - 1))
      );
    }
    id<MTLBuffer> exact_id_buffer = [device newBufferWithLength:static_cast<std::size_t>(pilot_n) * k * sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> exact_dist_buffer = [device newBufferWithLength:static_cast<std::size_t>(pilot_n) * k * sizeof(float) options:MTLResourceStorageModeShared];
    for (int block = 0; block < pilot_blocks; ++block) {
      SearchParamsHost exact_params{
        static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(p),
        static_cast<std::uint32_t>(qdim), static_cast<std::uint32_t>(nlist),
        0u, static_cast<std::uint32_t>(k),
        static_cast<std::uint32_t>(pilot_offsets[block]), 0u
      };
      id<MTLBuffer> exact_params_buffer = [device newBufferWithBytes:&exact_params length:sizeof(exact_params) options:MTLResourceStorageModeShared];
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:exact_pipeline];
      [encoder setBuffer:data_buffer offset:0 atIndex:0];
      [encoder setBuffer:exact_id_buffer
                   offset:static_cast<NSUInteger>(block) * pilot_block_size * k * sizeof(int)
                  atIndex:1];
      [encoder setBuffer:exact_dist_buffer
                   offset:static_cast<NSUInteger>(block) * pilot_block_size * k * sizeof(float)
                  atIndex:2];
      [encoder setBuffer:exact_params_buffer offset:0 atIndex:3];
      [encoder dispatchThreadgroups:MTLSizeMake(pilot_block_size, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [encoder endEncoding]; run_and_wait(command);
      [exact_params_buffer release];
    }
    const auto t_exact = Clock::now();

    bool use_projected_shortlist = n >= 20000 && p >= 256;
    int shortlist_per_group = kInitialShortlistPerGroup;
    int shortlist_size = kMetalSimdGroups * shortlist_per_group;
    int nprobe = std::min(nlist, 8);
    double measured = 0.0;
    const double tune_target = std::min(1.0, target + 0.003);
    std::vector<int> probe_trace;
    std::vector<double> recall_trace;
    id<MTLBuffer> pilot_id_buffer = [device newBufferWithLength:static_cast<std::size_t>(pilot_n) * k * sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> pilot_dist_buffer = [device newBufferWithLength:static_cast<std::size_t>(pilot_n) * k * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> pilot_candidate_buffer = [device newBufferWithLength:static_cast<std::size_t>(pilot_n) * kMetalSimdGroups * kMaxShortlistPerGroup * sizeof(int) options:MTLResourceStorageModePrivate];
    auto evaluate_probe = [&](int probe_count) {
      for (int block = 0; block < pilot_blocks; ++block) {
        SearchParamsHost search_params{
          static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(p),
          static_cast<std::uint32_t>(qdim), static_cast<std::uint32_t>(nlist),
          static_cast<std::uint32_t>(probe_count), static_cast<std::uint32_t>(k),
          static_cast<std::uint32_t>(pilot_offsets[block]),
          static_cast<std::uint32_t>(shortlist_per_group)
        };
        id<MTLBuffer> pilot_params_buffer = [device newBufferWithBytes:&search_params length:sizeof(search_params) options:MTLResourceStorageModeShared];
        const NSUInteger candidate_offset = static_cast<NSUInteger>(block) * pilot_block_size * shortlist_size * sizeof(int);
        const NSUInteger id_offset = static_cast<NSUInteger>(block) * pilot_block_size * k * sizeof(int);
        const NSUInteger distance_offset = static_cast<NSUInteger>(block) * pilot_block_size * k * sizeof(float);
        id<MTLCommandBuffer> command = [queue commandBuffer];
        if (use_projected_shortlist) {
          id<MTLComputeCommandEncoder> shortlist_encoder = [command computeCommandEncoder];
          [shortlist_encoder setComputePipelineState:shortlist_pipeline];
          [shortlist_encoder setBuffer:projected_buffer offset:0 atIndex:0];
          [shortlist_encoder setBuffer:centroid_buffer offset:0 atIndex:1];
          [shortlist_encoder setBuffer:list_offset_buffer offset:0 atIndex:2];
          [shortlist_encoder setBuffer:list_id_buffer offset:0 atIndex:3];
          [shortlist_encoder setBuffer:pilot_candidate_buffer offset:candidate_offset atIndex:4];
          [shortlist_encoder setBuffer:pilot_params_buffer offset:0 atIndex:5];
          [shortlist_encoder dispatchThreadgroups:MTLSizeMake(pilot_block_size, 1, 1)
                                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
          [shortlist_encoder endEncoding];

          id<MTLComputeCommandEncoder> rerank_encoder = [command computeCommandEncoder];
          [rerank_encoder setComputePipelineState:rerank_pipeline];
          [rerank_encoder setBuffer:data_buffer offset:0 atIndex:0];
          [rerank_encoder setBuffer:pilot_candidate_buffer offset:candidate_offset atIndex:1];
          [rerank_encoder setBuffer:pilot_id_buffer offset:id_offset atIndex:2];
          [rerank_encoder setBuffer:pilot_dist_buffer offset:distance_offset atIndex:3];
          [rerank_encoder setBuffer:pilot_params_buffer offset:0 atIndex:4];
          [rerank_encoder dispatchThreadgroups:MTLSizeMake(pilot_block_size, 1, 1)
                                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
          [rerank_encoder endEncoding];
        } else {
          id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
          [encoder setComputePipelineState:ivf_pipeline];
          [encoder setBuffer:data_buffer offset:0 atIndex:0];
          [encoder setBuffer:projected_buffer offset:0 atIndex:1];
          [encoder setBuffer:centroid_buffer offset:0 atIndex:2];
          [encoder setBuffer:list_offset_buffer offset:0 atIndex:3];
          [encoder setBuffer:list_id_buffer offset:0 atIndex:4];
          [encoder setBuffer:pilot_id_buffer offset:id_offset atIndex:5];
          [encoder setBuffer:pilot_dist_buffer offset:distance_offset atIndex:6];
          [encoder setBuffer:pilot_params_buffer offset:0 atIndex:7];
          [encoder dispatchThreadgroups:MTLSizeMake(pilot_block_size, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
          [encoder endEncoding];
        }
        run_and_wait(command);
        [pilot_params_buffer release];
      }
      double value = recall_at_k(static_cast<const int*>([exact_id_buffer contents]), static_cast<const int*>([pilot_id_buffer contents]), pilot_n, k);
      probe_trace.push_back(probe_count);
      recall_trace.push_back(value);
      return value;
    };
    auto tune_nprobe = [&]() {
      nprobe = std::min(nlist, 8);
      int low_fail = 0;
      int high_pass = -1;
      double high_recall = 0.0;
      while (true) {
        measured = evaluate_probe(nprobe);
        if (measured >= tune_target) { high_pass = nprobe; high_recall = measured; break; }
        low_fail = nprobe;
        if (nprobe >= std::min(nlist, kMaxProbe)) { high_pass = nprobe; high_recall = measured; break; }
        nprobe = std::min(std::min(nlist, kMaxProbe), std::max(nprobe + 1, static_cast<int>(std::ceil(nprobe * 1.5))));
      }
      if (high_recall >= tune_target) {
        while (high_pass - low_fail > 2) {
          int mid = low_fail + (high_pass - low_fail) / 2;
          double value = evaluate_probe(mid);
          if (value >= tune_target) { high_pass = mid; high_recall = value; }
          else low_fail = mid;
        }
      }
      nprobe = high_pass;
      measured = high_recall;
    };
    std::vector<int> shortlist_attempts;
    shortlist_attempts.push_back(shortlist_size);
    tune_nprobe();
    for (int expanded_per_group : {96, 128}) {
      if (!use_projected_shortlist || measured >= tune_target) break;
      shortlist_per_group = expanded_per_group;
      shortlist_size = kMetalSimdGroups * shortlist_per_group;
      shortlist_attempts.push_back(shortlist_size);
      probe_trace.clear();
      recall_trace.clear();
      tune_nprobe();
    }
    if (use_projected_shortlist && measured < tune_target) {
      use_projected_shortlist = false;
      probe_trace.clear();
      recall_trace.clear();
      tune_nprobe();
    }
    const auto t_tune = Clock::now();

    SearchParamsHost full_params{
      static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(p),
      static_cast<std::uint32_t>(qdim), static_cast<std::uint32_t>(nlist),
      static_cast<std::uint32_t>(nprobe), static_cast<std::uint32_t>(k), 0u,
      static_cast<std::uint32_t>(use_projected_shortlist ? shortlist_per_group : 0)
    };
    id<MTLBuffer> search_params_buffer = [device newBufferWithBytes:&full_params length:sizeof(full_params) options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_id_buffer = [device newBufferWithLength:static_cast<std::size_t>(n) * k * sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_dist_buffer = [device newBufferWithLength:static_cast<std::size_t>(n) * k * sizeof(float) options:MTLResourceStorageModeShared];
    if (use_projected_shortlist) {
      const int query_batch = std::min(n, 4096);
      id<MTLBuffer> candidate_buffer = [device newBufferWithLength:static_cast<std::size_t>(query_batch) * shortlist_size * sizeof(int) options:MTLResourceStorageModePrivate];
      for (int query_offset = 0; query_offset < n; query_offset += query_batch) {
        const int batch_rows = std::min(query_batch, n - query_offset);
        SearchParamsHost search_params{
          static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(p),
          static_cast<std::uint32_t>(qdim), static_cast<std::uint32_t>(nlist),
          static_cast<std::uint32_t>(nprobe), static_cast<std::uint32_t>(k),
          static_cast<std::uint32_t>(query_offset),
          static_cast<std::uint32_t>(shortlist_per_group)
        };
        std::memcpy([search_params_buffer contents], &search_params, sizeof(search_params));
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> shortlist_encoder = [command computeCommandEncoder];
        [shortlist_encoder setComputePipelineState:shortlist_pipeline];
        [shortlist_encoder setBuffer:projected_buffer offset:0 atIndex:0];
        [shortlist_encoder setBuffer:centroid_buffer offset:0 atIndex:1];
        [shortlist_encoder setBuffer:list_offset_buffer offset:0 atIndex:2];
        [shortlist_encoder setBuffer:list_id_buffer offset:0 atIndex:3];
        [shortlist_encoder setBuffer:candidate_buffer offset:0 atIndex:4];
        [shortlist_encoder setBuffer:search_params_buffer offset:0 atIndex:5];
        [shortlist_encoder dispatchThreadgroups:MTLSizeMake(batch_rows, 1, 1)
                                   threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [shortlist_encoder endEncoding];

        id<MTLComputeCommandEncoder> rerank_encoder = [command computeCommandEncoder];
        [rerank_encoder setComputePipelineState:rerank_pipeline];
        [rerank_encoder setBuffer:data_buffer offset:0 atIndex:0];
        [rerank_encoder setBuffer:candidate_buffer offset:0 atIndex:1];
        [rerank_encoder setBuffer:out_id_buffer
                           offset:static_cast<NSUInteger>(query_offset) * k * sizeof(int)
                          atIndex:2];
        [rerank_encoder setBuffer:out_dist_buffer
                           offset:static_cast<NSUInteger>(query_offset) * k * sizeof(float)
                          atIndex:3];
        [rerank_encoder setBuffer:search_params_buffer offset:0 atIndex:4];
        [rerank_encoder dispatchThreadgroups:MTLSizeMake(batch_rows, 1, 1)
                                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [rerank_encoder endEncoding];
        run_and_wait(command);
      }
      [candidate_buffer release];
    } else {
      SearchParamsHost search_params{
        static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(p),
        static_cast<std::uint32_t>(qdim), static_cast<std::uint32_t>(nlist),
        static_cast<std::uint32_t>(nprobe), static_cast<std::uint32_t>(k), 0u, 0u
      };
      std::memcpy([search_params_buffer contents], &search_params, sizeof(search_params));
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:ivf_pipeline];
      [encoder setBuffer:data_buffer offset:0 atIndex:0]; [encoder setBuffer:projected_buffer offset:0 atIndex:1];
      [encoder setBuffer:centroid_buffer offset:0 atIndex:2]; [encoder setBuffer:list_offset_buffer offset:0 atIndex:3];
      [encoder setBuffer:list_id_buffer offset:0 atIndex:4]; [encoder setBuffer:out_id_buffer offset:0 atIndex:5];
      [encoder setBuffer:out_dist_buffer offset:0 atIndex:6]; [encoder setBuffer:search_params_buffer offset:0 atIndex:7];
      [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [encoder endEncoding]; run_and_wait(command);
    }
    const auto t_search = Clock::now();

    const int* ids = static_cast<const int*>([out_id_buffer contents]);
    const float* d2 = static_cast<const float*>([out_dist_buffer contents]);
    Rcpp::IntegerMatrix indices(n, k);
    Rcpp::NumericMatrix distances(n, k);
    for (int i = 0; i < n; ++i) for (int j = 0; j < k; ++j) {
      std::size_t pos = static_cast<std::size_t>(i) * k + j;
      indices(i, j) = ids[pos] + 1;
      distances(i, j) = fastembedr::output_distance(d2[pos], metric);
    }
    Rcpp::List result = Rcpp::List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances,
      Rcpp::Named("backend") = "metal",
      Rcpp::Named("method") = "native_metal_ivf",
      Rcpp::Named("metric") = metric_name,
      Rcpp::Named("target_recall") = target,
      Rcpp::Named("nlist") = nlist,
      Rcpp::Named("nprobe") = nprobe,
      Rcpp::Named("projection_dim") = qdim,
      Rcpp::Named("search_strategy") = use_projected_shortlist ?
        "projected_shortlist_exact_rerank" : "direct_exact_rerank",
      Rcpp::Named("shortlist_size") = use_projected_shortlist ? shortlist_size : 0,
      Rcpp::Named("shortlist_attempts") = shortlist_attempts,
      Rcpp::Named("pilot_recall") = measured,
      Rcpp::Named("target_met") = measured >= target,
      Rcpp::Named("probe_trace") = probe_trace,
      Rcpp::Named("recall_trace") = recall_trace,
      Rcpp::Named("timing") = Rcpp::NumericVector::create(
        Rcpp::Named("convert") = std::chrono::duration<double>(t_convert - t0).count(),
        Rcpp::Named("metal_setup") = std::chrono::duration<double>(t_setup - t_convert).count(),
        Rcpp::Named("projection") = std::chrono::duration<double>(t_project - t_setup).count(),
        Rcpp::Named("ivf_training") = std::chrono::duration<double>(t_train - t_project).count(),
        Rcpp::Named("list_pack") = std::chrono::duration<double>(t_pack - t_train).count(),
        Rcpp::Named("pilot_exact") = std::chrono::duration<double>(t_exact - t_pack).count(),
        Rcpp::Named("pilot_tune") = std::chrono::duration<double>(t_tune - t_exact).count(),
        Rcpp::Named("full_search") = std::chrono::duration<double>(t_search - t_tune).count()
      )
    );
    result.attr("backend") = "metal";
    result.attr("method") = "native_metal_ivf";
    result.attr("metric") = metric_name;
    result.attr("target_recall") = target;
    result.attr("target_met") = measured >= target;

    [out_dist_buffer release]; [out_id_buffer release]; [search_params_buffer release];
    [pilot_candidate_buffer release];
    [pilot_dist_buffer release]; [pilot_id_buffer release];
    [exact_dist_buffer release]; [exact_id_buffer release]; [list_id_buffer release];
    [list_offset_buffer release]; [index_params_buffer release]; [centroid_count_buffer release];
    [centroid_sum_buffer release]; [assignment_buffer release];
    [centroid_buffer release]; [project_params_buffer release]; [projected_buffer release];
    [sign_buffer release]; [feature_buffer release]; [offset_map_buffer release]; [data_buffer release];
    return result;
  }
}

Rcpp::List native_metal_query_knn_impl(SEXP data_sexp,
                                       SEXP query_sexp,
                                       int k,
                                       const std::string& requested_method,
                                       const std::string& metric_name,
                                       double target) {
  using Clock = std::chrono::steady_clock;
  const auto t0 = Clock::now();
  const fastembedr::KnnMetric metric =
    fastembedr::parse_knn_metric(metric_name);
  fastembedr::FloatMatrix reference =
    fastembedr::matrix_to_row_major_float(data_sexp, metric);
  fastembedr::FloatMatrix query =
    fastembedr::matrix_to_row_major_float(query_sexp, metric);
  const auto t_convert = Clock::now();
  if (reference.nrow < 1 || query.nrow < 1 ||
      reference.ncol < 1 || reference.ncol != query.ncol ||
      reference.ncol > kMaxGlobalP || k < 1 || k > kMaxK ||
      k > reference.nrow) {
    Rcpp::stop(
      "Native Metal query KNN requires matching 1-%d dimensional matrices "
      "and k <= min(%d, nrow(reference)).",
      kMaxGlobalP, kMaxK
    );
  }
  std::string method = requested_method;
  if (method == "auto") {
    method = reference.nrow < 4096 ? "exact" : "ivf";
  }
  if (method != "exact" && method != "ivf") {
    Rcpp::stop(
      "Native Metal reference-query KNN supports `exact`, `ivf`, or `auto`."
    );
  }
  if (method == "ivf" && reference.ncol > kMaxP) {
    Rcpp::stop(
      "Native Metal IVF query KNN supports at most %d dimensions; use exact "
      "search for this %d-dimensional input.",
      kMaxP, reference.ncol
    );
  }
  const int qdim = projection_dim(reference.ncol);
  const int nlist = std::max(
    16,
    std::min(
      kMaxLists,
      static_cast<int>(
        std::ceil(4.0 * std::sqrt(static_cast<double>(reference.nrow)))
      )
    )
  );

  std::vector<std::uint32_t> feature_offsets(
    static_cast<std::size_t>(qdim) + 1u, 0u
  );
  std::vector<std::vector<std::pair<std::uint32_t, std::int8_t>>> buckets(
    qdim
  );
  for (int d = 0; d < reference.ncol; ++d) {
    const std::uint32_t h =
      static_cast<std::uint32_t>(d + 1) * 2654435761u;
    const int bucket = static_cast<int>(
      h & static_cast<std::uint32_t>(qdim - 1)
    );
    const std::int8_t sign =
      ((h >> 17u) & 1u) ? std::int8_t(1) : std::int8_t(-1);
    buckets[bucket].push_back(
      {static_cast<std::uint32_t>(d), sign}
    );
  }
  std::vector<std::uint32_t> feature_ids;
  std::vector<std::int8_t> feature_signs;
  feature_ids.reserve(reference.ncol);
  feature_signs.reserve(reference.ncol);
  for (int bucket = 0; bucket < qdim; ++bucket) {
    feature_offsets[bucket] =
      static_cast<std::uint32_t>(feature_ids.size());
    for (const auto& item : buckets[bucket]) {
      feature_ids.push_back(item.first);
      feature_signs.push_back(item.second);
    }
  }
  feature_offsets[qdim] =
    static_cast<std::uint32_t>(feature_ids.size());

  @autoreleasepool {
    MetalKnnState& state = metal_knn_state();
    id<MTLDevice> device = state.device;
    const auto t_setup = Clock::now();
    id<MTLBuffer> reference_buffer = [device
      newBufferWithBytes:reference.values.data()
      length:reference.values.size() * sizeof(float)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> query_buffer = [device
      newBufferWithBytes:query.values.data()
      length:query.values.size() * sizeof(float)
      options:MTLResourceStorageModeShared];
    const std::size_t output_items =
      static_cast<std::size_t>(query.nrow) * k;
    id<MTLBuffer> out_id_buffer = [device
      newBufferWithLength:output_items * sizeof(int)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_dist_buffer = [device
      newBufferWithLength:output_items * sizeof(float)
      options:MTLResourceStorageModeShared];

    auto exact_search = [&](id<MTLBuffer> queries,
                            int n_queries,
                            id<MTLBuffer> output_ids,
                            id<MTLBuffer> output_distances) {
      QuerySearchParamsHost params{
        static_cast<std::uint32_t>(reference.nrow),
        static_cast<std::uint32_t>(n_queries),
        static_cast<std::uint32_t>(reference.ncol),
        static_cast<std::uint32_t>(k)
      };
      id<MTLBuffer> params_buffer = [device
        newBufferWithBytes:&params
        length:sizeof(params)
        options:MTLResourceStorageModeShared];
      id<MTLCommandBuffer> command = [state.queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:
        reference.ncol <= kMaxP
          ? state.exact_query_pipeline
          : state.exact_query_global_pipeline];
      [encoder setBuffer:reference_buffer offset:0 atIndex:0];
      [encoder setBuffer:queries offset:0 atIndex:1];
      [encoder setBuffer:output_ids offset:0 atIndex:2];
      [encoder setBuffer:output_distances offset:0 atIndex:3];
      [encoder setBuffer:params_buffer offset:0 atIndex:4];
      [encoder
        dispatchThreadgroups:MTLSizeMake(n_queries, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [encoder endEncoding];
      run_and_wait(command);
      [params_buffer release];
    };

    if (method == "exact") {
      exact_search(
        query_buffer, query.nrow, out_id_buffer, out_dist_buffer
      );
      const auto t_search = Clock::now();
      const int* ids =
        static_cast<const int*>([out_id_buffer contents]);
      const float* internal_distances =
        static_cast<const float*>([out_dist_buffer contents]);
      Rcpp::IntegerMatrix indices(query.nrow, k);
      Rcpp::NumericMatrix distances(query.nrow, k);
      for (int i = 0; i < query.nrow; ++i) {
        for (int j = 0; j < k; ++j) {
          const std::size_t pos = static_cast<std::size_t>(i) * k + j;
          indices(i, j) = ids[pos] + 1;
          distances(i, j) =
            fastembedr::output_distance(internal_distances[pos], metric);
        }
      }
      const auto t_copy = Clock::now();
      Rcpp::List result = Rcpp::List::create(
        Rcpp::Named("indices") = indices,
        Rcpp::Named("distances") = distances,
        Rcpp::Named("backend") = "metal",
        Rcpp::Named("method") = "native_metal_exact_query",
        Rcpp::Named("metric") = metric_name,
        Rcpp::Named("target_recall") = target,
        Rcpp::Named("pilot_recall") = 1.0,
        Rcpp::Named("target_met") = true,
        Rcpp::Named("exact") = true,
        Rcpp::Named("exclude_self") = false,
        Rcpp::Named("n_reference") = reference.nrow,
        Rcpp::Named("n_query") = query.nrow,
        Rcpp::Named("timing") = Rcpp::NumericVector::create(
          Rcpp::Named("convert") =
            std::chrono::duration<double>(t_convert - t0).count(),
          Rcpp::Named("metal_setup") =
            std::chrono::duration<double>(t_setup - t_convert).count(),
          Rcpp::Named("search") =
            std::chrono::duration<double>(t_search - t_setup).count(),
          Rcpp::Named("copy_to_R") =
            std::chrono::duration<double>(t_copy - t_search).count()
        )
      );
      result.attr("backend") = "metal";
      result.attr("method") = "native_metal_exact_query";
      result.attr("metric") = metric_name;
      result.attr("exact") = true;
      result.attr("target_met") = true;
      result.attr("exclude_self") = false;
      [out_dist_buffer release];
      [out_id_buffer release];
      [query_buffer release];
      [reference_buffer release];
      return result;
    }

    id<MTLBuffer> offset_map_buffer = [device
      newBufferWithBytes:feature_offsets.data()
      length:feature_offsets.size() * sizeof(std::uint32_t)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> feature_buffer = [device
      newBufferWithBytes:feature_ids.data()
      length:feature_ids.size() * sizeof(std::uint32_t)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> sign_buffer = [device
      newBufferWithBytes:feature_signs.data()
      length:feature_signs.size() * sizeof(std::int8_t)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> reference_projected_buffer = [device
      newBufferWithLength:
        static_cast<std::size_t>(reference.nrow) * qdim * sizeof(float)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> query_projected_buffer = [device
      newBufferWithLength:
        static_cast<std::size_t>(query.nrow) * qdim * sizeof(float)
      options:MTLResourceStorageModeShared];

    auto project = [&](id<MTLBuffer> input_buffer,
                       id<MTLBuffer> projected_buffer,
                       int n_rows) {
      ProjectParamsHost params{
        static_cast<std::uint32_t>(n_rows),
        static_cast<std::uint32_t>(reference.ncol),
        static_cast<std::uint32_t>(qdim)
      };
      id<MTLBuffer> params_buffer = [device
        newBufferWithBytes:&params
        length:sizeof(params)
        options:MTLResourceStorageModeShared];
      id<MTLCommandBuffer> command = [state.queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:state.project_pipeline];
      [encoder setBuffer:input_buffer offset:0 atIndex:0];
      [encoder setBuffer:offset_map_buffer offset:0 atIndex:1];
      [encoder setBuffer:feature_buffer offset:0 atIndex:2];
      [encoder setBuffer:sign_buffer offset:0 atIndex:3];
      [encoder setBuffer:projected_buffer offset:0 atIndex:4];
      [encoder setBuffer:params_buffer offset:0 atIndex:5];
      [encoder
        dispatchThreadgroups:MTLSizeMake(n_rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(qdim, 1, 1)];
      [encoder endEncoding];
      run_and_wait(command);
      [params_buffer release];
    };
    project(
      reference_buffer, reference_projected_buffer, reference.nrow
    );
    project(query_buffer, query_projected_buffer, query.nrow);
    const auto t_project = Clock::now();

    const float* reference_projected =
      static_cast<const float*>([reference_projected_buffer contents]);
    std::mt19937 rng(4u);
    std::vector<int> order(reference.nrow);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<float> centroids(
      static_cast<std::size_t>(nlist) * qdim
    );
    for (int centroid = 0; centroid < nlist; ++centroid) {
      std::memcpy(
        centroids.data() + static_cast<std::size_t>(centroid) * qdim,
        reference_projected +
          static_cast<std::size_t>(order[centroid]) * qdim,
        static_cast<std::size_t>(qdim) * sizeof(float)
      );
    }
    id<MTLBuffer> centroid_buffer = [device
      newBufferWithBytes:centroids.data()
      length:centroids.size() * sizeof(float)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> assignment_buffer = [device
      newBufferWithLength:
        static_cast<std::size_t>(reference.nrow) * sizeof(int)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> centroid_sum_buffer = [device
      newBufferWithLength:
        static_cast<std::size_t>(nlist) * qdim * sizeof(float)
      options:MTLResourceStorageModePrivate];
    id<MTLBuffer> centroid_count_buffer = [device
      newBufferWithLength:
        static_cast<std::size_t>(nlist) * sizeof(std::uint32_t)
      options:MTLResourceStorageModePrivate];
    IndexParamsHost index_params{
      static_cast<std::uint32_t>(reference.nrow),
      static_cast<std::uint32_t>(reference.ncol),
      static_cast<std::uint32_t>(qdim),
      static_cast<std::uint32_t>(nlist)
    };
    id<MTLBuffer> index_params_buffer = [device
      newBufferWithBytes:&index_params
      length:sizeof(index_params)
      options:MTLResourceStorageModeShared];

    for (int iteration = 0; iteration < 4; ++iteration) {
      id<MTLCommandBuffer> command = [state.queue commandBuffer];
      id<MTLComputeCommandEncoder> assignment_encoder =
        [command computeCommandEncoder];
      [assignment_encoder setComputePipelineState:state.assign_pipeline];
      [assignment_encoder
        setBuffer:reference_projected_buffer offset:0 atIndex:0];
      [assignment_encoder setBuffer:centroid_buffer offset:0 atIndex:1];
      [assignment_encoder setBuffer:assignment_buffer offset:0 atIndex:2];
      [assignment_encoder setBuffer:index_params_buffer offset:0 atIndex:3];
      [assignment_encoder
        dispatchThreadgroups:MTLSizeMake(reference.nrow, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [assignment_encoder endEncoding];
      if (iteration < 3) {
        const std::size_t centroid_items =
          static_cast<std::size_t>(nlist) * qdim;
        id<MTLComputeCommandEncoder> clear_encoder =
          [command computeCommandEncoder];
        [clear_encoder
          setComputePipelineState:state.clear_centroids_pipeline];
        [clear_encoder setBuffer:centroid_sum_buffer offset:0 atIndex:0];
        [clear_encoder setBuffer:centroid_count_buffer offset:0 atIndex:1];
        [clear_encoder setBuffer:index_params_buffer offset:0 atIndex:2];
        [clear_encoder
          dispatchThreads:
            MTLSizeMake(std::max<std::size_t>(centroid_items, nlist), 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [clear_encoder endEncoding];

        id<MTLComputeCommandEncoder> accumulate_encoder =
          [command computeCommandEncoder];
        [accumulate_encoder
          setComputePipelineState:state.accumulate_centroids_pipeline];
        [accumulate_encoder
          setBuffer:reference_projected_buffer offset:0 atIndex:0];
        [accumulate_encoder setBuffer:assignment_buffer offset:0 atIndex:1];
        [accumulate_encoder setBuffer:centroid_sum_buffer offset:0 atIndex:2];
        [accumulate_encoder
          setBuffer:centroid_count_buffer offset:0 atIndex:3];
        [accumulate_encoder setBuffer:index_params_buffer offset:0 atIndex:4];
        [accumulate_encoder
          dispatchThreads:
            MTLSizeMake(
              static_cast<std::size_t>(reference.nrow) * qdim, 1, 1
            )
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [accumulate_encoder endEncoding];

        id<MTLComputeCommandEncoder> finalize_encoder =
          [command computeCommandEncoder];
        [finalize_encoder
          setComputePipelineState:state.finalize_centroids_pipeline];
        [finalize_encoder setBuffer:centroid_sum_buffer offset:0 atIndex:0];
        [finalize_encoder
          setBuffer:centroid_count_buffer offset:0 atIndex:1];
        [finalize_encoder setBuffer:centroid_buffer offset:0 atIndex:2];
        [finalize_encoder setBuffer:index_params_buffer offset:0 atIndex:3];
        [finalize_encoder
          dispatchThreads:MTLSizeMake(centroid_items, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [finalize_encoder endEncoding];
      }
      run_and_wait(command);
    }
    const auto t_train = Clock::now();

    const int* assignment =
      static_cast<const int*>([assignment_buffer contents]);
    std::vector<std::uint32_t> list_offsets(
      static_cast<std::size_t>(nlist) + 1u, 0u
    );
    for (int row = 0; row < reference.nrow; ++row) {
      if (assignment[row] >= 0 && assignment[row] < nlist) {
        ++list_offsets[static_cast<std::size_t>(assignment[row]) + 1u];
      }
    }
    for (int list = 0; list < nlist; ++list) {
      list_offsets[list + 1] += list_offsets[list];
    }
    std::vector<std::uint32_t> cursor = list_offsets;
    std::vector<int> list_ids(reference.nrow, -1);
    for (int row = 0; row < reference.nrow; ++row) {
      const int list = assignment[row];
      if (list >= 0 && list < nlist) {
        list_ids[cursor[list]++] = row;
      }
    }
    id<MTLBuffer> list_offset_buffer = [device
      newBufferWithBytes:list_offsets.data()
      length:list_offsets.size() * sizeof(std::uint32_t)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> list_id_buffer = [device
      newBufferWithBytes:list_ids.data()
      length:list_ids.size() * sizeof(int)
      options:MTLResourceStorageModeShared];
    const auto t_pack = Clock::now();

    auto ivf_search = [&](id<MTLBuffer> queries,
                          id<MTLBuffer> projected_queries,
                          int n_queries,
                          int nprobe,
                          id<MTLBuffer> output_ids,
                          id<MTLBuffer> output_distances) {
      QueryIvfParamsHost params{
        static_cast<std::uint32_t>(reference.nrow),
        static_cast<std::uint32_t>(n_queries),
        static_cast<std::uint32_t>(reference.ncol),
        static_cast<std::uint32_t>(qdim),
        static_cast<std::uint32_t>(nlist),
        static_cast<std::uint32_t>(nprobe),
        static_cast<std::uint32_t>(k)
      };
      id<MTLBuffer> params_buffer = [device
        newBufferWithBytes:&params
        length:sizeof(params)
        options:MTLResourceStorageModeShared];
      id<MTLCommandBuffer> command = [state.queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:state.ivf_query_pipeline];
      [encoder setBuffer:reference_buffer offset:0 atIndex:0];
      [encoder setBuffer:queries offset:0 atIndex:1];
      [encoder setBuffer:projected_queries offset:0 atIndex:2];
      [encoder setBuffer:centroid_buffer offset:0 atIndex:3];
      [encoder setBuffer:list_offset_buffer offset:0 atIndex:4];
      [encoder setBuffer:list_id_buffer offset:0 atIndex:5];
      [encoder setBuffer:output_ids offset:0 atIndex:6];
      [encoder setBuffer:output_distances offset:0 atIndex:7];
      [encoder setBuffer:params_buffer offset:0 atIndex:8];
      [encoder
        dispatchThreadgroups:MTLSizeMake(n_queries, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      [encoder endEncoding];
      run_and_wait(command);
      [params_buffer release];
    };

    const int pilot_n = std::min(query.nrow, 128);
    id<MTLBuffer> exact_pilot_ids = [device
      newBufferWithLength:
        static_cast<std::size_t>(pilot_n) * k * sizeof(int)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> exact_pilot_distances = [device
      newBufferWithLength:
        static_cast<std::size_t>(pilot_n) * k * sizeof(float)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> ivf_pilot_ids = [device
      newBufferWithLength:
        static_cast<std::size_t>(pilot_n) * k * sizeof(int)
      options:MTLResourceStorageModeShared];
    id<MTLBuffer> ivf_pilot_distances = [device
      newBufferWithLength:
        static_cast<std::size_t>(pilot_n) * k * sizeof(float)
      options:MTLResourceStorageModeShared];
    exact_search(
      query_buffer, pilot_n, exact_pilot_ids, exact_pilot_distances
    );
    const auto t_exact = Clock::now();

    int nprobe = std::min(nlist, 8);
    int low_fail = 0;
    int high_pass = 0;
    double measured = 0.0;
    const double tune_target = std::min(1.0, target + 0.002);
    std::vector<int> probe_trace;
    std::vector<double> recall_trace;
    while (true) {
      ivf_search(
        query_buffer, query_projected_buffer, pilot_n, nprobe,
        ivf_pilot_ids, ivf_pilot_distances
      );
      measured = recall_at_k(
        static_cast<const int*>([exact_pilot_ids contents]),
        static_cast<const int*>([ivf_pilot_ids contents]),
        pilot_n,
        k
      );
      probe_trace.push_back(nprobe);
      recall_trace.push_back(measured);
      if (measured >= tune_target) {
        high_pass = nprobe;
        break;
      }
      low_fail = nprobe;
      if (nprobe >= std::min(nlist, kMaxProbe)) {
        break;
      }
      nprobe = std::min(
        std::min(nlist, kMaxProbe),
        std::max(nprobe + 1, static_cast<int>(std::ceil(nprobe * 1.5)))
      );
    }
    while (high_pass > 0 && high_pass - low_fail > 1) {
      const int candidate = low_fail + (high_pass - low_fail) / 2;
      ivf_search(
        query_buffer, query_projected_buffer, pilot_n, candidate,
        ivf_pilot_ids, ivf_pilot_distances
      );
      const double candidate_recall = recall_at_k(
        static_cast<const int*>([exact_pilot_ids contents]),
        static_cast<const int*>([ivf_pilot_ids contents]),
        pilot_n,
        k
      );
      probe_trace.push_back(candidate);
      recall_trace.push_back(candidate_recall);
      if (candidate_recall >= tune_target) {
        high_pass = candidate;
        measured = candidate_recall;
      } else {
        low_fail = candidate;
      }
    }
    if (high_pass > 0) {
      nprobe = high_pass;
    }
    const auto t_tune = Clock::now();
    ivf_search(
      query_buffer, query_projected_buffer, query.nrow, nprobe,
      out_id_buffer, out_dist_buffer
    );
    const auto t_search = Clock::now();

    const int* ids =
      static_cast<const int*>([out_id_buffer contents]);
    const float* internal_distances =
      static_cast<const float*>([out_dist_buffer contents]);
    Rcpp::IntegerMatrix indices(query.nrow, k);
    Rcpp::NumericMatrix distances(query.nrow, k);
    for (int i = 0; i < query.nrow; ++i) {
      for (int j = 0; j < k; ++j) {
        const std::size_t pos = static_cast<std::size_t>(i) * k + j;
        indices(i, j) = ids[pos] + 1;
        distances(i, j) =
          fastembedr::output_distance(internal_distances[pos], metric);
      }
    }
    const auto t_copy = Clock::now();
    Rcpp::List result = Rcpp::List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances,
      Rcpp::Named("backend") = "metal",
      Rcpp::Named("method") = "native_metal_ivf_query",
      Rcpp::Named("metric") = metric_name,
      Rcpp::Named("target_recall") = target,
      Rcpp::Named("pilot_recall") = measured,
      Rcpp::Named("target_met") = measured >= target,
      Rcpp::Named("exact") = false,
      Rcpp::Named("exclude_self") = false,
      Rcpp::Named("n_reference") = reference.nrow,
      Rcpp::Named("n_query") = query.nrow,
      Rcpp::Named("nlist") = nlist,
      Rcpp::Named("nprobe") = nprobe,
      Rcpp::Named("projection_dim") = qdim,
      Rcpp::Named("probe_trace") = probe_trace,
      Rcpp::Named("recall_trace") = recall_trace,
      Rcpp::Named("timing") = Rcpp::NumericVector::create(
        Rcpp::Named("convert") =
          std::chrono::duration<double>(t_convert - t0).count(),
        Rcpp::Named("metal_setup") =
          std::chrono::duration<double>(t_setup - t_convert).count(),
        Rcpp::Named("projection") =
          std::chrono::duration<double>(t_project - t_setup).count(),
        Rcpp::Named("ivf_training") =
          std::chrono::duration<double>(t_train - t_project).count(),
        Rcpp::Named("list_pack") =
          std::chrono::duration<double>(t_pack - t_train).count(),
        Rcpp::Named("pilot_exact") =
          std::chrono::duration<double>(t_exact - t_pack).count(),
        Rcpp::Named("pilot_tune") =
          std::chrono::duration<double>(t_tune - t_exact).count(),
        Rcpp::Named("search") =
          std::chrono::duration<double>(t_search - t_tune).count(),
        Rcpp::Named("copy_to_R") =
          std::chrono::duration<double>(t_copy - t_search).count()
      )
    );
    result.attr("backend") = "metal";
    result.attr("method") = "native_metal_ivf_query";
    result.attr("metric") = metric_name;
    result.attr("exact") = false;
    result.attr("target_met") = measured >= target;
    result.attr("exclude_self") = false;

    [ivf_pilot_distances release];
    [ivf_pilot_ids release];
    [exact_pilot_distances release];
    [exact_pilot_ids release];
    [list_id_buffer release];
    [list_offset_buffer release];
    [index_params_buffer release];
    [centroid_count_buffer release];
    [centroid_sum_buffer release];
    [assignment_buffer release];
    [centroid_buffer release];
    [query_projected_buffer release];
    [reference_projected_buffer release];
    [sign_buffer release];
    [feature_buffer release];
    [offset_map_buffer release];
    [out_dist_buffer release];
    [out_id_buffer release];
    [query_buffer release];
    [reference_buffer release];
    return result;
  }
}
