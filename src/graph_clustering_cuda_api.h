#ifndef FASTEMBEDR_GRAPH_CLUSTERING_CUDA_API_H
#define FASTEMBEDR_GRAPH_CLUSTERING_CUDA_API_H

#include "graph_clustering_accel_common.h"

bool graph_clustering_cuda_available_impl();

fastembedr_graph_accel::MoveResult graph_local_move_cuda_impl(
  const fastembedr_graph_accel::Graph& graph,
  const std::vector<int>& initial,
  const std::vector<int>* parent,
  int max_passes,
  float resolution,
  std::uint64_t seed,
  bool refinement
);

fastembedr_graph_accel::MoveResult graph_local_move_cuda_devices_impl(
  const fastembedr_graph_accel::Graph& graph,
  const std::vector<int>& initial,
  const std::vector<int>* parent,
  int max_passes,
  float resolution,
  std::uint64_t seed,
  bool refinement,
  const std::vector<int>& devices
);

void* graph_stream_cuda_create(const std::vector<float>& degree,
  const std::vector<int>& initial, float graph_volume,
  float resolution, std::uint64_t seed, int chunk_rows,
  int edge_capacity, const std::vector<int>* parent = nullptr,
  const std::vector<float>* cuts = nullptr,
  const std::vector<float>* parent_volume = nullptr);
void graph_stream_cuda_begin_pass(void* handle, int pass);
void graph_stream_cuda_set_color(void* handle, int color);
void graph_stream_cuda_chunk(void* handle,
  const std::vector<int>& row_ptr,
  const std::vector<int>& col_idx,
  const std::vector<float>& weight, int first);
void graph_stream_cuda_fragment(void* handle,
  const std::vector<int>& col_idx,
  const std::vector<float>& weight, int row, bool last);
std::uint32_t graph_stream_cuda_end_pass(void* handle);
std::vector<int> graph_stream_cuda_labels(void* handle);
std::vector<int> graph_stream_cuda_counts(void* handle);
void graph_stream_cuda_destroy(void* handle);

#endif
