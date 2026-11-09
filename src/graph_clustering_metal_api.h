#ifndef FASTEMBEDR_GRAPH_CLUSTERING_METAL_API_H
#define FASTEMBEDR_GRAPH_CLUSTERING_METAL_API_H

#include "graph_clustering_accel_common.h"
#include <cstdint>
#include <vector>

bool graph_clustering_metal_available_impl();
std::string graph_clustering_metal_error_impl();

fastembedr_graph_accel::MoveResult graph_local_move_metal_impl(
  const fastembedr_graph_accel::Graph& graph,
  const std::vector<int>& initial,
  const std::vector<int>* parent,
  int max_passes,
  float resolution,
  std::uint64_t seed,
  bool refinement
);

void* graph_stream_metal_create(const std::vector<float>& degree,
                                const std::vector<int>& initial,
                                float graph_volume, float resolution,
                                std::uint64_t seed,
                                const std::vector<int>* parent = nullptr,
                                const std::vector<float>* cuts = nullptr,
                                const std::vector<float>* parent_volume = nullptr);
void graph_stream_metal_begin_pass(void* handle, int pass);
void graph_stream_metal_set_color(void* handle, int color);
void graph_stream_metal_chunk(void* handle,
                              const std::vector<int>& row_ptr,
                              const std::vector<int>& col_idx,
                              const std::vector<float>& weight,
                              std::uint32_t first);
void graph_stream_metal_fragment(void* handle,
                                 const std::vector<int>& col_idx,
                                 const std::vector<float>& weight,
                                 std::uint32_t row, bool last);
std::uint32_t graph_stream_metal_end_pass(void* handle);
std::vector<int> graph_stream_metal_labels(void* handle);
std::vector<int> graph_stream_metal_counts(void* handle);
void graph_stream_metal_destroy(void* handle);

#endif
