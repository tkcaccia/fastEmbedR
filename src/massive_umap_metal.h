#ifndef FASTEMBEDR_MASSIVE_UMAP_METAL_H
#define FASTEMBEDR_MASSIVE_UMAP_METAL_H

#include <cstdint>
#include <cstddef>
#include <functional>
#include <string>

void* fastembedr_umap_spectral_metal_create(const std::string& offsets,
    const std::string& indices, const std::string& weights,
    int rows, int block, std::uint64_t edges, const double* inv_degree,
    double device_budget);
void fastembedr_umap_spectral_metal_multiply(void* context,
    const double* input, double* result);
std::uint64_t fastembedr_umap_spectral_metal_buffers(const void* context);
void fastembedr_umap_spectral_metal_destroy(void* context);

struct FastembedrUmapCompactEdge {
  std::uint32_t head;
  std::uint32_t tail;
  float period;
};

constexpr std::uint32_t kMetalUmapCompactTileEdges = 1u << 29;

struct FastembedrUmapCompactResult {
  std::uint64_t bytes = 0;
  double prepare_seconds = 0;
  double device_seconds = 0;
  std::uint64_t active_edges = 0;
  std::uint64_t coordinates_changed = 0;
  std::uint32_t epochs_dispatched = 0;
};

FastembedrUmapCompactResult fastembedr_massive_umap_metal_compact(
    float* layout, int n, int epochs, int negatives, double learning_rate,
    double a, double b, double repulsion, int seed, std::uint64_t edge_bound,
    double memory_limit,
    const std::function<std::size_t(FastembedrUmapCompactEdge*)>& fill);

extern "C" {
void* fastembedr_massive_umap_metal_create(const float* init,
    int vertices, int dimensions, int epochs, int negatives,
    double learning_rate, double a, double b, double repulsion,
    int seed);
std::uint64_t fastembedr_massive_umap_metal_step(void* context,
    const int* heads, const int* tails, const int* negative_counts,
    const std::uint64_t* edge_ordinals, int count, int epoch);
void fastembedr_massive_umap_metal_download(void* context,
    float* output);
std::uint64_t fastembedr_massive_umap_metal_peak_buffers(const void* context);
void fastembedr_massive_umap_metal_destroy(void* context);
}

#endif
