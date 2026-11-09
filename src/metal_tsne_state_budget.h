#ifndef FASTEMBEDR_METAL_TSNE_STATE_BUDGET_H
#define FASTEMBEDR_METAL_TSNE_STATE_BUDGET_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

struct MetalTsneStateBudget {
  std::uint64_t layout_state_bytes;
  std::uint64_t graph_upload_bytes;
  std::uint64_t fft_reserve_bytes;
  std::uint64_t total_bytes;
};

inline std::uint64_t metal_tsne_checked_add(
    std::uint64_t left, std::uint64_t right) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw std::overflow_error("Metal t-SNE state size exceeds uint64.");
  }
  return left + right;
}

inline std::uint64_t metal_tsne_checked_mul(
    std::uint64_t left, std::uint64_t right) {
  if (left && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::overflow_error("Metal t-SNE state size exceeds uint64.");
  }
  return left * right;
}

inline MetalTsneStateBudget metal_tsne_state_budget(
    std::uint64_t rows, std::uint32_t dimensions,
    std::uint64_t edges, std::uint32_t edge_capacity,
    bool stream) {
  if (rows < 2 || (dimensions != 2 && dimensions != 3) ||
      (stream && edge_capacity == 0)) {
    throw std::invalid_argument("Invalid Metal t-SNE state dimensions.");
  }
  const auto coordinates = metal_tsne_checked_mul(rows, dimensions);
  const auto layout = metal_tsne_checked_add(
    metal_tsne_checked_mul(coordinates, 24),
    metal_tsne_checked_mul(rows, 8));
  const auto upload = stream ? metal_tsne_checked_add(
    metal_tsne_checked_mul(
      std::min<std::uint64_t>(edges, edge_capacity), 8),
    metal_tsne_checked_mul(
      std::min<std::uint64_t>(rows, 262144) + 1, 8)) :
    metal_tsne_checked_add(
      metal_tsne_checked_mul(edges, 8),
      metal_tsne_checked_mul(metal_tsne_checked_add(rows, 1), 8));
  constexpr std::uint64_t fft_reserve = std::uint64_t{1} << 30;
  const auto total = metal_tsne_checked_add(
    metal_tsne_checked_add(layout, upload), fft_reserve);
  return {layout, upload, fft_reserve, total};
}

inline bool metal_tsne_state_fits(
    const MetalTsneStateBudget& plan, std::uint64_t current,
    std::uint64_t recommended) {
  if (recommended == 0) return false;
  const auto cap = recommended / 10 * 7;
  return current <= cap && plan.total_bytes <= cap - current;
}

#endif
