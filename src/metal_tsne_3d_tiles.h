#ifndef FASTEMBEDR_METAL_TSNE_3D_TILES_H
#define FASTEMBEDR_METAL_TSNE_3D_TILES_H

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

struct MetalTsne3dTile {
  std::uint32_t first_row;
  std::uint32_t rows;
  std::uint64_t first_edge;
  std::uint64_t last_edge;
  bool split_row;
};

inline std::vector<MetalTsne3dTile> metal_tsne_3d_tiles(
    const std::uint64_t* offsets, std::uint32_t rows,
    std::uint32_t edge_capacity, std::uint32_t row_capacity) {
  if (offsets == nullptr || rows == 0 || edge_capacity == 0 ||
      row_capacity == 0) {
    throw std::invalid_argument("Invalid Metal t-SNE tile controls.");
  }
  for (std::uint32_t row = 0; row < rows; ++row) {
    if (offsets[row + 1] < offsets[row]) {
      throw std::invalid_argument("Metal t-SNE offsets decrease.");
    }
  }
  std::vector<MetalTsne3dTile> tiles;
  for (std::uint32_t row = 0; row < rows;) {
    const auto begin = offsets[row];
    const auto end = offsets[row + 1];
    if (end - begin > edge_capacity) {
      for (auto base = begin; base < end;) {
        const auto count = std::min<std::uint64_t>(
          edge_capacity, end - base);
        tiles.push_back({row, 1u, base, base + count, true});
        base += count;
      }
      ++row;
      continue;
    }
    std::uint32_t last = row;
    while (last < rows && last - row < row_capacity &&
           offsets[last + 1] - begin <= edge_capacity) ++last;
    tiles.push_back({row, last - row, begin,
                     offsets[last], false});
    row = last;
  }
  return tiles;
}

#endif
