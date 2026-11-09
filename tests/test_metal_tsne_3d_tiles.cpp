#include "../src/metal_tsne_3d_tiles.h"

#include <cstdint>
#include <stdexcept>

int main() {
  const std::uint64_t base = (std::uint64_t{1} << 31) - 2;
  const std::uint64_t offsets[] = {
    base, base + 5, base + 5, base + 8};
  const auto tiles = metal_tsne_3d_tiles(offsets, 3, 3, 2);
  if (tiles.size() != 3 ||
      tiles[0].first_row != 0 || tiles[0].rows != 1 ||
      tiles[0].first_edge != base ||
      tiles[0].last_edge != base + 3 || !tiles[0].split_row ||
      tiles[1].first_edge != base + 3 ||
      tiles[1].last_edge != base + 5 || !tiles[1].split_row ||
      tiles[2].first_row != 1 || tiles[2].rows != 2 ||
      tiles[2].first_edge != base + 5 ||
      tiles[2].last_edge != base + 8 || tiles[2].split_row) return 1;
  const std::uint64_t bad[] = {0, 3, 2};
  bool rejected = false;
  try {
    metal_tsne_3d_tiles(bad, 2, 3, 2);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  return rejected ? 0 : 2;
}
