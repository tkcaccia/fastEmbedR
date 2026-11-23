#include "../src/metal_tsne_state_budget.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

int main() {
  const auto streamed = metal_tsne_state_budget(
    100000000, 2, 3000000000ULL, 65536, true);
  const auto mapped = metal_tsne_state_budget(
    100000000, 2, 3000000000ULL, 65536, false);
  if (streamed.layout_state_bytes != 5600000000ULL ||
      streamed.graph_upload_bytes != 2621448ULL ||
      mapped.graph_upload_bytes != 24800000008ULL ||
      streamed.total_bytes >= mapped.total_bytes) return 1;
  if (!metal_tsne_state_fits(streamed, 0, 16000000000ULL) ||
      metal_tsne_state_fits(streamed, 0, 8000000000ULL) ||
      metal_tsne_state_fits(streamed, 0, 0)) return 2;
  constexpr std::uint64_t gib = std::uint64_t{1} << 30;
  constexpr auto wide_edges = std::uint64_t{5200000000};
  constexpr auto page = std::uint64_t{16384};
  if (!metal_tsne_mmap_fits(50000000, 2, wide_edges / 2, 200 * gib,
        4 * gib, 224 * gib, 64 * gib, page) ||
      !metal_tsne_mmap_fits(100000000, 2, wide_edges, 200 * gib,
        4 * gib, 224 * gib, 64 * gib, page) ||
      metal_tsne_mmap_fits(100000000, 2, wide_edges, 22 * gib,
        4 * gib, 224 * gib, 64 * gib, page) ||
      metal_tsne_mmap_fits(100000000, 2, wide_edges, 200 * gib,
        4 * gib, 64 * gib, 64 * gib, page) ||
      metal_tsne_mmap_fits(100000000, 2, wide_edges, 200 * gib,
        4 * gib, 224 * gib, 16 * gib, page) ||
      metal_tsne_mmap_fits(16, 2, 32, 2 * gib, 0, 8 * gib, page - 1, page)) {
    return 4;
  }
  bool overflow = false;
  try {
    metal_tsne_state_budget(
      std::numeric_limits<std::uint64_t>::max(), 3,
      1, 1, false);
  } catch (const std::overflow_error&) {
    overflow = true;
  }
  return overflow ? 0 : 3;
}
