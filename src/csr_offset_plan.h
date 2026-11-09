#ifndef FASTEMBEDR_CSR_OFFSET_PLAN_H
#define FASTEMBEDR_CSR_OFFSET_PLAN_H

#include <cstdint>
#include <limits>
#include <stdexcept>

enum class FastembedrCsrOffsetKind { Int32, Uint64 };

struct FastembedrCsrOffsetPlan {
  std::uint64_t maximum_edges;
  std::uint64_t offset_bytes;
  FastembedrCsrOffsetKind kind;
};

inline FastembedrCsrOffsetPlan fastembedr_csr_offset_plan(
    std::uint64_t rows, std::uint64_t width,
    std::uint64_t symmetry_factor = 2) {
  if (rows < 2 || width < 1 || symmetry_factor < 1 ||
      rows > static_cast<std::uint64_t>(
        std::numeric_limits<std::int32_t>::max())) {
    throw std::invalid_argument("Invalid CSR dimensions.");
  }
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  if (width > maximum / rows ||
      symmetry_factor > maximum / (rows * width)) {
    throw std::overflow_error("CSR edge bound exceeds uint64.");
  }
  const auto edge_bound = rows * width * symmetry_factor;
  const auto kind = edge_bound <= static_cast<std::uint64_t>(
    std::numeric_limits<std::int32_t>::max()) ?
      FastembedrCsrOffsetKind::Int32 :
      FastembedrCsrOffsetKind::Uint64;
  const auto bytes_per_offset = kind ==
    FastembedrCsrOffsetKind::Int32 ? 4u : 8u;
  return {edge_bound, (rows + 1) * bytes_per_offset, kind};
}

#endif
