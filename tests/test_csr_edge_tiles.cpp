#include "../src/csr_edge_tiles.h"
#include "../src/csr_offset_plan.h"
#include "../src/csr_offset_storage.h"

#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

static std::vector<std::pair<std::uint64_t, std::uint64_t>>
tile_rows(std::uint64_t base) {
  const std::vector<std::uint64_t> offsets{
    base, base + 5, base + 5, base + 10};
  auto offset = [&](std::uint64_t row) { return offsets.at(row); };
  FastembedrCsrTileRow<decltype(offset)> cursor(3, offset);
  std::vector<std::pair<std::uint64_t, std::uint64_t>> result;
  fastembedr_csr_edge_tiles(base, base + 10, 4,
    [&](std::uint64_t tile_base, std::uint32_t count) {
      for (std::uint32_t local = 0; local < count; ++local) {
        result.emplace_back(tile_base + local,
                            cursor.at(tile_base + local));
      }
    });
  return result;
}

static std::vector<std::pair<std::uint64_t, std::uint64_t>>
row_tile_fragments(std::uint64_t base) {
  const std::vector<std::uint64_t> offsets{
    base, base + 5, base + 5, base + 10};
  auto offset = [&](std::uint64_t row) { return offsets.at(row); };
  std::vector<std::pair<std::uint64_t, std::uint64_t>> result;
  std::vector<std::tuple<std::uint64_t, std::uint32_t,
                         std::uint32_t, std::uint64_t>> fragments;
  fastembedr_csr_row_tiles(3, offset, base, base + 10, 4,
    [&](std::uint64_t tile_base, std::uint32_t local,
        std::uint32_t count, std::uint64_t row) {
      if (local + count > 4) throw std::logic_error("Tile overflow.");
      fragments.emplace_back(tile_base - base, local, count, row);
      for (std::uint32_t i = 0; i < count; ++i) {
        result.emplace_back(tile_base + local + i, row);
      }
    });
  const decltype(fragments) expected{
    {0, 0, 4, 0}, {4, 0, 1, 0},
    {4, 1, 3, 2}, {8, 0, 2, 2}};
  if (fragments != expected) {
    throw std::logic_error("CSR row crossing changed tile fragments.");
  }
  return result;
}

int main() {
  FastembedrCsrOffsetStorage narrow_offsets(3, 10);
  FastembedrCsrOffsetStorage wide_offsets(3,
    (std::uint64_t{1} << 31) + 10);
  for (const auto count : {5u, 0u, 5u}) narrow_offsets.append_row(count);
  wide_offsets.append_row((std::uint64_t{1} << 31) - 2);
  wide_offsets.append_row(0);
  wide_offsets.append_row(12);
  narrow_offsets.finish();
  wide_offsets.finish();
  if (narrow_offsets.kind() != FastembedrCsrOffsetKind::Int32 ||
      wide_offsets.kind() != FastembedrCsrOffsetKind::Uint64 ||
      narrow_offsets.offset(3) != 10 ||
      wide_offsets.offset(3) != (std::uint64_t{1} << 31) + 10 ||
      wide_offsets.offset(2) != wide_offsets.offset(1)) return 6;
  const auto counted_small = FastembedrCsrOffsetStorage::from_counts(
    std::vector<int>{5, 0, 5});
  const auto counted_wide = FastembedrCsrOffsetStorage::from_counts(
    std::vector<std::uint64_t>{(std::uint64_t{1} << 31) - 2,
                               0, 12});
  if (counted_small.kind() != FastembedrCsrOffsetKind::Int32 ||
      counted_wide.kind() != FastembedrCsrOffsetKind::Uint64 ||
      counted_small.offset(3) != 10 ||
      counted_wide.offset(3) != wide_offsets.offset(3)) return 7;
  bool memory_rejected = false;
  try {
    FastembedrCsrOffsetStorage::from_counts(
      std::vector<std::uint64_t>{(std::uint64_t{1} << 31) + 1},
      15);
  } catch (const std::length_error&) {
    memory_rejected = true;
  }
  if (!memory_rejected || counted_wide.offset_bytes() != 32) return 8;
  const auto small = fastembedr_csr_offset_plan(1000, 30);
  const auto large = fastembedr_csr_offset_plan(50000000, 30);
  if (small.kind != FastembedrCsrOffsetKind::Int32 ||
      small.maximum_edges != 60000 ||
      small.offset_bytes != 4004 ||
      large.kind != FastembedrCsrOffsetKind::Uint64 ||
      large.maximum_edges != 3000000000ULL ||
      large.offset_bytes != 400000008ULL) return 4;
  bool overflow = false;
  try {
    fastembedr_csr_offset_plan(50000000,
      std::numeric_limits<std::uint64_t>::max());
  } catch (const std::overflow_error&) {
    overflow = true;
  }
  if (!overflow) return 5;
  const auto narrow = tile_rows(0);
  const auto wide = tile_rows((std::uint64_t{1} << 31) - 2);
  if (narrow.size() != 10 || wide.size() != narrow.size()) return 1;
  for (std::size_t i = 0; i < narrow.size(); ++i) {
    if (wide[i].first - wide.front().first != narrow[i].first ||
        wide[i].second != narrow[i].second ||
        wide[i].second != (i < 5 ? 0u : 2u)) return 2;
  }
  const auto narrow_fragments = row_tile_fragments(0);
  const auto wide_fragments = row_tile_fragments(
    (std::uint64_t{1} << 31) - 2);
  if (narrow_fragments != narrow || wide_fragments != wide) return 9;
  bool rejected = false;
  try {
    fastembedr_csr_edge_tiles(0, 2, 0,
      [](std::uint64_t, std::uint32_t) {});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) return 3;
  const std::vector<std::uint64_t> bad_offsets{0, 5, 4, 10};
  auto bad = [&](std::uint64_t row) { return bad_offsets.at(row); };
  rejected = false;
  try {
    fastembedr_csr_row_tiles(3, bad, 0, 10, 4,
      [](std::uint64_t, std::uint32_t, std::uint32_t,
         std::uint64_t) {});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  return rejected ? 0 : 10;
}
