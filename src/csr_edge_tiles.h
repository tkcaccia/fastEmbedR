#ifndef FASTEMBEDR_CSR_EDGE_TILES_H
#define FASTEMBEDR_CSR_EDGE_TILES_H

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

template <typename Consume>
void fastembedr_csr_edge_tiles(std::uint64_t begin,
    std::uint64_t end, std::uint32_t capacity, Consume consume) {
  if (end < begin || capacity == 0 ||
      capacity > static_cast<std::uint32_t>(
        std::numeric_limits<std::int32_t>::max())) {
    throw std::invalid_argument("Invalid CSR edge tile bounds.");
  }
  for (std::uint64_t base = begin; base < end;) {
    const auto count = static_cast<std::uint32_t>(
      std::min<std::uint64_t>(capacity, end - base));
    consume(base, count);
    base += count;
  }
}

template <typename Offset>
class FastembedrCsrTileRow {
 public:
  FastembedrCsrTileRow(std::uint64_t count, Offset offset)
      : count_(count), offset_(offset), row_(0),
        row_end_(count == 0 ? 0 : offset(1)) {
    if (count > 0 && row_end_ < offset_(0)) {
      throw std::invalid_argument("CSR row offsets decrease.");
    }
  }

  std::uint64_t at(std::uint64_t edge) {
    while (row_ < count_ && edge >= row_end_) {
      ++row_;
      if (row_ < count_) {
        const auto next = offset_(row_ + 1);
        if (next < row_end_) {
          throw std::invalid_argument("CSR row offsets decrease.");
        }
        row_end_ = next;
      }
    }
    if (row_ == count_) {
      throw std::out_of_range("CSR tile exceeds the row range.");
    }
    return row_;
  }

 private:
  std::uint64_t count_;
  Offset offset_;
  std::uint64_t row_;
  std::uint64_t row_end_;
};

template <typename Offset, typename Consume>
void fastembedr_csr_row_fragments(FastembedrCsrTileRow<Offset>& cursor,
    Offset offset, std::uint64_t base, std::uint32_t count,
    Consume consume) {
  const auto limit = base + count;
  for (auto edge = base; edge < limit;) {
    const auto row = cursor.at(edge);
    const auto row_end = std::min(limit, offset(row + 1));
    if (row_end <= edge) {
      throw std::logic_error("CSR row offsets are not increasing.");
    }
    consume(static_cast<std::uint32_t>(edge - base),
            static_cast<std::uint32_t>(row_end - edge), row);
    edge = row_end;
  }
}

template <typename Offset, typename Consume>
void fastembedr_csr_row_tiles(std::uint64_t rows, Offset offset,
    std::uint64_t begin, std::uint64_t end, std::uint32_t capacity,
    Consume consume) {
  if (rows == 0 || begin > end || begin < offset(0) ||
      end > offset(rows)) {
    throw std::invalid_argument("Invalid CSR row tile range.");
  }
  FastembedrCsrTileRow<Offset> cursor(rows, offset);
  fastembedr_csr_edge_tiles(begin, end, capacity,
    [&](std::uint64_t base, std::uint32_t count) {
      fastembedr_csr_row_fragments(cursor, offset, base, count,
        [&](std::uint32_t local, std::uint32_t items,
            std::uint64_t row) {
          consume(base, local, items, row);
        });
    });
}

#endif
