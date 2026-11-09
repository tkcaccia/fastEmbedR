#ifndef FASTEMBEDR_CSR_OFFSET_STORAGE_H
#define FASTEMBEDR_CSR_OFFSET_STORAGE_H

#include "csr_offset_plan.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

class FastembedrCsrOffsetStorage {
 public:
  template <typename Count>
  static FastembedrCsrOffsetStorage from_counts(
      const std::vector<Count>& counts,
      std::uint64_t offset_byte_limit =
        std::numeric_limits<std::uint64_t>::max()) {
    static_assert(std::is_integral<Count>::value,
      "CSR row counts must be integers.");
    if (counts.empty() || counts.size() > static_cast<std::size_t>(
        std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("Invalid CSR row count.");
    }
    std::uint64_t total = 0;
    for (const auto count : counts) {
      if constexpr (std::is_signed<Count>::value) {
        if (count < 0) {
          throw std::invalid_argument("Negative CSR row count.");
        }
      }
      const auto value = static_cast<std::uint64_t>(count);
      if (value > std::numeric_limits<std::uint64_t>::max() - total) {
        throw std::overflow_error("CSR edge count exceeds uint64.");
      }
      total += value;
    }
    const auto bytes_per_offset = total <= static_cast<std::uint64_t>(
      std::numeric_limits<std::int32_t>::max()) ? 4u : 8u;
    const auto offset_bytes =
      (static_cast<std::uint64_t>(counts.size()) + 1) *
        bytes_per_offset;
    if (offset_bytes > offset_byte_limit) {
      throw std::length_error("CSR offsets exceed the memory budget.");
    }
    FastembedrCsrOffsetStorage result(counts.size(), total);
    for (const auto count : counts) result.append_row(count);
    return result;
  }

  FastembedrCsrOffsetStorage(std::uint64_t rows,
                            std::uint64_t maximum_edges)
      : rows_(rows), maximum_edges_(maximum_edges) {
    if (rows < 1 || rows > static_cast<std::uint64_t>(
        std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("Invalid CSR row count.");
    }
    kind_ = maximum_edges <= static_cast<std::uint64_t>(
      std::numeric_limits<std::int32_t>::max()) ?
      FastembedrCsrOffsetKind::Int32 :
      FastembedrCsrOffsetKind::Uint64;
    if (kind_ == FastembedrCsrOffsetKind::Int32) {
      narrow_.reserve(static_cast<std::size_t>(rows) + 1);
      narrow_.push_back(0);
    } else {
      wide_.reserve(static_cast<std::size_t>(rows) + 1);
      wide_.push_back(0);
    }
  }

  void append_row(std::uint64_t count) {
    if (completed_ >= rows_ || count > maximum_edges_ - edges_) {
      throw std::overflow_error("CSR row count exceeds its bound.");
    }
    edges_ += count;
    ++completed_;
    if (kind_ == FastembedrCsrOffsetKind::Int32) {
      narrow_.push_back(static_cast<std::int32_t>(edges_));
    } else {
      wide_.push_back(edges_);
    }
  }

  std::uint64_t offset(std::uint64_t row) const {
    if (row > completed_) {
      throw std::out_of_range("CSR offset row is unavailable.");
    }
    return kind_ == FastembedrCsrOffsetKind::Int32 ?
      static_cast<std::uint64_t>(narrow_[row]) : wide_[row];
  }

  void finish() const {
    if (completed_ != rows_) {
      throw std::logic_error("CSR offset rows are incomplete.");
    }
  }

  std::uint64_t edges() const { return edges_; }
  std::uint64_t rows() const { return rows_; }
  std::uint64_t offset_bytes() const {
    return (rows_ + 1) * (kind_ ==
      FastembedrCsrOffsetKind::Int32 ? 4u : 8u);
  }
  FastembedrCsrOffsetKind kind() const { return kind_; }
  const std::vector<std::int32_t>& narrow() const { return narrow_; }
  const std::vector<std::uint64_t>& wide() const { return wide_; }

  std::vector<std::int32_t> take_narrow() && {
    finish();
    if (kind_ != FastembedrCsrOffsetKind::Int32) {
      throw std::logic_error("CSR offsets need the 64-bit path.");
    }
    return std::move(narrow_);
  }

 private:
  std::uint64_t rows_ = 0;
  std::uint64_t maximum_edges_ = 0;
  std::uint64_t completed_ = 0;
  std::uint64_t edges_ = 0;
  FastembedrCsrOffsetKind kind_ = FastembedrCsrOffsetKind::Int32;
  std::vector<std::int32_t> narrow_;
  std::vector<std::uint64_t> wide_;
};

#endif
