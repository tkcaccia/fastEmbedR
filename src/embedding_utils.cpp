#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

using Rcpp::IntegerMatrix;
using Rcpp::IntegerVector;
using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;

namespace {

double layout_distance_sq(const NumericMatrix& layout, const int i, const int j) {
  if (layout.ncol() == 2) {
    const double dx = layout(i, 0) - layout(j, 0);
    const double dy = layout(i, 1) - layout(j, 1);
    return dx * dx + dy * dy;
  }
  double d2 = 0.0;
  for (int c = 0; c < layout.ncol(); ++c) {
    const double diff = layout(i, c) - layout(j, c);
    d2 += diff * diff;
  }
  return d2;
}

int find_high_rank(const IntegerMatrix& indices,
                   const int query,
                   const int candidate,
                   const int high_rank_limit) {
  for (int r = 0; r < high_rank_limit; ++r) {
    if (indices(query, r) - 1 == candidate) return r + 1;
  }
  return high_rank_limit + 1;
}

template <typename T>
double median_inplace(std::vector<T>& values) {
  if (values.empty()) return R_NaReal;
  const std::size_t mid = values.size() / 2u;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  const double upper = static_cast<double>(values[mid]);
  if (values.size() % 2u == 1u) return upper;
  std::nth_element(values.begin(), values.begin() + mid - 1u, values.begin() + mid);
  return 0.5 * (static_cast<double>(values[mid - 1u]) + upper);
}

} // namespace

// [[Rcpp::export]]
List standardize_cpu_cpp(NumericMatrix data) {
  const int n = data.nrow();
  const int p = data.ncol();
  if (n < 1 || p < 1) Rcpp::stop("data must have at least one row and one column");

  NumericMatrix out(n, p);
  NumericVector center(p);
  NumericVector scale(p);
  const double denom = static_cast<double>(std::max(1, n - 1));

  for (int col = 0; col < p; ++col) {
    double sum = 0.0;
    for (int row = 0; row < n; ++row) {
      const double value = data(row, col);
      if (!std::isfinite(value)) Rcpp::stop("data must contain only finite values");
      sum += value;
    }
    const double mean = sum / static_cast<double>(n);
    center[col] = mean;

    double ss = 0.0;
    for (int row = 0; row < n; ++row) {
      const double centered = data(row, col) - mean;
      out(row, col) = centered;
      ss += centered * centered;
    }
    double sd = std::sqrt(ss / denom);
    if (!std::isfinite(sd) || sd == 0.0) sd = 1.0;
    scale[col] = sd;
    const double inv_sd = 1.0 / sd;
    for (int row = 0; row < n; ++row) {
      out(row, col) *= inv_sd;
    }
  }

  return List::create(
    Rcpp::Named("data") = out,
    Rcpp::Named("center") = center,
    Rcpp::Named("scale") = scale
  );
}

// [[Rcpp::export]]
List strip_self_neighbors_cpp(IntegerMatrix indices, NumericMatrix distances) {
  const int n = indices.nrow();
  const int k = indices.ncol();
  if (distances.nrow() != n || distances.ncol() != k) {
    Rcpp::stop("KNN indices and distances must have the same dimensions");
  }
  if (k < 1) {
    return List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances,
      Rcpp::Named("has_self") = false,
      Rcpp::Named("col_start") = 0,
      Rcpp::Named("n_neighbors") = 0,
      Rcpp::Named("materialized") = false
    );
  }

  int min_idx = std::numeric_limits<int>::max();
  int max_idx = std::numeric_limits<int>::min();
  const double tolerance = std::max(std::sqrt(std::numeric_limits<double>::epsilon()), 1e-12);
  for (int col = 0; col < k; ++col) {
    for (int row = 0; row < n; ++row) {
      const int idx = indices(row, col);
      const double dist = distances(row, col);
      if (idx == NA_INTEGER) Rcpp::stop("KNN indices must not contain NA values");
      if (!std::isfinite(dist) || dist < 0.0) {
        Rcpp::stop("KNN distances must be finite and non-negative");
      }
      min_idx = std::min(min_idx, idx);
      max_idx = std::max(max_idx, idx);
    }
  }
  const bool one_based = min_idx >= 1 && max_idx <= n;
  const int offset = one_based ? 1 : 0;

  bool first_self = true;
  for (int row = 0; row < n; ++row) {
    const int expected = row + offset;
    if (indices(row, 0) != expected || distances(row, 0) > tolerance) {
      first_self = false;
      break;
    }
  }
  if (first_self) {
    return List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances,
      Rcpp::Named("has_self") = true,
      Rcpp::Named("col_start") = 1,
      Rcpp::Named("n_neighbors") = k - 1,
      Rcpp::Named("materialized") = false
    );
  }

  std::vector<int> self_pos(static_cast<std::size_t>(n), -1);
  bool all_rows_have_self = true;
  for (int row = 0; row < n; ++row) {
    const int expected = row + offset;
    for (int col = 0; col < k; ++col) {
      if (indices(row, col) == expected && distances(row, col) <= tolerance) {
        self_pos[static_cast<std::size_t>(row)] = col;
        break;
      }
    }
    if (self_pos[static_cast<std::size_t>(row)] < 0) {
      all_rows_have_self = false;
      break;
    }
  }

  if (!all_rows_have_self) {
    return List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances,
      Rcpp::Named("has_self") = false,
      Rcpp::Named("col_start") = 0,
      Rcpp::Named("n_neighbors") = k,
      Rcpp::Named("materialized") = false
    );
  }

  if (k == 1) {
    IntegerMatrix out_indices(n, 0);
    NumericMatrix out_distances(n, 0);
    return List::create(
      Rcpp::Named("indices") = out_indices,
      Rcpp::Named("distances") = out_distances,
      Rcpp::Named("has_self") = true,
      Rcpp::Named("col_start") = 0,
      Rcpp::Named("n_neighbors") = 0,
      Rcpp::Named("materialized") = true
    );
  }

  IntegerMatrix out_indices(n, k - 1);
  NumericMatrix out_distances(n, k - 1);
  for (int row = 0; row < n; ++row) {
    int out_col = 0;
    const int skip = self_pos[static_cast<std::size_t>(row)];
    for (int col = 0; col < k; ++col) {
      if (col == skip) continue;
      out_indices(row, out_col) = indices(row, col);
      out_distances(row, out_col) = distances(row, col);
      ++out_col;
    }
  }

  return List::create(
    Rcpp::Named("indices") = out_indices,
    Rcpp::Named("distances") = out_distances,
    Rcpp::Named("has_self") = true,
    Rcpp::Named("col_start") = 0,
    Rcpp::Named("n_neighbors") = k - 1,
    Rcpp::Named("materialized") = true
  );
}

// [[Rcpp::export]]
List validate_projection_knn_cpp(IntegerMatrix indices,
                                 NumericMatrix distances,
                                 int n_reference,
                                 int k) {
  const int n = indices.nrow();
  const int width = indices.ncol();
  if (distances.nrow() != n || distances.ncol() != width) {
    Rcpp::stop("KNN indices and distances must have the same dimensions");
  }
  if (n < 1 || width < 1) {
    Rcpp::stop("KNN must have at least one row and one neighbor column");
  }
  if (n_reference < 1) Rcpp::stop("n_reference must be positive");
  if (k < 1 || k > width) Rcpp::stop("k must be in the available neighbor range");

  for (int col = 0; col < width; ++col) {
    for (int row = 0; row < n; ++row) {
      const int idx = indices(row, col);
      const double dist = distances(row, col);
      if (idx == NA_INTEGER || idx < 1 || idx > n_reference) {
        Rcpp::stop("KNN indices must be 1-based row numbers into the reference layout");
      }
      if (!std::isfinite(dist) || dist < 0.0) {
        Rcpp::stop("KNN distances must be finite and non-negative");
      }
    }
  }

  if (k == width) {
    return List::create(
      Rcpp::Named("indices") = indices,
      Rcpp::Named("distances") = distances
    );
  }

  IntegerMatrix out_indices(n, k);
  NumericMatrix out_distances(n, k);
  for (int col = 0; col < k; ++col) {
    for (int row = 0; row < n; ++row) {
      out_indices(row, col) = indices(row, col);
      out_distances(row, col) = distances(row, col);
    }
  }
  return List::create(
    Rcpp::Named("indices") = out_indices,
    Rcpp::Named("distances") = out_distances
  );
}

// [[Rcpp::export]]
double mean_neighbor_rank_error_cpp(IntegerMatrix high_indices,
                                    IntegerMatrix embed_indices,
                                    int k) {
  const int n = high_indices.nrow();
  if (embed_indices.nrow() != n) {
    Rcpp::stop("high_indices and embed_indices must have the same row count");
  }
  k = std::min(k, std::min(high_indices.ncol(), embed_indices.ncol()));
  if (k < 1) return R_NaReal;

  double sum_error = 0.0;
  double count = 0.0;
  std::vector<int> ranks;
  ranks.reserve(static_cast<std::size_t>(k));
  for (int row = 0; row < n; ++row) {
    ranks.clear();
    for (int col = 0; col < k; ++col) ranks.push_back(high_indices(row, col));
    for (int col = 0; col < k; ++col) {
      const int target = embed_indices(row, col);
      int high_rank = k + 1;
      for (int r = 0; r < k; ++r) {
        if (ranks[static_cast<std::size_t>(r)] == target) {
          high_rank = r + 1;
          break;
        }
      }
      sum_error += std::abs(high_rank - (col + 1));
      count += 1.0;
    }
  }
  return count > 0.0 ? sum_error / count : R_NaReal;
}

// [[Rcpp::export]]
NumericVector knn_recall_cpp(IntegerMatrix approx_indices,
                             IntegerMatrix exact_indices,
                             int k) {
  const int n = approx_indices.nrow();
  if (exact_indices.nrow() != n) {
    Rcpp::stop("Approximate and exact KNN must have the same number of rows");
  }
  k = std::min(k, std::min(approx_indices.ncol(), exact_indices.ncol()));
  if (k < 1) Rcpp::stop("k must be positive");

  std::vector<double> recalls(static_cast<std::size_t>(n), 0.0);
  for (int row = 0; row < n; ++row) {
    int shared = 0;
    for (int a = 0; a < k; ++a) {
      const int candidate = approx_indices(row, a);
      for (int e = 0; e < k; ++e) {
        if (candidate == exact_indices(row, e)) {
          ++shared;
          break;
        }
      }
    }
    recalls[static_cast<std::size_t>(row)] =
      static_cast<double>(shared) / static_cast<double>(k);
  }

  double sum = 0.0;
  double min_value = std::numeric_limits<double>::infinity();
  for (const double value : recalls) {
    sum += value;
    min_value = std::min(min_value, value);
  }
  std::vector<double> sorted = recalls;
  const std::size_t mid = sorted.size() / 2u;
  std::nth_element(sorted.begin(), sorted.begin() + mid, sorted.end());
  double median = sorted[mid];
  if (sorted.size() % 2u == 0u) {
    std::nth_element(sorted.begin(), sorted.begin() + mid - 1u, sorted.begin() + mid);
    median = 0.5 * (median + sorted[mid - 1u]);
  }

  NumericVector out = NumericVector::create(
    Rcpp::Named("recall_at_k") = sum / static_cast<double>(n),
    Rcpp::Named("median_recall_at_k") = median,
    Rcpp::Named("min_recall_at_k") = min_value
  );
  return out;
}

// [[Rcpp::export]]
IntegerVector majority_vote_knn_labels_cpp(IntegerMatrix embed_indices,
                                           IntegerVector labels,
                                           int k,
                                           int n_label_levels) {
  const int n = embed_indices.nrow();
  if (labels.size() != n) Rcpp::stop("labels length must match KNN row count");
  k = std::min(k, embed_indices.ncol());
  if (k < 1) Rcpp::stop("k must be positive");

  IntegerVector out(n, NA_INTEGER);
  std::vector<int> counts(static_cast<std::size_t>(std::max(0, n_label_levels)) + 1u, 0);
  for (int row = 0; row < n; ++row) {
    std::fill(counts.begin(), counts.end(), 0);
    for (int col = 0; col < k; ++col) {
      const int idx = embed_indices(row, col) - 1;
      if (idx < 0 || idx >= n) continue;
      const int label = labels[idx];
      if (label != NA_INTEGER && label >= 1 && label <= n_label_levels) {
        ++counts[static_cast<std::size_t>(label)];
      }
    }
    int best_label = NA_INTEGER;
    int best_count = 0;
    for (int label = 1; label <= n_label_levels; ++label) {
      const int count = counts[static_cast<std::size_t>(label)];
      if (count > best_count) {
        best_count = count;
        best_label = label;
      }
    }
    out[row] = best_label;
  }
  return out;
}

// [[Rcpp::export]]
NumericVector batch_entropy_cpp(IntegerMatrix embed_indices,
                                IntegerVector batch,
                                int k,
                                int n_batch_levels) {
  const int n = embed_indices.nrow();
  if (batch.size() != n) Rcpp::stop("batch length must match KNN row count");
  k = std::min(k, embed_indices.ncol());
  if (k < 1 || n_batch_levels < 2) {
    return NumericVector::create(
      Rcpp::Named("batch_entropy") = R_NaReal,
      Rcpp::Named("batch_mixing_score") = R_NaReal
    );
  }

  const double denom = std::log(static_cast<double>(n_batch_levels));
  std::vector<int> counts(static_cast<std::size_t>(n_batch_levels) + 1u, 0);
  double entropy_sum = 0.0;
  int scored = 0;
  for (int row = 0; row < n; ++row) {
    std::fill(counts.begin(), counts.end(), 0);
    int valid = 0;
    for (int col = 0; col < k; ++col) {
      const int idx = embed_indices(row, col) - 1;
      if (idx < 0 || idx >= n) continue;
      const int label = batch[idx];
      if (label != NA_INTEGER && label >= 1 && label <= n_batch_levels) {
        ++counts[static_cast<std::size_t>(label)];
        ++valid;
      }
    }
    if (valid == 0) continue;
    double entropy = 0.0;
    for (int label = 1; label <= n_batch_levels; ++label) {
      const int count = counts[static_cast<std::size_t>(label)];
      if (count == 0) continue;
      const double prob = static_cast<double>(count) / static_cast<double>(valid);
      entropy -= prob * std::log(prob);
    }
    entropy_sum += entropy / denom;
    ++scored;
  }
  const double value = scored > 0 ? entropy_sum / static_cast<double>(scored) : R_NaReal;
  return NumericVector::create(
    Rcpp::Named("batch_entropy") = value,
    Rcpp::Named("batch_mixing_score") = value
  );
}

// [[Rcpp::export]]
NumericVector sampled_pair_distances_cpp(NumericMatrix x,
                                         IntegerVector a,
                                         IntegerVector b,
                                         int n_threads) {
  const int n_pairs = a.size();
  if (b.size() != n_pairs) Rcpp::stop("a and b must have the same length");
  const int n = x.nrow();
  const int p = x.ncol();
  NumericVector out(n_pairs);
  if (n_pairs == 0) return out;
  if (n_threads < 1) n_threads = 1;
  n_threads = std::min(n_threads, n_pairs);

  auto write_range = [&](const int begin, const int end) {
    for (int i = begin; i < end; ++i) {
      const int ai = a[i] - 1;
      const int bi = b[i] - 1;
      if (ai < 0 || ai >= n || bi < 0 || bi >= n) {
        Rcpp::stop("pair indices are out of range");
      }
      double d2 = 0.0;
      for (int col = 0; col < p; ++col) {
        const double diff = x(ai, col) - x(bi, col);
        d2 += diff * diff;
      }
      out[i] = std::sqrt(std::max(0.0, d2));
    }
  };

  if (n_threads == 1) {
    write_range(0, n_pairs);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(n_threads));
    for (int t = 0; t < n_threads; ++t) {
      const int start = (n_pairs * t) / n_threads;
      const int end = (n_pairs * (t + 1)) / n_threads;
      workers.emplace_back(write_range, start, end);
    }
    for (auto& worker : workers) worker.join();
  }
  return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector knn_structure_score_cpp(NumericMatrix layout,
                                            IntegerMatrix indices,
                                            Rcpp::IntegerVector keep,
                                            int preserve_k,
                                            Rcpp::IntegerVector labels,
                                            int n_label_levels) {
  const int n = layout.nrow();
  const bool compact_indices = indices.nrow() == keep.size();
  if (indices.nrow() != n && !compact_indices) {
    Rcpp::stop("indices row count must match layout row count or keep length");
  }
  if (preserve_k < 1 || preserve_k > indices.ncol()) Rcpp::stop("invalid preserve_k");
  if (labels.size() != 0 && labels.size() != n) Rcpp::stop("labels length must match layout row count");

  const int high_rank_limit = indices.ncol();
  struct ScoreAccum {
    double preservation_sum = 0.0;
    double trust_sum = 0.0;
    double continuity_sum = 0.0;
    double label_accuracy_sum = 0.0;
    int label_accuracy_n = 0;
    int scored = 0;
  };

  const int keep_n = keep.size();
  int score_threads = 1;
  if (keep_n >= 128) {
    const unsigned int hw = std::thread::hardware_concurrency();
    if (hw > 1u) score_threads = std::min<int>(keep_n, std::min<unsigned int>(hw, 4u));
  }
  std::vector<ScoreAccum> accumulators(static_cast<std::size_t>(score_threads));

  auto score_range = [&](const int thread_id, const int begin, const int end) {
    ScoreAccum local;
    std::vector<int> label_counts(static_cast<std::size_t>(std::max(0, n_label_levels)) + 1u);
    std::vector<std::pair<double, int>> low_order;
    std::vector<std::pair<double, int>> continuity_targets;
    std::vector<int> continuity_diff;
    low_order.reserve(static_cast<std::size_t>(std::max(0, preserve_k)));
    continuity_targets.reserve(static_cast<std::size_t>(std::max(0, preserve_k)));
    continuity_diff.reserve(static_cast<std::size_t>(std::max(0, preserve_k)) + 1u);

    for (int kk = begin; kk < end; ++kk) {
      const int query = keep[kk] - 1;
      if (query < 0 || query >= n) continue;
      const int index_row = compact_indices ? kk : query;

      low_order.clear();
      continuity_targets.clear();
      for (int r = 0; r < preserve_k; ++r) {
        const int high_nb = indices(index_row, r) - 1;
        if (high_nb < 0 || high_nb >= n) continue;
        continuity_targets.emplace_back(layout_distance_sq(layout, query, high_nb), high_nb);
      }
      std::sort(continuity_targets.begin(), continuity_targets.end());
      continuity_diff.assign(continuity_targets.size() + 1u, 0);
      for (int candidate = 0; candidate < n; ++candidate) {
        if (candidate == query) continue;
        const std::pair<double, int> candidate_rank_key(
          layout_distance_sq(layout, query, candidate),
          candidate
        );
        if (static_cast<int>(low_order.size()) < preserve_k) {
          low_order.push_back(candidate_rank_key);
          std::push_heap(low_order.begin(), low_order.end());
        } else if (candidate_rank_key < low_order.front()) {
          std::pop_heap(low_order.begin(), low_order.end());
          low_order.back() = candidate_rank_key;
          std::push_heap(low_order.begin(), low_order.end());
        }
        const auto first_greater = std::upper_bound(
          continuity_targets.begin(),
          continuity_targets.end(),
          candidate_rank_key
        );
        const std::size_t pos = static_cast<std::size_t>(
          first_greater - continuity_targets.begin()
        );
        if (pos < continuity_targets.size()) {
          ++continuity_diff[pos];
          --continuity_diff[continuity_targets.size()];
        }
      }
      if (static_cast<int>(low_order.size()) < preserve_k) continue;
      std::sort(low_order.begin(), low_order.end());

      int shared = 0;
      double trust_penalty = 0.0;
      for (int r = 0; r < preserve_k; ++r) {
        const int low_nb = low_order[static_cast<std::size_t>(r)].second;
        const int high_rank = find_high_rank(indices, index_row, low_nb, high_rank_limit);
        if (high_rank <= preserve_k) ++shared;
        trust_penalty += std::max(0, high_rank - preserve_k);
      }
      const double trust_denom = static_cast<double>(preserve_k) *
        static_cast<double>(std::max(1, high_rank_limit + 1 - preserve_k));

      double cont_penalty = 0.0;
      int lower_rank_count = 0;
      for (std::size_t t = 0; t < continuity_targets.size(); ++t) {
        lower_rank_count += continuity_diff[t];
        const int low_rank = 1 + lower_rank_count;
        cont_penalty += std::max(0, low_rank - preserve_k);
      }
      const double cont_denom = static_cast<double>(preserve_k) *
        static_cast<double>(std::max(1, n - preserve_k));

      local.preservation_sum += static_cast<double>(shared) / static_cast<double>(preserve_k);
      local.trust_sum += std::max(0.0, std::min(1.0, 1.0 - trust_penalty / trust_denom));
      local.continuity_sum += std::max(0.0, std::min(1.0, 1.0 - cont_penalty / cont_denom));

      if (labels.size() == n && n_label_levels > 0 && labels[query] != NA_INTEGER) {
        std::fill(label_counts.begin(), label_counts.end(), 0);
        for (int r = 0; r < preserve_k; ++r) {
          const int label = labels[low_order[static_cast<std::size_t>(r)].second];
          if (label != NA_INTEGER && label >= 1 && label <= n_label_levels) {
            ++label_counts[static_cast<std::size_t>(label)];
          }
        }
        int best_label = 0;
        int best_count = 0;
        for (int label = 1; label <= n_label_levels; ++label) {
          if (label_counts[static_cast<std::size_t>(label)] > best_count) {
            best_count = label_counts[static_cast<std::size_t>(label)];
            best_label = label;
          }
        }
        if (best_count > 0) {
          local.label_accuracy_sum += best_label == labels[query] ? 1.0 : 0.0;
          ++local.label_accuracy_n;
        }
      }

      ++local.scored;
    }
    accumulators[static_cast<std::size_t>(thread_id)] = local;
  };

  if (score_threads == 1) {
    score_range(0, 0, keep_n);
  } else {
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(score_threads - 1));
    const int chunk = (keep_n + score_threads - 1) / score_threads;
    for (int t = 1; t < score_threads; ++t) {
      const int begin = t * chunk;
      const int end = std::min(keep_n, begin + chunk);
      workers.emplace_back(score_range, t, begin, end);
    }
    score_range(0, 0, std::min(keep_n, chunk));
    for (auto& worker : workers) worker.join();
  }

  double preservation_sum = 0.0;
  double trust_sum = 0.0;
  double continuity_sum = 0.0;
  double label_accuracy_sum = 0.0;
  int label_accuracy_n = 0;
  int scored = 0;
  for (const ScoreAccum& acc : accumulators) {
    preservation_sum += acc.preservation_sum;
    trust_sum += acc.trust_sum;
    continuity_sum += acc.continuity_sum;
    label_accuracy_sum += acc.label_accuracy_sum;
    label_accuracy_n += acc.label_accuracy_n;
    scored += acc.scored;
  }

  if (scored == 0) {
    return Rcpp::NumericVector::create(
      Rcpp::Named("knn_preservation") = NA_REAL,
      Rcpp::Named("local_trustworthiness") = NA_REAL,
      Rcpp::Named("local_continuity") = NA_REAL,
      Rcpp::Named("structure_score") = NA_REAL,
      Rcpp::Named("embedding_knn_accuracy") = NA_REAL
    );
  }

  const double preservation = preservation_sum / scored;
  const double trustworthiness = trust_sum / scored;
  const double continuity = continuity_sum / scored;
  const double structure = (preservation + trustworthiness + continuity) / 3.0;
  const double label_accuracy = label_accuracy_n > 0 ?
    label_accuracy_sum / label_accuracy_n :
    R_NaN;

  return Rcpp::NumericVector::create(
    Rcpp::Named("knn_preservation") = preservation,
    Rcpp::Named("local_trustworthiness") = trustworthiness,
    Rcpp::Named("local_continuity") = continuity,
    Rcpp::Named("structure_score") = structure,
    Rcpp::Named("embedding_knn_accuracy") = label_accuracy
  );
}

// [[Rcpp::export]]
double silhouette_score_cpp(NumericMatrix layout, Rcpp::IntegerVector labels) {
  const int n = layout.nrow();
  const int n_components = layout.ncol();
  if (labels.size() != n) Rcpp::stop("labels length must match layout row count");

  int max_label = 0;
  std::vector<int> valid;
  valid.reserve(n);
  for (int i = 0; i < n; ++i) {
    const int label = labels[i];
    if (label == NA_INTEGER || label < 1) continue;
    bool finite = true;
    for (int c = 0; c < n_components; ++c) {
      if (!std::isfinite(layout(i, c))) {
        finite = false;
        break;
      }
    }
    if (!finite) continue;
    max_label = std::max(max_label, label);
    valid.push_back(i);
  }

  const int n_valid = static_cast<int>(valid.size());
  if (n_valid < 2 || max_label < 2) return NA_REAL;

  std::vector<int> counts(static_cast<std::size_t>(max_label) + 1u, 0);
  for (const int i : valid) ++counts[static_cast<std::size_t>(labels[i])];

  int n_nonempty_classes = 0;
  for (int label = 1; label <= max_label; ++label) {
    if (counts[static_cast<std::size_t>(label)] > 0) ++n_nonempty_classes;
  }
  if (n_nonempty_classes < 2) return NA_REAL;

  std::vector<double> class_sums(static_cast<std::size_t>(max_label) + 1u, 0.0);
  double total = 0.0;
  int scored = 0;
  for (const int i : valid) {
    std::fill(class_sums.begin(), class_sums.end(), 0.0);
    const int own_label = labels[i];

    if (n_components == 2) {
      const double xi0 = layout(i, 0);
      const double xi1 = layout(i, 1);
      for (const int j : valid) {
        if (j == i) continue;
        const double dx = xi0 - layout(j, 0);
        const double dy = xi1 - layout(j, 1);
        class_sums[static_cast<std::size_t>(labels[j])] += std::sqrt(dx * dx + dy * dy);
      }
    } else {
      for (const int j : valid) {
        if (j == i) continue;
        double dist_sq = 0.0;
        for (int c = 0; c < n_components; ++c) {
          const double diff = layout(i, c) - layout(j, c);
          dist_sq += diff * diff;
        }
        class_sums[static_cast<std::size_t>(labels[j])] += std::sqrt(std::max(0.0, dist_sq));
      }
    }

    const int own_count = counts[static_cast<std::size_t>(own_label)] - 1;
    const double a = own_count > 0 ?
      class_sums[static_cast<std::size_t>(own_label)] / static_cast<double>(own_count) :
      0.0;

    double b = std::numeric_limits<double>::infinity();
    for (int label = 1; label <= max_label; ++label) {
      if (label == own_label || counts[static_cast<std::size_t>(label)] == 0) continue;
      b = std::min(
        b,
        class_sums[static_cast<std::size_t>(label)] /
          static_cast<double>(counts[static_cast<std::size_t>(label)])
      );
    }

    if (!std::isfinite(b)) {
      total += 0.0;
    } else {
      const double denom = std::max(a, b);
      total += denom > 0.0 ? (b - a) / denom : 0.0;
    }
    ++scored;
  }

  return scored > 0 ? total / static_cast<double>(scored) : NA_REAL;
}

// [[Rcpp::export]]
NumericMatrix interpolate_landmark_layout_cpp(NumericMatrix landmark_layout,
                                              Rcpp::IntegerVector landmark_indices,
                                              IntegerMatrix projection_indices,
                                              NumericMatrix projection_distances,
                                              int n) {
  const int n_landmarks = landmark_layout.nrow();
  const int n_components = landmark_layout.ncol();
  const int projection_n = projection_indices.nrow();
  const int projection_k = projection_indices.ncol();

  if (n < 1) Rcpp::stop("n must be positive");
  if (projection_n != n) Rcpp::stop("projection_indices row count must equal n");
  if (projection_distances.nrow() != projection_n ||
      projection_distances.ncol() != projection_k) {
    Rcpp::stop("projection_indices and projection_distances must have the same dimensions");
  }
  if (landmark_indices.size() != n_landmarks) {
    Rcpp::stop("landmark_indices length must match landmark_layout rows");
  }

  NumericMatrix layout(n, n_components);
  const double eps = std::sqrt(std::numeric_limits<double>::epsilon());
  std::vector<float> adjusted;
  std::vector<float> positive;
  adjusted.reserve(static_cast<std::size_t>(projection_k));
  positive.reserve(static_cast<std::size_t>(projection_k));

  for (int i = 0; i < n; ++i) {
    int zero_col = -1;
    double rho = std::numeric_limits<double>::infinity();
    for (int j = 0; j < projection_k; ++j) {
      const int idx = projection_indices(i, j);
      if (idx < 1 || idx > n_landmarks) Rcpp::stop("projection indices out of range");
      const double d = std::max(0.0, projection_distances(i, j));
      if (d <= eps && zero_col < 0) zero_col = j;
      if (d < rho) rho = d;
    }

    if (zero_col >= 0) {
      const int landmark = projection_indices(i, zero_col) - 1;
      for (int c = 0; c < n_components; ++c) layout(i, c) = landmark_layout(landmark, c);
      continue;
    }

    adjusted.clear();
    positive.clear();
    for (int j = 0; j < projection_k; ++j) {
      const double d = std::max(0.0, projection_distances(i, j));
      const double value = std::max(0.0, d - rho);
      adjusted.push_back(static_cast<float>(value));
      if (value > eps) positive.push_back(static_cast<float>(value));
    }

    double sigma = R_NaReal;
    if (positive.empty()) {
      std::vector<float> distances;
      distances.reserve(static_cast<std::size_t>(projection_k));
      for (int j = 0; j < projection_k; ++j) {
        distances.push_back(static_cast<float>(std::max(0.0, projection_distances(i, j))));
      }
      sigma = median_inplace(distances);
    } else {
      sigma = median_inplace(positive);
    }
    if (!std::isfinite(sigma) || sigma < eps) sigma = eps;

    double weight_sum = 0.0;
    for (int j = 0; j < projection_k; ++j) {
      const double w = std::exp(-adjusted[static_cast<std::size_t>(j)] / sigma);
      adjusted[static_cast<std::size_t>(j)] = static_cast<float>(w);
      weight_sum += w;
    }
    if (!std::isfinite(weight_sum) || weight_sum <= 0.0) {
      weight_sum = static_cast<double>(projection_k);
      std::fill(adjusted.begin(), adjusted.end(), 1.0f);
    }

    for (int c = 0; c < n_components; ++c) {
      double value = 0.0;
      for (int j = 0; j < projection_k; ++j) {
        const int landmark = projection_indices(i, j) - 1;
        value += static_cast<double>(adjusted[static_cast<std::size_t>(j)]) *
          landmark_layout(landmark, c);
      }
      layout(i, c) = value / weight_sum;
    }
  }

  for (int i = 0; i < n_landmarks; ++i) {
    const int row = landmark_indices[i] - 1;
    if (row < 0 || row >= n) Rcpp::stop("landmark indices out of range");
    for (int c = 0; c < n_components; ++c) layout(row, c) = landmark_layout(i, c);
  }
  return layout;
}

// [[Rcpp::export]]
List select_low_confidence_rows_cpp(NumericVector confidence,
                                    IntegerVector landmark_indices,
                                    double fraction) {
  const int n = confidence.size();
  if (n < 1) Rcpp::stop("confidence must be non-empty");
  if (!std::isfinite(fraction) || fraction <= 0.0) {
    Rcpp::stop("fraction must be a positive finite value");
  }
  fraction = std::min(1.0, fraction);

  std::vector<unsigned char> is_landmark(static_cast<std::size_t>(n), 0u);
  for (int i = 0; i < landmark_indices.size(); ++i) {
    const int row = landmark_indices[i] - 1;
    if (row >= 0 && row < n) {
      is_landmark[static_cast<std::size_t>(row)] = 1u;
    }
  }

  std::vector<std::pair<double, int>> eligible;
  eligible.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const double score = confidence[i];
    if (!std::isfinite(score)) Rcpp::stop("confidence scores must be finite");
    if (is_landmark[static_cast<std::size_t>(i)] == 0u) {
      eligible.emplace_back(score, i);
    }
  }

  if (eligible.empty()) {
    return List::create(
      Rcpp::Named("rows") = IntegerVector(0),
      Rcpp::Named("policy") = "low_confidence",
      Rcpp::Named("selected") = 0,
      Rcpp::Named("selected_fraction") = 0.0,
      Rcpp::Named("confidence_threshold") = R_NaReal,
      Rcpp::Named("selection_backend") = "cpp_confidence_mask"
    );
  }

  int count = static_cast<int>(std::ceil(static_cast<double>(eligible.size()) * fraction));
  count = std::max(1, std::min(static_cast<int>(eligible.size()), count));
  if (count >= static_cast<int>(eligible.size())) {
    return List::create(
      Rcpp::Named("rows") = R_NilValue,
      Rcpp::Named("policy") = "all",
      Rcpp::Named("selected") = static_cast<int>(eligible.size()),
      Rcpp::Named("selected_fraction") =
        static_cast<double>(eligible.size()) / static_cast<double>(n),
      Rcpp::Named("confidence_threshold") = R_NaReal,
      Rcpp::Named("selection_backend") = "cpp_confidence_mask"
    );
  }

  auto row_less = [](const std::pair<double, int>& a,
                     const std::pair<double, int>& b) {
    if (a.first == b.first) return a.second < b.second;
    return a.first < b.first;
  };
  std::nth_element(eligible.begin(), eligible.begin() + count, eligible.end(), row_less);
  eligible.resize(static_cast<std::size_t>(count));
  std::sort(eligible.begin(), eligible.end(), row_less);

  IntegerVector rows(count);
  double threshold = eligible.front().first;
  for (int i = 0; i < count; ++i) {
    rows[i] = eligible[static_cast<std::size_t>(i)].second + 1;
    threshold = std::max(threshold, eligible[static_cast<std::size_t>(i)].first);
  }
  std::sort(rows.begin(), rows.end());

  return List::create(
    Rcpp::Named("rows") = rows,
    Rcpp::Named("policy") = "low_confidence",
    Rcpp::Named("selected") = count,
    Rcpp::Named("selected_fraction") = static_cast<double>(count) / static_cast<double>(n),
    Rcpp::Named("confidence_threshold") = threshold,
    Rcpp::Named("selection_backend") = "cpp_confidence_mask"
  );
}

// [[Rcpp::export]]
NumericMatrix project_embedding_knn_cpp(NumericMatrix reference_layout,
                                        IntegerMatrix projection_indices,
                                        NumericMatrix projection_distances) {
  const int n_reference = reference_layout.nrow();
  const int n_components = reference_layout.ncol();
  const int n_query = projection_indices.nrow();
  const int projection_k = projection_indices.ncol();

  if (n_reference < 1) Rcpp::stop("reference_layout must have at least one row");
  if (n_components < 1) Rcpp::stop("reference_layout must have at least one column");
  if (n_query < 1) Rcpp::stop("projection_indices must have at least one row");
  if (projection_k < 1) Rcpp::stop("projection_indices must have at least one column");
  if (projection_distances.nrow() != n_query ||
      projection_distances.ncol() != projection_k) {
    Rcpp::stop("projection_indices and projection_distances must have the same dimensions");
  }

  NumericMatrix layout(n_query, n_components);
  const double eps = std::sqrt(std::numeric_limits<double>::epsilon());
  std::vector<float> adjusted;
  std::vector<float> positive;
  std::vector<float> row_distances;
  adjusted.reserve(static_cast<std::size_t>(projection_k));
  positive.reserve(static_cast<std::size_t>(projection_k));
  row_distances.reserve(static_cast<std::size_t>(projection_k));

  for (int i = 0; i < n_query; ++i) {
    int zero_count = 0;
    double rho = std::numeric_limits<double>::infinity();
    row_distances.clear();
    for (int j = 0; j < projection_k; ++j) {
      const int idx = projection_indices(i, j);
      if (idx < 1 || idx > n_reference) Rcpp::stop("projection indices out of range");
      const double d = projection_distances(i, j);
      if (!std::isfinite(d) || d < 0.0) {
        Rcpp::stop("projection distances must be finite and non-negative");
      }
      row_distances.push_back(static_cast<float>(d));
      if (d <= eps) ++zero_count;
      if (d < rho) rho = d;
    }

    if (zero_count > 0) {
      const double inv_zero_count = 1.0 / static_cast<double>(zero_count);
      for (int c = 0; c < n_components; ++c) {
        double value = 0.0;
        for (int j = 0; j < projection_k; ++j) {
          if (static_cast<double>(row_distances[static_cast<std::size_t>(j)]) <= eps) {
            const int reference_row = projection_indices(i, j) - 1;
            value += inv_zero_count * reference_layout(reference_row, c);
          }
        }
        layout(i, c) = value;
      }
      continue;
    }

    adjusted.clear();
    positive.clear();
    for (int j = 0; j < projection_k; ++j) {
      const double value = std::max(
        0.0,
        static_cast<double>(row_distances[static_cast<std::size_t>(j)]) - rho
      );
      adjusted.push_back(static_cast<float>(value));
      if (value > eps) positive.push_back(static_cast<float>(value));
    }

    double sigma = positive.empty() ? median_inplace(row_distances) : median_inplace(positive);
    if (!std::isfinite(sigma) || sigma < eps) sigma = eps;

    double weight_sum = 0.0;
    for (int j = 0; j < projection_k; ++j) {
      const double w = std::exp(-adjusted[static_cast<std::size_t>(j)] / sigma);
      adjusted[static_cast<std::size_t>(j)] = static_cast<float>(w);
      weight_sum += w;
    }
    if (!std::isfinite(weight_sum) || weight_sum <= 0.0) {
      weight_sum = static_cast<double>(projection_k);
      std::fill(adjusted.begin(), adjusted.end(), 1.0f);
    }

    for (int c = 0; c < n_components; ++c) {
      double value = 0.0;
      for (int j = 0; j < projection_k; ++j) {
        const int reference_row = projection_indices(i, j) - 1;
        value += static_cast<double>(adjusted[static_cast<std::size_t>(j)]) *
          reference_layout(reference_row, c);
      }
      layout(i, c) = value / weight_sum;
    }
  }
  return layout;
}
