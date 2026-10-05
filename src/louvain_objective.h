#ifndef FASTEMBEDR_LOUVAIN_OBJECTIVE_H
#define FASTEMBEDR_LOUVAIN_OBJECTIVE_H

#include <algorithm>

inline double louvain_move_score(double edge_weight, double node_degree,
                                 double community_volume,
                                 double graph_volume, double resolution) {
  return edge_weight - resolution * node_degree *
    community_volume / graph_volume;
}

inline double louvain_move_tolerance(double total_edge_weight) {
  return 1e-12 * std::max(1.0, total_edge_weight);
}

#endif
