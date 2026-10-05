/*
 * SPDX-FileCopyrightText: 2026 Stefano Cacciatore
 * SPDX-License-Identifier: MIT
 */

#ifndef FASTEMBEDR_MASSIVE_GRAPH_ATTRACTION_H
#define FASTEMBEDR_MASSIVE_GRAPH_ATTRACTION_H

#include <string>
#include <vector>

void massive_csr_attraction(const std::string& offsets_path,
                            const std::string& indices_path,
                            const std::string& weights_path,
                            int n, const std::string& access,
                            const std::vector<float>& y, int dims,
                            float exaggeration,
                            std::vector<float>& grad);

#endif
