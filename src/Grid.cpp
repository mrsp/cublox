/**
 * Copyright (C) Stylianos Piperakis, Ownage Dynamics L.P.
 * cublox is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, version 3.
 *
 * cublox is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * cublox. If not, see <https://www.gnu.org/licenses/>.
 **/
#include <cublox/Grid.hpp>
#include <cublox/utils.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace cublox {

namespace {

void appendSlabHashes(std::vector<int> &hash_batch, const Grid::Config &cfg,
                      const std::vector<int> &clear_id, const int axis) {
  const std::array<int, 3> ids{axis, (axis + 1) % 3, (axis + 2) % 3};
  const int h1 = cfg.half_map_size_i(ids[1]);
  const int h2 = cfg.half_map_size_i(ids[2]);
  // max possible number of hashes in a slab
  hash_batch.reserve(hash_batch.size() + static_cast<size_t>(clear_id.size()) *
                                             static_cast<size_t>(2 * h1 + 1) *
                                             static_cast<size_t>(2 * h2 + 1));
  for (const int idd : clear_id) {
    for (int x = -h1; x <= h1; x++) {
      for (int y = -h2; y <= h2; y++) {
        Eigen::Vector3i temp_clear_id;
        temp_clear_id(ids[0]) = idd;
        temp_clear_id(ids[1]) = x;
        temp_clear_id(ids[2]) = y;
        hash_batch.push_back(localIndexToHashId(temp_clear_id, cfg.map_size_i,
                                                cfg.half_map_size_i));
      }
    }
  }
}

} // namespace

Grid::Grid(const Eigen::Vector3i &half_map_size_i, const float resolution,
           const std::optional<float> recenter_threshold,
           const bool origin_at_center) {
  config_.resolution = resolution;
  config_.resolution_inv = 1.0f / resolution;
  config_.half_map_size_i = half_map_size_i;
  config_.map_size_i = 2 * half_map_size_i + Eigen::Vector3i::Ones();
  config_.recenter_threshold = recenter_threshold;
  if (config_.recenter_threshold.has_value()) {
    config_.recenter_threshold_squared =
        config_.recenter_threshold.value() * config_.recenter_threshold.value();
  }
  config_.origin_at_center = origin_at_center;

  // Compute the voxel count in int64 so we can catch overflow before
  // it silently wraps into a negative int32. The kernel/Grid plumbing
  // still treats voxel_num as int (hash indices, kernel grid math, the
  // `op == 0` early-exit, etc.), so anything that doesn't fit in
  // 31 bits is not just a memory problem — it's an API contract
  // violation that needs the dense-grid layout to be replaced with a
  // sparse one. Throw a clear error here if that is the case.
  const std::int64_t nx = config_.map_size_i.x();
  const std::int64_t ny = config_.map_size_i.y();
  const std::int64_t nz = config_.map_size_i.z();
  const std::int64_t total = nx * ny * nz;
  if (total > std::numeric_limits<int>::max()) {
    throw std::runtime_error("cublox::Grid: requested voxel_num = " +
                             std::to_string(total) + " exceeds int32 max (" +
                             std::to_string(std::numeric_limits<int>::max()) +
                             "). Pick a smaller half_map_size_i.");
  }
  config_.voxel_num = static_cast<int>(total);
  config_.origin_at_center = origin_at_center;
  std::cout << "cublox::Grid initialized with " << config_.voxel_num
            << " voxels." << std::endl;
}

bool Grid::inside(const Eigen::Vector3f &pos) const {
  Eigen::Vector3i id_g;
  posToGlobalIndex(pos, config_.resolution_inv, config_.origin_at_center, id_g);
  return inside(id_g);
}

bool Grid::inside(const Eigen::Vector3i &id_g) const {
  const Eigen::Vector3i diff = (id_g - origin_i_).cwiseAbs();
  return (diff.array() <= config_.half_map_size_i.array()).all();
}

void Grid::recenter(const Eigen::Vector3f &pos) {
  if (!config_.recenter_threshold.has_value()) {
    return;
  }

  if ((pos - origin_f_).squaredNorm() < config_.recenter_threshold_squared) {
    return;
  }

  // Compute the shifting index
  Eigen::Vector3i new_origin_i;
  posToGlobalIndex(pos, config_.resolution_inv, config_.origin_at_center,
                   new_origin_i);
  Eigen::Vector3f new_origin_f =
      new_origin_i.cast<float>() * config_.resolution;

  // Compute the delta shift
  Eigen::Vector3i shift_num = new_origin_i - origin_i_;
  if (shift_num.cwiseAbs().maxCoeff() >= config_.map_size_i.maxCoeff()) {
    // Clear the map
    std::cout << "Resetting map" << std::endl;
    reset();
    updateOriginAndBound(new_origin_f, new_origin_i);
    std::cout << "Map is reset" << std::endl;
    return;
  }

  static auto normalize = [](int x, int a, int b) -> int {
    const int range = b - a + 1;
    const int y = (x - a) % range;
    return (y < 0 ? y + range : y) + a;
  };

  std::vector<int> x_slices, y_slices, z_slices;
  for (int i = 0; i < 3; i++) {
    if (shift_num(i) == 0) {
      continue;
    }

    const int min_id_g = -config_.half_map_size_i(i) + origin_i_(i);
    const int min_id_l = min_id_g % config_.map_size_i(i);
    std::vector<int> clear_id;
    if (shift_num(i) > 0) {
      // forward shift, the min id should be cut
      for (int k = 0; k < shift_num(i); k++) {
        int temp_id = min_id_l + k;
        temp_id = normalize(temp_id, -config_.half_map_size_i(i),
                            config_.half_map_size_i(i));
        clear_id.push_back(temp_id);
      }
    } else {
      /// backward shift, the max should be shifted
      for (int k = -1; k >= shift_num(i); k--) {
        int temp_id = min_id_l + k;
        temp_id = normalize(temp_id, -config_.half_map_size_i(i),
                            config_.half_map_size_i(i));
        clear_id.push_back(temp_id);
      }
    }

    if (clear_id.empty()) {
      continue;
    }

    if (i == 0) {
      x_slices = std::move(clear_id);
    } else if (i == 1) {
      y_slices = std::move(clear_id);
    } else {
      z_slices = std::move(clear_id);
    }
  }

  clearRecenterExitSlabs(x_slices, y_slices, z_slices);
  updateOriginAndBound(new_origin_f, new_origin_i);
}

void Grid::clearRecenterExitSlabs(const std::vector<int> &x_slices,
                                  const std::vector<int> &y_slices,
                                  const std::vector<int> &z_slices) {
  std::vector<int> total_hash_ids;
  if (!x_slices.empty()) {
    appendSlabHashes(total_hash_ids, config_, x_slices, 0);
  }
  if (!y_slices.empty()) {
    appendSlabHashes(total_hash_ids, config_, y_slices, 1);
  }
  if (!z_slices.empty()) {
    appendSlabHashes(total_hash_ids, config_, z_slices, 2);
  }
  if (!total_hash_ids.empty()) {
    resetVoxels(total_hash_ids);
  }
}

void Grid::clearVoxelsOutOfGrid(const std::vector<int> &clear_id,
                                const int &i) {
  std::vector<int> total_hash_ids;
  appendSlabHashes(total_hash_ids, config_, clear_id, i);
  resetVoxels(total_hash_ids);
}

void Grid::updateOriginAndBound(const Eigen::Vector3f &new_origin_f,
                                const Eigen::Vector3i &new_origin_i) {
  // update local map origin and local map bound
  origin_i_ = new_origin_i;
  origin_f_ = new_origin_f;

  bound_max_i_ = origin_i_ + config_.half_map_size_i;
  bound_min_i_ = origin_i_ - config_.half_map_size_i;

  globalIndexToPos(bound_min_i_, config_.resolution, config_.origin_at_center,
                   bound_min_f_);
  globalIndexToPos(bound_max_i_, config_.resolution, config_.origin_at_center,
                   bound_max_f_);
}

void Grid::resetVoxels(const std::vector<int> &hash_ids) {
  for (const int h : hash_ids) {
    resetVoxel(h);
  }
}

} // namespace cublox
