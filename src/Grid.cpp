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

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace cublox {

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

float Grid::maxInWindowRadius(const Eigen::Vector3f &center) const {
  Eigen::Vector3i center_g;
  posToGlobalIndex(center, config_.resolution_inv, config_.origin_at_center,
                   center_g);
  const Eigen::Vector3i margin_i =
      config_.half_map_size_i - (center_g - origin_i_).cwiseAbs();
  if ((margin_i.array() < 0).any()) {
    return 0.0f;
  }
  return static_cast<float>(margin_i.minCoeff()) * config_.resolution;
}

float Grid::maxHorizontalInWindowRadius(const Eigen::Vector3f &center) const {
  Eigen::Vector3i center_g;
  posToGlobalIndex(center, config_.resolution_inv, config_.origin_at_center,
                   center_g);
  const Eigen::Vector3i diff = (center_g - origin_i_).cwiseAbs();
  const int margin_x = config_.half_map_size_i.x() - diff.x();
  const int margin_y = config_.half_map_size_i.y() - diff.y();
  if (margin_x < 0 || margin_y < 0) {
    return 0.0f;
  }
  return static_cast<float>(std::min(margin_x, margin_y)) * config_.resolution;
}

void Grid::updateOriginAndBound(const Eigen::Vector3i &new_origin_i) {
  origin_i_ = new_origin_i;
  globalIndexToPos(origin_i_, config_.resolution, config_.origin_at_center,
                   origin_f_);

  bound_max_i_ = origin_i_ + config_.half_map_size_i;
  bound_min_i_ = origin_i_ - config_.half_map_size_i;

  globalIndexToPos(bound_min_i_, config_.resolution, config_.origin_at_center,
                   bound_min_f_);
  globalIndexToPos(bound_max_i_, config_.resolution, config_.origin_at_center,
                   bound_max_f_);
}

} // namespace cublox
