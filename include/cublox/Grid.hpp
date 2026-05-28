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

#pragma once

#include <Eigen/Dense>
#include <optional>

namespace cublox {

class Grid {
public:
  Grid(const Eigen::Vector3i &half_map_size_i, const float resolution,
       const std::optional<float> recenter_threshold = std::nullopt,
       const bool origin_at_center = false);
  Grid() = default;
  ~Grid() = default;

  bool inside(const Eigen::Vector3f &pos) const;
  bool inside(const Eigen::Vector3i &id_g) const;
  void updateOriginAndBound(const Eigen::Vector3i &new_origin_i);

  virtual void reset() = 0;
  virtual void resetVoxel(const int hash_id) = 0;

  Eigen::Vector3i getOriginIndex() const { return origin_i_; }
  Eigen::Vector3f getOriginPosition() const { return origin_f_; }

  // Horizontal (xy) margin only — for lidar range/fetch; z half_map is often
  // much smaller.
  float maxHorizontalInWindowRadius(const Eigen::Vector3f &center) const;

  struct Config {
    float resolution{0.0};
    float resolution_inv{0.0};
    Eigen::Vector3i map_size_i{Eigen::Vector3i::Zero()};
    Eigen::Vector3i half_map_size_i{Eigen::Vector3i::Zero()};
    int voxel_num{0};
    bool origin_at_center{false};
    std::optional<float> recenter_threshold{std::nullopt};
    float recenter_threshold_squared{0.0};
  } config_;

  Eigen::Vector3f origin_f_, bound_min_f_, bound_max_f_;
  Eigen::Vector3i origin_i_, bound_min_i_, bound_max_i_;
};

} // namespace cublox