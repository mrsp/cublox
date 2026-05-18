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
#include <vector>

namespace cublox {

class Grid {
public:
  Grid(const Eigen::Vector3i &half_map_size_i, const float resolution,
       const std::optional<float> recenter_threshold = std::nullopt,
       const bool origin_at_center = true);
  Grid() = default;
  ~Grid() = default;

  bool inside(const Eigen::Vector3f &pos) const;
  bool inside(const Eigen::Vector3i &id_g) const;
  void clearVoxelsOutOfGrid(const std::vector<int> &clear_id, const int &i);
  void updateOriginAndBound(const Eigen::Vector3f &new_origin_f,
                            const Eigen::Vector3i &new_origin_i);
  void recenter(const Eigen::Vector3f &pos);

  virtual void reset() = 0;
  virtual void resetVoxel(const int hash_id) = 0;
  virtual void resetVoxels(const std::vector<int> &hash_ids);
  // After a sliding-window recenter: zero voxels on exiting slabs. Default
  // expands slabs to hash IDs on the CPU; OccupancyGrid overrides with a GPU
  // mask sweep (check clearRecenterExitSlabs there).
  virtual void clearRecenterExitSlabs(const std::vector<int> &x_slices,
                                      const std::vector<int> &y_slices,
                                      const std::vector<int> &z_slices);

  Eigen::Vector3i getOriginIndex() const { return origin_i_; }
  Eigen::Vector3f getOriginPosition() const { return origin_f_; }

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