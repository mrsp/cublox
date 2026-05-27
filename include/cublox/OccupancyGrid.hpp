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
#include <cuda_runtime.h>
#include <optional>
#include <vector>

#include <cublox/Grid.hpp>
#include <cublox/kernels.hpp>

namespace cublox {

using PointCloud = Eigen::Matrix<float, Eigen::Dynamic, 3>;

class OccupancyGrid : public Grid {
public:
  enum class VoxelState {
    UNKNOWN = 0,
    FREE = 1,
    OCCUPIED = 2,
  };

  struct OccupancySample {
    Eigen::Vector3f position;
    VoxelState state;
  };

  OccupancyGrid(const Eigen::Vector3i &half_map_size_i, const float resolution,
                const bool origin_at_center,
                const std::optional<double> recenter_threshold,
                const Eigen::Vector3f &origin = Eigen::Vector3f::Zero());
  OccupancyGrid() = default;
  ~OccupancyGrid();

  void reset() override;
  void resetVoxel(const int hash_id) override;

  // Run one raycast pass for `input_cloud` originating at `sensor_origin`.
  void update(const PointCloud &input_cloud,
              const Eigen::Vector3f &sensor_origin);

  // Sliding-window recenter entirely on GPU (slab plan + clear).
  void recenter(const Eigen::Vector3f &pos);

  // Pull voxels within `radius` [m] of `center` (sphere) from GPU memory only.
  // `state_filter`: std::nullopt returns all states; otherwise only matching
  // voxels. `max_results`: cap output count (0 = no cap beyond fetch buffer).
  std::vector<OccupancySample> fetchOccupancyAround(
      const Eigen::Vector3f &center, const float radius,
      const std::optional<VoxelState> state_filter = std::nullopt,
      const int max_results = 0) const;

  // Default: 25m. Call this once during setup.
  inline void setMaxRaycastRange(const float range) {
    max_raycast_range_ = range;
  }
  inline float getMaxRaycastRange() const { return max_raycast_range_; }

  // Defaults values are (p_hit=0.70, p_miss=0.40, p_min=0.12, p_max=0.97). Call
  // this once during setup
  void setLogOddsParams(const float l_hit, const float l_miss,
                        const float l_min, const float l_max,
                        const float l_free, const float l_occupied) {
    l_hit_ = l_hit;
    l_miss_ = l_miss;
    l_min_ = l_min;
    l_max_ = l_max;
    l_free_ = l_free;
    l_occupied_ = l_occupied;
    // stored == 0 decodes to logit(0.5): true 50/50 unknown prior.
    l_unknown_ = 0.0f;
  }

  inline float getResolutionInv() const { return config_.resolution_inv; }
  inline bool getOriginAtCenter() const { return config_.origin_at_center; }

  bool isOccupied(const Eigen::Vector3f &pos) const;
  bool isUnknown(const Eigen::Vector3f &pos) const;
  bool isFree(const Eigen::Vector3f &pos) const;
  bool isOccupied(const Eigen::Vector3i &id_g) const;
  bool isUnknown(const Eigen::Vector3i &id_g) const;
  bool isFree(const Eigen::Vector3i &id_g) const;
  bool isOccupied(const int hash_id) const;
  bool isUnknown(const int hash_id) const;
  bool isFree(const int hash_id) const;

private:
  bool first_run_{true};

  // Buffer holds logits relative to l_unknown_ (logit(0.5) == 0). Unknown prior
  // is stored == 0 (cudaMemset-friendly); decode with + l_unknown_.
  float storedToLogit(const float s) const noexcept { return s + l_unknown_; }
  bool occOccupied(const float s) const noexcept {
    return storedToLogit(s) >= l_occupied_;
  }
  bool occFree(const float s) const noexcept {
    return storedToLogit(s) < l_free_;
  }
  bool occUnknown(const float s) const noexcept {
    const float l = storedToLogit(s);
    return l >= l_free_ && l < l_occupied_;
  }

  float storedAt(const int hash_id) const;

  // Grow d_cloud_{x,y,z}_ to hold at least `n` points. Cheap no-op when
  // capacity already suffices.
  void ensureCloudCapacity(const int n);

  // Device voxel buffers; called from the sized constructor when
  // config_.voxel_num > 0. Rolls back partial CUDA allocations on failure.
  void allocateVoxelBuffers();

  // Grow fetch output buffers to hold at least `n` entries.
  void ensureFetchCapacity(const int n) const;

  // Per-voxel atomics buffers, allocated in the sized constructor.
  // Sized to config_.voxel_num once in allocateVoxelBuffers.
  int *d_op_cnt_{nullptr};
  int *d_hit_cnt_{nullptr};

  // Persistent device-side buffer: zero-centered logit; unknown prior == 0.
  float *d_occ_{nullptr};

  // Device buffers for recenter slab planning (one list per axis).
  int *d_recenter_slices_[3]{nullptr, nullptr, nullptr};
  RecenterResult *d_recenter_result_{nullptr};

  // SoA device-side mirror of the input cloud. Reused across frames;
  // grown by ensureCloudCapacity on demand.
  float *d_cloud_x_{nullptr};
  float *d_cloud_y_{nullptr};
  float *d_cloud_z_{nullptr};
  int cloud_capacity_{0};

  // Device buffers for fetchOccupancyAround (reused; grown on demand).
  mutable unsigned int *d_fetch_count_{nullptr};
  mutable float3 *d_fetch_pos_{nullptr};
  mutable unsigned char *d_fetch_state_{nullptr};
  mutable int fetch_capacity_{0};

  // Rays longer than this are clipped to this length before being cast.
  // Lives here (not in Grid::Config) because clipping is a raycaster
  // concern, not a grid-geometry one.
  float max_raycast_range_{25.0f};

  // Log-odds increments and clamps. Defaults are logit(p) for the
  // probabilities listed in setLogOddsParams above.
  float l_hit_{0.847f};       //  logit(0.70)
  float l_miss_{-0.405f};     //  logit(0.40)
  float l_min_{-1.992f};      //  logit(0.12)
  float l_max_{3.476f};       //  logit(0.97)
  float l_free_{-0.0004f};    //  logit(0.499)
  float l_occupied_{1.7346f}; //  logit(0.85)
  float l_unknown_{0.0f};
};

} // namespace cublox
