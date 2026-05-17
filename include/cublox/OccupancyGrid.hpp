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
#include <vector>

#include <cublox/Grid.hpp>

namespace cublox {

using PointCloud = Eigen::Matrix<float, Eigen::Dynamic, 3>;

class OccupancyGrid : public Grid {
public:
  enum class VoxelState {
    UNKNOWN = 0,
    KNOWN_FREE = 1,
    OCCUPIED = 2,
  };

  OccupancyGrid(const Eigen::Vector3i &half_map_size_i, float resolution,
                bool origin_at_center, std::optional<double> recenter_threshold,
                const Eigen::Vector3f &origin = Eigen::Vector3f::Zero());
  OccupancyGrid() = default;

  ~OccupancyGrid();

  void reset() override;
  void resetVoxel(const int &hash_id) override;
  void resetVoxels(const std::vector<int> &hash_ids) override;
  void clearRecenterExitSlabs(const std::vector<int> &x_slices,
                              const std::vector<int> &y_slices,
                              const std::vector<int> &z_slices) override;

  // Run one raycast pass for `input_cloud` originating at `sensor_origin`.
  void update(const PointCloud &input_cloud,
              const Eigen::Vector3f &sensor_origin);

  // Knobs that will eventually move into a proper Config. Inline so the
  // call site stays cheap and so we don't drag a translation unit into a
  // setter that's literally one store.
  void setMaxRaycastRange(const float range) { max_raycast_range_ = range; }
  float getMaxRaycastRange() const { return max_raycast_range_; }

  // Log-odds increments / clamps passed to applyUpdateKernel each frame.
  // Defaults values are (p_hit=0.70, p_miss=0.40, p_min=0.12, p_max=0.97). Call
  // this once during setup, or pass the logits directly if you've already
  // computed them.
  void setLogOddsParams(const float l_hit, const float l_miss,
                        const float l_min, const float l_max,
                        const float l_free, const float l_occupied) {
    l_hit_ = l_hit;
    l_miss_ = l_miss;
    l_min_ = l_min;
    l_max_ = l_max;
    l_free_ = l_free;
    l_occupied_ = l_occupied;
    l_unknown_ = 0.5f * (l_free + l_occupied);
  }

  bool isOccupied(const Eigen::Vector3f &pos) const;
  bool isUnknown(const Eigen::Vector3f &pos) const;
  bool isKnownFree(const Eigen::Vector3f &pos) const;
  bool isOccupied(const Eigen::Vector3i &id_g) const;
  bool isUnknown(const Eigen::Vector3i &id_g) const;
  bool isKnownFree(const Eigen::Vector3i &id_g) const;
  bool isOccupied(const int hash_id) const;
  bool isUnknown(const int hash_id) const;
  bool isKnownFree(const int hash_id) const;

private:
  bool first_run_{true};

  // Buffer holds zero-centered logits: stored == logit - l_unknown_. Unknown
  // prior is stored == 0 (cudaMemset-friendly); decode with + l_unknown_.
  float storedToLogit_(float s) const noexcept { return s + l_unknown_; }
  bool occOccupied_(float s) const noexcept {
    return storedToLogit_(s) >= l_occupied_;
  }
  bool occKnownFree_(float s) const noexcept {
    return storedToLogit_(s) < l_free_;
  }
  bool occUnknown_(float s) const noexcept {
    const float l = storedToLogit_(s);
    return l >= l_free_ && l < l_occupied_;
  }

  // Grow d_cloud_{x,y,z}_ to hold at least `n` points. Cheap no-op when
  // capacity already suffices.
  void ensureCloudCapacity_(int n);

  // Host + device voxel buffers; called from the sized constructor when
  // config_.voxel_num > 0. Rolls back partial CUDA allocations on failure.
  void allocateVoxelBuffers_();

  // Host-side mirror of d_occ_. Updated incrementally from GPU modified lists
  // inside update() instead of copying the entire volume each frame.
  std::vector<float> occupancy_buffer_;

  // GPU + host staging for voxels whose log-odds change in applyUpdateKernel.
  // Capacity equals voxel_num (at most one list entry per voxel per frame).
  unsigned int *d_modified_count_{nullptr};
  int *d_modified_idx_{nullptr};
  float *d_modified_val_{nullptr};
  std::vector<int> h_modified_idx_;
  std::vector<float> h_modified_val_;

  // Per-voxel atomics buffers, allocated in the sized constructor.
  // Sized to config_.voxel_num once.
  int *d_op_cnt_{nullptr};
  int *d_hit_cnt_{nullptr};

  // Persistent device-side buffer: same encoding as occupancy_buffer_
  // (zero-centered logit; unknown prior == 0). applyUpdateKernel reads/writes
  // each frame.
  float *d_occ_{nullptr};

  // Recenter: one stream + small slice-ID buffer per axis so x/y/z slab
  // kernels can run concurrently (overlapping execution after H2D).
  cudaStream_t recenter_stream_[3]{nullptr, nullptr, nullptr};
  int *d_recenter_slices_[3]{nullptr, nullptr, nullptr};
  // Page-locked host staging for slice IDs (true async cudaMemcpyAsync H2D).
  int *h_recenter_slices_pin_[3]{nullptr, nullptr, nullptr};

  // SoA device-side mirror of the input cloud. Reused across frames;
  // grown by ensureCloudCapacity_ on demand.
  float *d_cloud_x_{nullptr};
  float *d_cloud_y_{nullptr};
  float *d_cloud_z_{nullptr};
  int cloud_capacity_{0};

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
  float l_unknown_{0.5f * (l_free_ + l_occupied_)};
};

} // namespace cublox
