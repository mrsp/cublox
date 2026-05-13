#pragma once

#include <Eigen/Dense>
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

  // Required by the Grid interface (pure virtuals). Currently no-ops; will
  // be implemented once the occupancy buffer lifecycle is wired up.
  void reset() override;
  void resetVoxel(const int &hash_id) override;

  // Run one raycast pass for `input_cloud` originating at `sensor_origin`.
  void update(const PointCloud &input_cloud,
              const Eigen::Vector3f &sensor_origin);

  // Knobs that will eventually move into a proper Config. Inline so the
  // call site stays cheap and so we don't drag a translation unit into a
  // setter that's literally one store.
  void setMaxRaycastRange(float range) { max_raycast_range_ = range; }
  float getMaxRaycastRange() const { return max_raycast_range_; }

  // Log-odds increments / clamps passed to applyUpdateKernel each frame.
  // Defaults values are (p_hit=0.70, p_miss=0.40, p_min=0.12, p_max=0.97). Call
  // this once during setup, or pass the logits directly if you've already
  // computed them.
  void setLogOddsParams(float l_hit, float l_miss, float l_min, float l_max) {
    l_hit_ = l_hit;
    l_miss_ = l_miss;
    l_min_ = l_min;
    l_max_ = l_max;
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
  // Threshold separating occupied / free / unknown in the log-odds
  // buffer. `kOccThr == kFreeThr == 0` partitions on the sign of the
  // log-odds value (positive → occupied, negative → free, zero →
  // unknown).
  static constexpr float kOccThr = 0.0f;
  static constexpr float kFreeThr = 0.0f;
  static constexpr float kDefaultMaxRaycastRange = 10.0f;

  static bool isOccupied(float v) { return v > kOccThr; }
  static bool isKnownFree(float v) { return v < kFreeThr; }
  static bool isUnknown(float v) { return v == 0.0f; }

  // Grow d_cloud_{x,y,z}_ to hold at least `n` points. Cheap no-op when
  // capacity already suffices.
  void ensureCloudCapacity_(int n);

  // Host + device voxel buffers; called from the sized constructor when
  // config_.voxel_num > 0. Rolls back partial CUDA allocations on failure.
  void allocateVoxelBuffers_();

  // Host-side mirror of d_occ_. Updated incrementally from GPU dirty lists
  // inside update() instead of copying the entire volume each frame.
  std::vector<float> occupancy_buffer_;

  // GPU + host staging for voxels whose log-odds change in applyUpdateKernel.
  // Capacity equals voxel_num (at most one list entry per voxel per frame).
  unsigned int *d_dirty_count_{nullptr};
  int *d_dirty_idx_{nullptr};
  float *d_dirty_val_{nullptr};
  std::vector<int> h_dirty_idx_;
  std::vector<float> h_dirty_val_;

  // Per-voxel atomics buffers, allocated in the sized constructor.
  // Sized to config_.voxel_num once.
  int *d_op_cnt_{nullptr};
  int *d_hit_cnt_{nullptr};

  // Persistent device-side log-odds buffer (UNKNOWN == 0.0f). Same
  // length as occupancy_buffer_. applyUpdateKernel reads/writes this
  // every frame.
  float *d_occ_{nullptr};

  // SoA device-side mirror of the input cloud. Reused across frames;
  // grown by ensureCloudCapacity_ on demand.
  float *d_cloud_x_{nullptr};
  float *d_cloud_y_{nullptr};
  float *d_cloud_z_{nullptr};
  int cloud_capacity_{0};

  // Rays longer than this are clipped to this length before being cast.
  // Lives here (not in Grid::Config) because clipping is a raycaster
  // concern, not a grid-geometry one.
  float max_raycast_range_{kDefaultMaxRaycastRange};

  // Log-odds increments and clamps. Defaults are logit(p) for the
  // probabilities listed in setLogOddsParams above.
  float l_hit_{0.847f};   //  logit(0.70)
  float l_miss_{-0.405f}; //  logit(0.40)
  float l_min_{-1.992f};  //  logit(0.12)
  float l_max_{3.476f};   //  logit(0.97)
};

} // namespace cublox
