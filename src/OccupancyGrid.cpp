#include <cublox/OccupancyGrid.hpp>
#include <cublox/kernels.hpp>
#include <cublox/utils.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace cublox {

namespace {

// Translate a cudaError_t into an exception. We use exceptions (rather
// than logging + return) so callers can't silently keep working with a
// grid whose device state is corrupt.
inline void cudaCheck(cudaError_t err, const char *expr, const char *file,
                      int line) {
  if (err == cudaSuccess) {
    return;
  }
  std::string msg = "cublox::cudaCheck: CUDA error in `";
  msg += expr;
  msg += "`: ";
  msg += cudaGetErrorString(err);
  msg += " (";
  msg += file;
  msg += ":";
  msg += std::to_string(line);
  msg += ")";
  throw std::runtime_error(msg);
}

#define CUDA_OK(call) ::cublox::cudaCheck((call), #call, __FILE__, __LINE__)

} // namespace

OccupancyGrid::OccupancyGrid(const Eigen::Vector3i &half_map_size_i,
                             float resolution, bool origin_at_center,
                             std::optional<double> recenter_threshold,
                             const Eigen::Vector3f &origin)
    : Grid(half_map_size_i, resolution, recenter_threshold, origin_at_center) {
  Eigen::Vector3i origin_i;
  posToGlobalIndex(origin, config_.resolution_inv, config_.origin_at_center,
                   origin_i);
  const Eigen::Vector3f origin_d = origin_i.cast<float>() * config_.resolution;
  updateOriginAndBound(origin_d, origin_i);
  allocateVoxelBuffers_();
}

OccupancyGrid::~OccupancyGrid() {
  // Free in the same order we allocate, and clear pointers
  if (d_op_cnt_) {
    cudaFree(d_op_cnt_);
    d_op_cnt_ = nullptr;
  }
  if (d_hit_cnt_) {
    cudaFree(d_hit_cnt_);
    d_hit_cnt_ = nullptr;
  }
  if (d_occ_) {
    cudaFree(d_occ_);
    d_occ_ = nullptr;
  }
  if (d_cloud_x_) {
    cudaFree(d_cloud_x_);
    d_cloud_x_ = nullptr;
  }
  if (d_cloud_y_) {
    cudaFree(d_cloud_y_);
    d_cloud_y_ = nullptr;
  }
  if (d_cloud_z_) {
    cudaFree(d_cloud_z_);
    d_cloud_z_ = nullptr;
  }
  if (d_dirty_val_) {
    cudaFree(d_dirty_val_);
    d_dirty_val_ = nullptr;
  }
  if (d_dirty_idx_) {
    cudaFree(d_dirty_idx_);
    d_dirty_idx_ = nullptr;
  }
  if (d_dirty_count_) {
    cudaFree(d_dirty_count_);
    d_dirty_count_ = nullptr;
  }
}

void OccupancyGrid::reset() {
  // Host mirror — keep in lock-step with the device buffer so an
  // is*() query right after reset() returns UNKNOWN as expected.
  std::fill(occupancy_buffer_.begin(), occupancy_buffer_.end(), 0.0f);

  // Default-constructed grid: no voxel storage. Sized grid: ctor allocated
  // device buffers whenever voxel_num > 0.
  if (d_occ_ == nullptr) {
    return;
  }
  const size_t n = static_cast<size_t>(config_.voxel_num);
  CUDA_OK(cudaMemset(d_occ_, 0, n * sizeof(float)));
  CUDA_OK(cudaMemset(d_op_cnt_, 0, n * sizeof(int)));
  CUDA_OK(cudaMemset(d_hit_cnt_, 0, n * sizeof(int)));
  if (d_dirty_count_) {
    CUDA_OK(cudaMemset(d_dirty_count_, 0, sizeof(unsigned int)));
  }
}

void OccupancyGrid::resetVoxel(const int &hash_id) {
  if (hash_id < 0 || hash_id >= static_cast<int>(occupancy_buffer_.size())) {
    return;
  }
  occupancy_buffer_[hash_id] = 0.0f;

  // Mirror the single-voxel clear to the device buffer if it exists.
  // A single-voxel cudaMemset is wasteful but resetVoxel is called
  // rarely (only by map-sliding clearVoxelsOutOfGrid for now); a
  // batched clear API can replace this once sliding is GPU-side.
  if (d_occ_ != nullptr) {
    CUDA_OK(cudaMemset(d_occ_ + hash_id, 0, sizeof(float)));
  }
}

void OccupancyGrid::allocateVoxelBuffers_() {
  const size_t n = static_cast<size_t>(config_.voxel_num);
  if (n == 0) {
    return;
  }

  occupancy_buffer_.assign(n, 0.0f);

  const size_t int_bytes = n * sizeof(int);
  const size_t flt_bytes = n * sizeof(float);

  try {
    CUDA_OK(cudaMalloc(&d_op_cnt_, int_bytes));
    CUDA_OK(cudaMalloc(&d_hit_cnt_, int_bytes));
    CUDA_OK(cudaMalloc(&d_occ_, flt_bytes));
    CUDA_OK(cudaMalloc(&d_dirty_count_, sizeof(unsigned int)));
    CUDA_OK(cudaMalloc(&d_dirty_idx_, int_bytes));
    CUDA_OK(cudaMalloc(&d_dirty_val_, flt_bytes));
    CUDA_OK(cudaMemset(d_op_cnt_, 0, int_bytes));
    CUDA_OK(cudaMemset(d_hit_cnt_, 0, int_bytes));
    CUDA_OK(cudaMemset(d_occ_, 0, flt_bytes));
    CUDA_OK(cudaMemset(d_dirty_count_, 0, sizeof(unsigned int)));
    h_dirty_idx_.resize(n);
    h_dirty_val_.resize(n);
  } catch (...) {
    if (d_dirty_val_) {
      cudaFree(d_dirty_val_);
      d_dirty_val_ = nullptr;
    }
    if (d_dirty_idx_) {
      cudaFree(d_dirty_idx_);
      d_dirty_idx_ = nullptr;
    }
    if (d_dirty_count_) {
      cudaFree(d_dirty_count_);
      d_dirty_count_ = nullptr;
    }
    if (d_occ_) {
      cudaFree(d_occ_);
      d_occ_ = nullptr;
    }
    if (d_hit_cnt_) {
      cudaFree(d_hit_cnt_);
      d_hit_cnt_ = nullptr;
    }
    if (d_op_cnt_) {
      cudaFree(d_op_cnt_);
      d_op_cnt_ = nullptr;
    }
    h_dirty_idx_.clear();
    h_dirty_val_.clear();
    occupancy_buffer_.clear();
    throw;
  }
}

void OccupancyGrid::ensureCloudCapacity_(int n) {
  if (n <= cloud_capacity_) {
    return;
  }
  if (d_cloud_x_) {
    CUDA_OK(cudaFree(d_cloud_x_));
    CUDA_OK(cudaFree(d_cloud_y_));
    CUDA_OK(cudaFree(d_cloud_z_));
  }
  cloud_capacity_ = n;
  const size_t bytes = static_cast<size_t>(cloud_capacity_) * sizeof(float);
  CUDA_OK(cudaMalloc(&d_cloud_x_, bytes));
  CUDA_OK(cudaMalloc(&d_cloud_y_, bytes));
  CUDA_OK(cudaMalloc(&d_cloud_z_, bytes));
}

void OccupancyGrid::update(const PointCloud &cloud,
                           const Eigen::Vector3f &sensor_origin) {
  if (config_.voxel_num <= 0) {
    // Grid hasn't been sized yet (default-constructed).
    return;
  }

  const int n = static_cast<int>(cloud.rows());
  if (n <= 0) {
    return;
  }

  if (first_run_) {
    if (config_.recenter_threshold) {
      Eigen::Vector3i origin_i;
      posToGlobalIndex(sensor_origin, config_.resolution_inv,
                       config_.origin_at_center, origin_i);
      updateOriginAndBound(sensor_origin, origin_i);
    }
    first_run_ = false;
  }

  ensureCloudCapacity_(n);

  // Eigen::Matrix<float, Dynamic, 3> defaults to column-major, so each
  // .col(i).data() is a contiguous run of `rows()` floats — exactly what
  // we need for a one-shot cudaMemcpy per axis.
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  CUDA_OK(cudaMemcpyAsync(d_cloud_x_, cloud.col(0).data(), bytes,
                          cudaMemcpyHostToDevice, /*stream=*/0));
  CUDA_OK(cudaMemcpyAsync(d_cloud_y_, cloud.col(1).data(), bytes,
                          cudaMemcpyHostToDevice, /*stream=*/0));
  CUDA_OK(cudaMemcpyAsync(d_cloud_z_, cloud.col(2).data(), bytes,
                          cudaMemcpyHostToDevice, /*stream=*/0));

  // Build the per-frame kernel config. Eigen -> int3/float3 conversion
  // is done here (host-only) so utils.cuh keeps its current shape.
  RayCastCfg cfg{};
  cfg.origin =
      make_float3(sensor_origin.x(), sensor_origin.y(), sensor_origin.z());
  cfg.map_size_i = make_int3(config_.map_size_i.x(), config_.map_size_i.y(),
                             config_.map_size_i.z());
  cfg.half_map_size_i =
      make_int3(config_.half_map_size_i.x(), config_.half_map_size_i.y(),
                config_.half_map_size_i.z());
  cfg.resolution = config_.resolution;
  cfg.inv_resolution = config_.resolution_inv;
  cfg.max_range = max_raycast_range_;
  cfg.map_vox_num = config_.voxel_num;

  // Pass 1: walk every ray, atomic-increment op_cnt per voxel and
  // hit_cnt at endpoints.
  launchRayCastUpdate(cfg, d_cloud_x_, d_cloud_y_, d_cloud_z_, n, d_op_cnt_,
                      d_hit_cnt_, /*stream=*/0);

  CUDA_OK(cudaMemsetAsync(d_dirty_count_, 0, sizeof(unsigned int),
                          /*stream=*/0));

  // Pass 2: fold counters into d_occ_; record voxels whose log-odds change.
  launchApplyUpdate(d_occ_, d_op_cnt_, d_hit_cnt_, config_.voxel_num, l_hit_,
                    l_miss_, l_min_, l_max_, /*stream=*/0, d_dirty_count_,
                    d_dirty_idx_, d_dirty_val_,
                    static_cast<unsigned int>(config_.voxel_num));

  CUDA_OK(cudaDeviceSynchronize());

  unsigned int dirty_n = 0;
  CUDA_OK(cudaMemcpy(&dirty_n, d_dirty_count_, sizeof(unsigned int),
                     cudaMemcpyDeviceToHost));

  dirty_n = std::min(dirty_n, static_cast<unsigned int>(config_.voxel_num));
  if (dirty_n > 0) {
    CUDA_OK(cudaMemcpy(h_dirty_idx_.data(), d_dirty_idx_,
                       static_cast<size_t>(dirty_n) * sizeof(int),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h_dirty_val_.data(), d_dirty_val_,
                       static_cast<size_t>(dirty_n) * sizeof(float),
                       cudaMemcpyDeviceToHost));

    for (unsigned int i = 0U; i < dirty_n; ++i) {
      const int hid = h_dirty_idx_[static_cast<size_t>(i)];
      if (hid >= 0 && hid < config_.voxel_num) {
        occupancy_buffer_[static_cast<size_t>(hid)] =
            h_dirty_val_[static_cast<size_t>(i)];
      }
    }
  }
}

// ─── public is*() queries ────────────────────────────────────────────────────

bool OccupancyGrid::isOccupied(const Eigen::Vector3f &pos) const {
  if (!inside(pos)) {
    return false;
  }
  return isOccupied(occupancy_buffer_[posToHashIndex(
      pos, config_.map_size_i, config_.half_map_size_i, config_.resolution_inv,
      config_.origin_at_center)]);
}

bool OccupancyGrid::isUnknown(const Eigen::Vector3f &pos) const {
  if (!inside(pos)) {
    return true;
  }

  return isUnknown(occupancy_buffer_[posToHashIndex(
      pos, config_.map_size_i, config_.half_map_size_i, config_.resolution_inv,
      config_.origin_at_center)]);
}

bool OccupancyGrid::isKnownFree(const Eigen::Vector3f &pos) const {
  if (!inside(pos)) {
    return false;
  }

  return isKnownFree(occupancy_buffer_[posToHashIndex(
      pos, config_.map_size_i, config_.half_map_size_i, config_.resolution_inv,
      config_.origin_at_center)]);
}

bool OccupancyGrid::isOccupied(const Eigen::Vector3i &id_g) const {
  if (!inside(id_g)) {
    return false;
  }

  return isOccupied(occupancy_buffer_[globalIndexToHashId(
      id_g, config_.map_size_i, config_.half_map_size_i)]);
}

bool OccupancyGrid::isUnknown(const Eigen::Vector3i &id_g) const {
  if (!inside(id_g)) {
    return true;
  }

  return isUnknown(occupancy_buffer_[globalIndexToHashId(
      id_g, config_.map_size_i, config_.half_map_size_i)]);
}

bool OccupancyGrid::isKnownFree(const Eigen::Vector3i &id_g) const {
  if (!inside(id_g)) {
    return false;
  }

  return isKnownFree(occupancy_buffer_[globalIndexToHashId(
      id_g, config_.map_size_i, config_.half_map_size_i)]);
}

bool OccupancyGrid::isOccupied(const int hash_id) const {
  return isOccupied(occupancy_buffer_[hash_id]);
}

bool OccupancyGrid::isUnknown(const int hash_id) const {
  return isUnknown(occupancy_buffer_[hash_id]);
}

bool OccupancyGrid::isKnownFree(const int hash_id) const {
  return isKnownFree(occupancy_buffer_[hash_id]);
}

} // namespace cublox
