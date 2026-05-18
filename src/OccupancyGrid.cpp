#include <cublox/OccupancyGrid.hpp>
#include <cublox/kernels.hpp>
#include <cublox/utils.hpp>

#include <cuda_runtime.h>
#include <vector_types.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
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

// Loose upper bound on slab voxel touches (double-counts corners). Chooses
// between targeted host writes and full-volume D2H after GPU slab clear.
size_t upperBoundSlabTouches(const Grid::Config &cfg,
                             const std::vector<int> &x_slices,
                             const std::vector<int> &y_slices,
                             const std::vector<int> &z_slices) {
  size_t v = 0;
  if (!x_slices.empty()) {
    v += x_slices.size() * static_cast<size_t>(cfg.map_size_i.y()) *
         static_cast<size_t>(cfg.map_size_i.z());
  }

  if (!y_slices.empty()) {
    v += y_slices.size() * static_cast<size_t>(cfg.map_size_i.x()) *
         static_cast<size_t>(cfg.map_size_i.z());
  }

  if (!z_slices.empty()) {
    v += z_slices.size() * static_cast<size_t>(cfg.map_size_i.x()) *
         static_cast<size_t>(cfg.map_size_i.y());
  }

  return v;
}

// Host mirror: set exiting slabs to stored 0 (unknown prior in zero-centered
// encoding). Device slabs cleared by launchClearRecenterSlabsForAxis.
void zeroHostMirrorSlabs(std::vector<float> &occ, const Grid::Config &cfg,
                         const std::vector<int> &x_slices,
                         const std::vector<int> &y_slices,
                         const std::vector<int> &z_slices) {
  const size_t bufsz = occ.size();
  auto per_axis = [&](const std::vector<int> &slices, int axis) {
    if (slices.empty()) {
      return;
    }
    const std::array<int, 3> ids{axis, (axis + 1) % 3, (axis + 2) % 3};
    const int h1 = cfg.half_map_size_i(ids[1]);
    const int h2 = cfg.half_map_size_i(ids[2]);
    for (const int idd : slices) {
      for (int a = -h1; a <= h1; ++a) {
        for (int b = -h2; b <= h2; ++b) {
          Eigen::Vector3i tl;
          tl(ids[0]) = idd;
          tl(ids[1]) = a;
          tl(ids[2]) = b;
          const int h =
              localIndexToHashId(tl, cfg.map_size_i, cfg.half_map_size_i);
          if (h >= 0 && static_cast<size_t>(h) < bufsz) {
            occ[static_cast<size_t>(h)] = 0.0f;
          }
        }
      }
    }
  };

  per_axis(x_slices, 0);
  per_axis(y_slices, 1);
  per_axis(z_slices, 2);
}

} // namespace

// ─── public interface ────────────────────────────────────────────────────
OccupancyGrid::OccupancyGrid(const Eigen::Vector3i &half_map_size_i,
                             const float resolution,
                             const bool origin_at_center,
                             const std::optional<double> recenter_threshold,
                             const Eigen::Vector3f &origin)
    : Grid(half_map_size_i, resolution, recenter_threshold, origin_at_center) {
  Eigen::Vector3i origin_i;
  posToGlobalIndex(origin, config_.resolution_inv, config_.origin_at_center,
                   origin_i);
  const Eigen::Vector3f origin_d = origin_i.cast<float>() * config_.resolution;
  updateOriginAndBound(origin_d, origin_i);
  allocateVoxelBuffers();
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
  if (d_modified_val_) {
    cudaFree(d_modified_val_);
    d_modified_val_ = nullptr;
  }
  if (d_modified_idx_) {
    cudaFree(d_modified_idx_);
    d_modified_idx_ = nullptr;
  }
  if (d_modified_count_) {
    cudaFree(d_modified_count_);
    d_modified_count_ = nullptr;
  }
  for (int a = 0; a < 3; ++a) {
    if (recenter_stream_[a] != nullptr) {
      cudaStreamDestroy(recenter_stream_[a]);
      recenter_stream_[a] = nullptr;
    }
    if (d_recenter_slices_[a] != nullptr) {
      cudaFree(d_recenter_slices_[a]);
      d_recenter_slices_[a] = nullptr;
    }
    if (h_recenter_slices_pin_[a] != nullptr) {
      cudaFreeHost(h_recenter_slices_pin_[a]);
      h_recenter_slices_pin_[a] = nullptr;
    }
  }
}

void OccupancyGrid::reset() {
  // Stored unknown prior == 0 (decodes to (l_free + l_occ)/2 via + l_unknown_).
  std::fill(occupancy_buffer_.begin(), occupancy_buffer_.end(), 0.0f);

  // Default-constructed grid: no voxel storage. Sized grid:
  // allocateVoxelBuffers allocated device buffers whenever voxel_num > 0.
  if (d_occ_ == nullptr) {
    return;
  }

  const size_t n = static_cast<size_t>(config_.voxel_num);
  CUDA_OK(cudaMemset(d_occ_, 0, n * sizeof(float)));
  CUDA_OK(cudaMemset(d_op_cnt_, 0, n * sizeof(int)));
  CUDA_OK(cudaMemset(d_hit_cnt_, 0, n * sizeof(int)));
  if (d_modified_count_) {
    CUDA_OK(cudaMemset(d_modified_count_, 0, sizeof(unsigned int)));
  }
}

void OccupancyGrid::resetVoxel(const int hash_id) {
  if (hash_id < 0 || hash_id >= static_cast<int>(occupancy_buffer_.size())) {
    return;
  }
  occupancy_buffer_[hash_id] = 0.0f;

  if (d_occ_ != nullptr) {
    CUDA_OK(cudaMemset(d_occ_ + hash_id, 0, sizeof(float)));
  }
}

void OccupancyGrid::resetVoxels(const std::vector<int> &hash_ids) {
  if (hash_ids.empty()) {
    return;
  }

  const size_t buf_sz = occupancy_buffer_.size();
  for (const int h : hash_ids) {
    if (h >= 0 && static_cast<size_t>(h) < buf_sz) {
      occupancy_buffer_[static_cast<size_t>(h)] = 0.0f;
    }
  }

  if (d_occ_ == nullptr) {
    return;
  }

  const int cap = config_.voxel_num;
  const int total = static_cast<int>(hash_ids.size());
  int offset = 0;
  while (offset < total) {
    const int chunk = std::min(cap, total - offset);
    CUDA_OK(cudaMemcpy(d_modified_idx_, hash_ids.data() + offset,
                       static_cast<size_t>(chunk) * sizeof(int),
                       cudaMemcpyHostToDevice));
    launchClearVoxelsByIndex(d_occ_, d_modified_idx_, chunk, cap,
                             /*stream=*/0);
    offset += chunk;
  }
  CUDA_OK(cudaDeviceSynchronize());
}

void OccupancyGrid::clearRecenterExitSlabs(const std::vector<int> &x_slices,
                                           const std::vector<int> &y_slices,
                                           const std::vector<int> &z_slices) {
  const bool any = !x_slices.empty() || !y_slices.empty() || !z_slices.empty();
  if (!any) {
    return;
  }

  if (d_occ_ == nullptr || d_recenter_slices_[0] == nullptr ||
      h_recenter_slices_pin_[0] == nullptr) {
    Grid::clearRecenterExitSlabs(x_slices, y_slices, z_slices);
    return;
  }

  const int3 ms = make_int3(config_.map_size_i.x(), config_.map_size_i.y(),
                            config_.map_size_i.z());
  const int3 hs =
      make_int3(config_.half_map_size_i.x(), config_.half_map_size_i.y(),
                config_.half_map_size_i.z());

  auto issue = [&](const std::vector<int> &slices, int axis) {
    if (slices.empty()) {
      return;
    }
    const int n = static_cast<int>(slices.size());
    const cudaStream_t st = recenter_stream_[axis];
    std::memcpy(h_recenter_slices_pin_[axis], slices.data(),
                static_cast<size_t>(n) * sizeof(int));
    CUDA_OK(cudaMemcpyAsync(
        d_recenter_slices_[axis], h_recenter_slices_pin_[axis],
        static_cast<size_t>(n) * sizeof(int), cudaMemcpyHostToDevice, st));
    launchClearRecenterSlabsForAxis(d_occ_, ms, hs, d_recenter_slices_[axis], n,
                                    axis, st);
  };

  issue(x_slices, 0);
  issue(y_slices, 1);
  issue(z_slices, 2);
  CUDA_OK(cudaDeviceSynchronize());

  const size_t slab_ub =
      upperBoundSlabTouches(config_, x_slices, y_slices, z_slices);
  const size_t d2h_threshold =
      static_cast<size_t>(std::max(1, config_.voxel_num)) / 4;
  if (slab_ub <= d2h_threshold) {
    zeroHostMirrorSlabs(occupancy_buffer_, config_, x_slices, y_slices,
                        z_slices);
  } else {
    CUDA_OK(cudaMemcpy(occupancy_buffer_.data(), d_occ_,
                       static_cast<size_t>(config_.voxel_num) * sizeof(float),
                       cudaMemcpyDeviceToHost));
  }
}

void OccupancyGrid::update(const PointCloud &cloud,
                           const Eigen::Vector3f &sensor_origin) {
  if (config_.voxel_num <= 0) {
    // Grid hasn't been sized yet (default-constructed).
    return;
  }

  const int n = static_cast<int>(cloud.rows());
  if (n <= 0) {
    // Empty cloud. Nothing to do.
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

  ensureCloudCapacity(n);

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
  launchRayCastUpdate(cfg, d_cloud_x_, d_cloud_y_, d_cloud_z_, n, /*stream=*/0,
                      d_op_cnt_, d_hit_cnt_);
  CUDA_OK(cudaMemsetAsync(d_modified_count_, 0, sizeof(unsigned int),
                          /*stream=*/0));

  // Pass 2: fold counters into d_occ_; record voxels whose log-odds change.
  launchApplyUpdate(d_occ_, d_op_cnt_, d_hit_cnt_, config_.voxel_num, l_hit_,
                    l_miss_, l_min_, l_max_, l_unknown_, /*stream=*/0,
                    d_modified_count_, d_modified_idx_, d_modified_val_,
                    static_cast<unsigned int>(config_.voxel_num));
  CUDA_OK(cudaDeviceSynchronize());

  // Copy the modified voxels to the host occupancy buffer.
  unsigned int modified_n = 0;
  CUDA_OK(cudaMemcpy(&modified_n, d_modified_count_, sizeof(unsigned int),
                     cudaMemcpyDeviceToHost));
  modified_n =
      std::min(modified_n, static_cast<unsigned int>(config_.voxel_num));
  if (modified_n > 0) {
    CUDA_OK(cudaMemcpy(h_modified_idx_.data(), d_modified_idx_,
                       static_cast<size_t>(modified_n) * sizeof(int),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h_modified_val_.data(), d_modified_val_,
                       static_cast<size_t>(modified_n) * sizeof(float),
                       cudaMemcpyDeviceToHost));

    for (unsigned int i = 0; i < modified_n; ++i) {
      const int hid = h_modified_idx_[static_cast<size_t>(i)];
      if (hid >= 0 && hid < config_.voxel_num) {
        occupancy_buffer_[static_cast<size_t>(hid)] =
            h_modified_val_[static_cast<size_t>(i)];
      }
    }
  }
}

bool OccupancyGrid::isOccupied(const Eigen::Vector3f &pos) const {
  if (!inside(pos)) {
    return false;
  }
  return occOccupied(occupancy_buffer_[posToHashIndex(
      pos, config_.map_size_i, config_.half_map_size_i, config_.resolution_inv,
      config_.origin_at_center)]);
}

bool OccupancyGrid::isUnknown(const Eigen::Vector3f &pos) const {
  if (!inside(pos)) {
    return true;
  }

  return occUnknown(occupancy_buffer_[posToHashIndex(
      pos, config_.map_size_i, config_.half_map_size_i, config_.resolution_inv,
      config_.origin_at_center)]);
}

bool OccupancyGrid::isFree(const Eigen::Vector3f &pos) const {
  if (!inside(pos)) {
    return false;
  }

  return occFree(occupancy_buffer_[posToHashIndex(
      pos, config_.map_size_i, config_.half_map_size_i, config_.resolution_inv,
      config_.origin_at_center)]);
}

bool OccupancyGrid::isOccupied(const Eigen::Vector3i &id_g) const {
  if (!inside(id_g)) {
    return false;
  }

  return occOccupied(occupancy_buffer_[globalIndexToHashId(
      id_g, config_.map_size_i, config_.half_map_size_i)]);
}

bool OccupancyGrid::isUnknown(const Eigen::Vector3i &id_g) const {
  if (!inside(id_g)) {
    return true;
  }

  return occUnknown(occupancy_buffer_[globalIndexToHashId(
      id_g, config_.map_size_i, config_.half_map_size_i)]);
}

bool OccupancyGrid::isFree(const Eigen::Vector3i &id_g) const {
  if (!inside(id_g)) {
    return false;
  }

  return occFree(occupancy_buffer_[globalIndexToHashId(
      id_g, config_.map_size_i, config_.half_map_size_i)]);
}

bool OccupancyGrid::isOccupied(const int hash_id) const {
  return occOccupied(occupancy_buffer_[hash_id]);
}

bool OccupancyGrid::isUnknown(const int hash_id) const {
  return occUnknown(occupancy_buffer_[hash_id]);
}

bool OccupancyGrid::isFree(const int hash_id) const {
  return occFree(occupancy_buffer_[hash_id]);
}

// ─── private interface ────────────────────────────────────────────────────
void OccupancyGrid::allocateVoxelBuffers() {
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
    CUDA_OK(cudaMalloc(&d_modified_count_, sizeof(unsigned int)));
    CUDA_OK(cudaMalloc(&d_modified_idx_, int_bytes));
    CUDA_OK(cudaMalloc(&d_modified_val_, flt_bytes));
    CUDA_OK(cudaMemset(d_op_cnt_, 0, int_bytes));
    CUDA_OK(cudaMemset(d_hit_cnt_, 0, int_bytes));
    CUDA_OK(cudaMemset(d_occ_, 0, flt_bytes));
    CUDA_OK(cudaMemset(d_modified_count_, 0, sizeof(unsigned int)));

    h_modified_idx_.resize(n);
    h_modified_val_.resize(n);
    for (int a = 0; a < 3; ++a) {
      CUDA_OK(cudaStreamCreate(&recenter_stream_[a]));
      const size_t slice_bytes =
          static_cast<size_t>(config_.map_size_i(a)) * sizeof(int);
      CUDA_OK(cudaMalloc(&d_recenter_slices_[a], slice_bytes));
      CUDA_OK(cudaMallocHost(&h_recenter_slices_pin_[a], slice_bytes));
    }
  } catch (...) {
    for (int a = 0; a < 3; ++a) {
      if (h_recenter_slices_pin_[a] != nullptr) {
        cudaFreeHost(h_recenter_slices_pin_[a]);
        h_recenter_slices_pin_[a] = nullptr;
      }
      if (d_recenter_slices_[a] != nullptr) {
        cudaFree(d_recenter_slices_[a]);
        d_recenter_slices_[a] = nullptr;
      }
      if (recenter_stream_[a] != nullptr) {
        cudaStreamDestroy(recenter_stream_[a]);
        recenter_stream_[a] = nullptr;
      }
    }
    if (d_modified_val_) {
      cudaFree(d_modified_val_);
      d_modified_val_ = nullptr;
    }
    if (d_modified_idx_) {
      cudaFree(d_modified_idx_);
      d_modified_idx_ = nullptr;
    }
    if (d_modified_count_) {
      cudaFree(d_modified_count_);
      d_modified_count_ = nullptr;
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
    h_modified_idx_.clear();
    h_modified_val_.clear();
    occupancy_buffer_.clear();
    throw std::runtime_error(
        "cublox::OccupancyGrid: failed to allocate voxel buffers");
  }
}

void OccupancyGrid::ensureCloudCapacity(const int n) {
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

} // namespace cublox
