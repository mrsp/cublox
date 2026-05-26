#include <cublox/OccupancyGrid.hpp>
#include <cublox/kernels.hpp>
#include <cublox/utils.hpp>

#include <cuda_runtime.h>
#include <vector_types.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace cublox {

namespace {

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
  updateOriginAndBound(origin_i);
  allocateVoxelBuffers();
}

OccupancyGrid::~OccupancyGrid() {
  if (d_fetch_state_) {
    cudaFree(d_fetch_state_);
    d_fetch_state_ = nullptr;
  }
  if (d_fetch_pos_) {
    cudaFree(d_fetch_pos_);
    d_fetch_pos_ = nullptr;
  }
  if (d_fetch_count_) {
    cudaFree(d_fetch_count_);
    d_fetch_count_ = nullptr;
  }
  if (d_recenter_result_) {
    cudaFree(d_recenter_result_);
    d_recenter_result_ = nullptr;
  }
  for (int a = 0; a < 3; ++a) {
    if (d_recenter_slices_[a] != nullptr) {
      cudaFree(d_recenter_slices_[a]);
      d_recenter_slices_[a] = nullptr;
    }
  }
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
}

void OccupancyGrid::reset() {
  if (d_occ_ == nullptr) {
    return;
  }

  const size_t n = static_cast<size_t>(config_.voxel_num);
  CUDA_OK(cudaMemset(d_occ_, 0, n * sizeof(float)));
  CUDA_OK(cudaMemset(d_op_cnt_, 0, n * sizeof(int)));
  CUDA_OK(cudaMemset(d_hit_cnt_, 0, n * sizeof(int)));
}

void OccupancyGrid::resetVoxel(const int hash_id) {
  if (d_occ_ == nullptr || hash_id < 0 || hash_id >= config_.voxel_num) {
    return;
  }
  CUDA_OK(cudaMemset(d_occ_ + hash_id, 0, sizeof(float)));
  if (d_op_cnt_ != nullptr) {
    CUDA_OK(cudaMemset(d_op_cnt_ + hash_id, 0, sizeof(int)));
  }
  if (d_hit_cnt_ != nullptr) {
    CUDA_OK(cudaMemset(d_hit_cnt_ + hash_id, 0, sizeof(int)));
  }
}

void OccupancyGrid::recenter(const Eigen::Vector3f &pos) {
  if (!config_.recenter_threshold.has_value() || d_occ_ == nullptr ||
      d_recenter_result_ == nullptr) {
    return;
  }

  // Skip GPU launch when the robot is still inside the recenter window.
  if ((pos - origin_f_).squaredNorm() < config_.recenter_threshold_squared) {
    return;
  }

  RecenterCfg cfg{};
  cfg.pos = make_float3(pos.x(), pos.y(), pos.z());
  cfg.origin_f = make_float3(origin_f_.x(), origin_f_.y(), origin_f_.z());
  cfg.origin_i = make_int3(origin_i_.x(), origin_i_.y(), origin_i_.z());
  cfg.map_size_i = make_int3(config_.map_size_i.x(), config_.map_size_i.y(),
                             config_.map_size_i.z());
  cfg.half_map_size_i =
      make_int3(config_.half_map_size_i.x(), config_.half_map_size_i.y(),
                config_.half_map_size_i.z());
  cfg.resolution = config_.resolution;
  cfg.inv_resolution = config_.resolution_inv;
  cfg.recenter_threshold_sq = config_.recenter_threshold_squared;
  cfg.origin_at_center = config_.origin_at_center;
  cfg.map_vox_num = config_.voxel_num;

  RecenterResult result{};
  launchRecenter(d_occ_, d_op_cnt_, d_hit_cnt_, cfg, d_recenter_slices_[0],
                 d_recenter_slices_[1], d_recenter_slices_[2],
                 d_recenter_result_, &result, /*stream=*/0);

  if (result.action == 0) {
    return;
  }

  const Eigen::Vector3i new_origin_i(
      result.new_origin_i.x, result.new_origin_i.y, result.new_origin_i.z);
  updateOriginAndBound(new_origin_i);
}

void OccupancyGrid::update(const PointCloud &cloud,
                           const Eigen::Vector3f &sensor_origin) {
  if (config_.voxel_num <= 0) {
    return;
  }

  const int n = static_cast<int>(cloud.rows());
  if (n <= 0) {
    return;
  }

  if (first_run_) {
    Eigen::Vector3i origin_i;
    posToGlobalIndex(sensor_origin, config_.resolution_inv,
                     config_.origin_at_center, origin_i);
    updateOriginAndBound(origin_i);
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

  RayCastCfg cfg{};
  cfg.origin =
      make_float3(sensor_origin.x(), sensor_origin.y(), sensor_origin.z());
  cfg.origin_i = make_int3(origin_i_.x(), origin_i_.y(), origin_i_.z());
  cfg.map_size_i = make_int3(config_.map_size_i.x(), config_.map_size_i.y(),
                             config_.map_size_i.z());
  cfg.half_map_size_i =
      make_int3(config_.half_map_size_i.x(), config_.half_map_size_i.y(),
                config_.half_map_size_i.z());
  cfg.resolution = config_.resolution;
  cfg.inv_resolution = config_.resolution_inv;
  cfg.max_range =
      std::min(max_raycast_range_, maxHorizontalInWindowRadius(sensor_origin));
  cfg.map_vox_num = config_.voxel_num;
  cfg.origin_at_center = config_.origin_at_center;

  // Pass 1: walk every ray, atomic-increment op_cnt per voxel and
  // hit_cnt at endpoints.
  launchRayCastUpdate(cfg, d_cloud_x_, d_cloud_y_, d_cloud_z_, n, /*stream=*/0,
                      d_op_cnt_, d_hit_cnt_);
  CUDA_OK(cudaStreamSynchronize(/*stream=*/0));

  // Pass 2: fold counters into d_occ_; record voxels whose log-odds change.
  launchApplyUpdate(d_occ_, d_op_cnt_, d_hit_cnt_, config_.voxel_num, l_hit_,
                    l_miss_, l_min_, l_max_, l_unknown_, /*stream=*/0);
  CUDA_OK(cudaStreamSynchronize(/*stream=*/0));
}

std::vector<OccupancyGrid::OccupancySample> OccupancyGrid::fetchOccupancyAround(
    const Eigen::Vector3f &center, const float radius,
    const std::optional<VoxelState> state_filter, const int max_results) const {
  std::vector<OccupancySample> out;
  if (d_occ_ == nullptr || config_.voxel_num <= 0 || radius <= 0.0f) {
    return out;
  }

  const float effective_r =
      std::min(radius, maxHorizontalInWindowRadius(center));
  if (effective_r <= 0.0f) {
    return out;
  }

  Eigen::Vector3i center_g;
  posToGlobalIndex(center, config_.resolution_inv, config_.origin_at_center,
                   center_g);

  const Eigen::Vector3f world_min =
      center - Eigen::Vector3f::Constant(effective_r);
  const Eigen::Vector3f world_max =
      center + Eigen::Vector3f::Constant(effective_r);

  Eigen::Vector3i g_min;
  Eigen::Vector3i g_max;
  posToGlobalIndex(world_min, config_.resolution_inv, config_.origin_at_center,
                   g_min);
  posToGlobalIndex(world_max, config_.resolution_inv, config_.origin_at_center,
                   g_max);

  const Eigen::Vector3i win_min = origin_i_ - config_.half_map_size_i;
  const Eigen::Vector3i win_max = origin_i_ + config_.half_map_size_i;
  g_min = g_min.cwiseMax(win_min);
  g_max = g_max.cwiseMin(win_max);

  if ((g_max.array() < g_min.array()).any()) {
    return out;
  }

  const int disk_radius_vox = static_cast<int>(
      std::ceil(static_cast<double>(effective_r * config_.resolution_inv)));
  const int gz_min = g_min.z();
  const int gz_max = g_max.z();
  const int nz = gz_max - gz_min + 1;
  const int disk_diam = 2 * disk_radius_vox + 1;
  const std::int64_t est_vox =
      static_cast<std::int64_t>(disk_diam) * disk_diam * nz;
  if (est_vox <= 0 ||
      est_vox > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
    return out;
  }

  const unsigned int output_cap =
      max_results > 0 ? static_cast<unsigned int>(max_results) : 500000u;
  const int fetch_buf =
      static_cast<int>(std::min<std::int64_t>(output_cap, est_vox));
  ensureFetchCapacity(fetch_buf);

  OccupancyFetchCfg cfg{};
  cfg.center_g = make_int3(center_g.x(), center_g.y(), center_g.z());
  cfg.origin_i = make_int3(origin_i_.x(), origin_i_.y(), origin_i_.z());
  cfg.gz_min = gz_min;
  cfg.gz_max = gz_max;
  cfg.disk_radius_vox = disk_radius_vox;
  cfg.center = make_float3(center.x(), center.y(), center.z());
  cfg.radius_sq = effective_r * effective_r;
  cfg.map_size_i = make_int3(config_.map_size_i.x(), config_.map_size_i.y(),
                             config_.map_size_i.z());
  cfg.half_map_size_i =
      make_int3(config_.half_map_size_i.x(), config_.half_map_size_i.y(),
                config_.half_map_size_i.z());
  cfg.resolution = config_.resolution;
  cfg.l_free = l_free_;
  cfg.l_occupied = l_occupied_;
  cfg.l_unknown = l_unknown_;
  cfg.origin_at_center = config_.origin_at_center;
  cfg.state_filter =
      state_filter.has_value() ? static_cast<int>(state_filter.value()) : -1;

  CUDA_OK(cudaMemset(d_fetch_count_, 0, sizeof(unsigned int)));
  const unsigned int max_out =
      max_results > 0
          ? std::min(static_cast<unsigned int>(max_results), output_cap)
          : output_cap;
  launchFetchOccupancyAround(cfg, d_occ_, d_fetch_count_, d_fetch_pos_,
                             d_fetch_state_, output_cap, max_out,
                             /*stream=*/0);
  CUDA_OK(cudaDeviceSynchronize());

  unsigned int n = 0;
  CUDA_OK(cudaMemcpy(&n, d_fetch_count_, sizeof(unsigned int),
                     cudaMemcpyDeviceToHost));
  n = std::min(n, output_cap);
  if (max_results > 0) {
    n = std::min(n, static_cast<unsigned int>(max_results));
  }
  if (n == 0) {
    return out;
  }

  std::vector<float3> pos_host(n);
  std::vector<unsigned char> state_host(n);
  CUDA_OK(cudaMemcpy(pos_host.data(), d_fetch_pos_, n * sizeof(float3),
                     cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(state_host.data(), d_fetch_state_,
                     n * sizeof(unsigned char), cudaMemcpyDeviceToHost));

  out.reserve(n);
  for (unsigned int i = 0; i < n; ++i) {
    OccupancySample sample;
    sample.position =
        Eigen::Vector3f(pos_host[i].x, pos_host[i].y, pos_host[i].z);
    sample.state = static_cast<VoxelState>(state_host[i]);
    out.push_back(sample);
  }
  return out;
}

float OccupancyGrid::storedAt(const int hash_id) const {
  if (d_occ_ == nullptr || hash_id < 0 || hash_id >= config_.voxel_num) {
    return 0.0f;
  }
  float stored = 0.0f;
  CUDA_OK(cudaMemcpy(&stored, d_occ_ + hash_id, sizeof(float),
                     cudaMemcpyDeviceToHost));
  return stored;
}

bool OccupancyGrid::isOccupied(const Eigen::Vector3f &pos) const {
  if (!inside(pos)) {
    return false;
  }
  return occOccupied(storedAt(
      posToHashIndex(pos, config_.map_size_i, config_.half_map_size_i,
                     config_.resolution_inv, config_.origin_at_center)));
}

bool OccupancyGrid::isUnknown(const Eigen::Vector3f &pos) const {
  if (!inside(pos)) {
    return true;
  }
  return occUnknown(storedAt(
      posToHashIndex(pos, config_.map_size_i, config_.half_map_size_i,
                     config_.resolution_inv, config_.origin_at_center)));
}

bool OccupancyGrid::isFree(const Eigen::Vector3f &pos) const {
  if (!inside(pos)) {
    return false;
  }
  return occFree(storedAt(
      posToHashIndex(pos, config_.map_size_i, config_.half_map_size_i,
                     config_.resolution_inv, config_.origin_at_center)));
}

bool OccupancyGrid::isOccupied(const Eigen::Vector3i &id_g) const {
  if (!inside(id_g)) {
    return false;
  }
  return occOccupied(storedAt(
      globalIndexToHashId(id_g, config_.map_size_i, config_.half_map_size_i)));
}

bool OccupancyGrid::isUnknown(const Eigen::Vector3i &id_g) const {
  if (!inside(id_g)) {
    return true;
  }
  return occUnknown(storedAt(
      globalIndexToHashId(id_g, config_.map_size_i, config_.half_map_size_i)));
}

bool OccupancyGrid::isFree(const Eigen::Vector3i &id_g) const {
  if (!inside(id_g)) {
    return false;
  }
  return occFree(storedAt(
      globalIndexToHashId(id_g, config_.map_size_i, config_.half_map_size_i)));
}

bool OccupancyGrid::isOccupied(const int hash_id) const {
  return occOccupied(storedAt(hash_id));
}

bool OccupancyGrid::isUnknown(const int hash_id) const {
  return occUnknown(storedAt(hash_id));
}

bool OccupancyGrid::isFree(const int hash_id) const {
  return occFree(storedAt(hash_id));
}

void OccupancyGrid::allocateVoxelBuffers() {
  const size_t n = static_cast<size_t>(config_.voxel_num);
  if (n == 0) {
    return;
  }

  const size_t int_bytes = n * sizeof(int);
  const size_t flt_bytes = n * sizeof(float);

  try {
    CUDA_OK(cudaMalloc(&d_op_cnt_, int_bytes));
    CUDA_OK(cudaMalloc(&d_hit_cnt_, int_bytes));
    CUDA_OK(cudaMalloc(&d_occ_, flt_bytes));
    CUDA_OK(cudaMemset(d_op_cnt_, 0, int_bytes));
    CUDA_OK(cudaMemset(d_hit_cnt_, 0, int_bytes));
    CUDA_OK(cudaMemset(d_occ_, 0, flt_bytes));
    CUDA_OK(cudaMalloc(&d_recenter_result_, sizeof(RecenterResult)));

    for (int a = 0; a < 3; ++a) {
      const size_t slice_bytes =
          static_cast<size_t>(config_.map_size_i(a)) * sizeof(int);
      CUDA_OK(cudaMalloc(&d_recenter_slices_[a], slice_bytes));
    }
  } catch (...) {
    if (d_recenter_result_) {
      cudaFree(d_recenter_result_);
      d_recenter_result_ = nullptr;
    }
    for (int a = 0; a < 3; ++a) {
      if (d_recenter_slices_[a] != nullptr) {
        cudaFree(d_recenter_slices_[a]);
        d_recenter_slices_[a] = nullptr;
      }
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

void OccupancyGrid::ensureFetchCapacity(const int n) const {
  if (n <= fetch_capacity_) {
    return;
  }
  if (d_fetch_state_) {
    CUDA_OK(cudaFree(d_fetch_state_));
    d_fetch_state_ = nullptr;
  }
  if (d_fetch_pos_) {
    CUDA_OK(cudaFree(d_fetch_pos_));
    d_fetch_pos_ = nullptr;
  }
  if (d_fetch_count_) {
    CUDA_OK(cudaFree(d_fetch_count_));
    d_fetch_count_ = nullptr;
  }

  fetch_capacity_ = n;
  const size_t n_bytes = static_cast<size_t>(fetch_capacity_);
  CUDA_OK(cudaMalloc(&d_fetch_count_, sizeof(unsigned int)));
  CUDA_OK(cudaMalloc(&d_fetch_pos_, n_bytes * sizeof(float3)));
  CUDA_OK(cudaMalloc(&d_fetch_state_, n_bytes * sizeof(unsigned char)));
}

} // namespace cublox
