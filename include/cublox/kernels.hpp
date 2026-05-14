#pragma once

// Public C++ interface to the raycast CUDA kernels.
//
// This header is safe to include from plain .cpp (compiled by the host
// compiler): it contains only POD structs and ordinary function
// declarations, no <<<...>>> syntax and no __global__ / __device__
// qualifiers. The matching definitions live in OccupancyGrid.cu.

#include <cuda_runtime.h>
#include <vector_types.h>

namespace cublox {

// Per-frame configuration uploaded into __constant__ memory before each
// raycast launch. Field order is chosen so the 12-byte int3 / float3
// members come first, followed by the 4-byte scalars — keeps the struct
// tightly packed at 16-byte alignment.
struct RayCastCfg {
  float3 origin;        // sensor position (world frame)
  int3 map_size_i;      // voxels per axis on the local map
  int3 half_map_size_i; // (map_size_i - 1) / 2
  float resolution;
  float inv_resolution;
  float max_range; // rays longer than this are clipped
  int map_vox_num; // == map_size_i.x * map_size_i.y * map_size_i.z
};

// Host-side launcher for rayCastUpdateKernel.
//
// Each input point spawns one CUDA thread that runs an Amanatides–Woo
// traversal from `cfg.origin` to the point, atomic-incrementing
// `d_op_cnt` per traversed voxel and `d_hit_cnt` at the endpoint.
//
// Caller owns all device pointers; this function does not allocate.
// Pass stream=0 for the default (synchronizing) stream.
void launchRayCastUpdate(const RayCastCfg &cfg, const float *d_cloud_x,
                         const float *d_cloud_y, const float *d_cloud_z,
                         int cloud_size, int *d_op_cnt, int *d_hit_cnt,
                         cudaStream_t stream);

// Host-side launcher for applyUpdateKernel.
//
// One CUDA thread per voxel: reads (op_cnt, hit_cnt), folds them into
// the log-odds buffer (hit dominates when both are non-zero — matches
// ProbMap::probabilisticMapFromCache in the CPU reference), clamps to
// [l_min, l_max], then zeros both counters so the next raycast pass
// starts from a clean slate. Voxels with op_cnt == 0 early-exit.
//
// d_occ:    voxel_num floats, persistent log-odds buffer (UNKNOWN = 0).
// d_op_cnt, d_hit_cnt: voxel_num ints, written by rayCastUpdateKernel.
//
// If `d_dirty_count` is non-null, voxels whose log-odds value changes are
// appended as (dirty_idx[i], dirty_val[i]) with `dirty_capacity >= voxel_num`
// (each voxel updates at most once per frame). `d_dirty_count` must be cleared
// to 0 before this launch each frame.
void launchApplyUpdate(float *d_occ, int *d_op_cnt, int *d_hit_cnt,
                       int voxel_num, float l_hit, float l_miss, float l_min,
                       float l_max, cudaStream_t stream,
                       unsigned int *d_dirty_count = nullptr,
                       int *d_dirty_idx = nullptr, float *d_dirty_val = nullptr,
                       unsigned int dirty_capacity = 0);

// Zero `d_occ[h]` for each index in `d_indices[0..n-1]` (one CUDA launch).
void launchClearVoxelsByIndex(float *d_occ, const int *d_indices, int n,
                              int voxel_num, cudaStream_t stream);

// Recenter: clear exiting axis-aligned slabs on GPU. One thread per row;
// inner loop uses contiguous d_occ indices (++h or h += nz) for bandwidth.
void launchClearRecenterSlabsForAxis(float *d_occ, int3 map_size_i,
                                     int3 half_map_size_i,
                                     const int *d_slice_local_values,
                                     int n_slices, int axis,
                                     cudaStream_t stream);

} // namespace cublox
