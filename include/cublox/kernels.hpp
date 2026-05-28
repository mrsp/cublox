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

// Public C++ interface to the CUDA kernels.
//
// This header is safe to include from plain .cpp (compiled by the host
// compiler): it contains only POD structs and ordinary function
// declarations, no <<<...>>> syntax and no __global__ / __device__
// qualifiers. The matching definitions live in kernels.cu.

#include <cuda_runtime.h>
#include <vector_types.h>

namespace cublox {

// Per-frame configuration uploaded into __constant__ memory before each
// raycast launch. Plain aggregate (no defaulted members) — required for
// `__constant__` storage in CUDA. 65 bytes + 3 bytes padding = 68 bytes on the
// GPU.
struct RayCastCfg {
  float3 origin;        // sensor position (world frame) - 12 bytes
  int3 origin_i;        // sliding-window origin (global voxel index) - 12 bytes
  int3 map_size_i;      // voxels per axis on the local map - 12 bytes
  int3 half_map_size_i; // (map_size_i - 1) / 2 conceptually — see Grid - 12
                        // bytes
  float resolution;     // 4 bytes
  float inv_resolution; // 4 bytes
  float max_range;      // rays longer than this are clipped - 4 bytes
  int map_vox_num;      // map_size_i.x * map_size_i.y * map_size_i.z - 4 bytes
  bool origin_at_center; // corner binning vs centered bins — see Grid - 1 byte
};

// Host-side launcher for rayCastUpdateKernel
//
// Each input point spawns one CUDA thread that runs an Amanatides–Woo
// traversal from `cfg.origin` to the point, atomic-incrementing
// `d_op_cnt` per traversed voxel and `d_hit_cnt` at the endpoint.
//
// Caller owns all device pointers; this function does not allocate.
// Pass stream=0 for the default (synchronizing) stream.
void launchRayCastUpdate(const RayCastCfg &cfg, const float *d_cloud_x,
                         const float *d_cloud_y, const float *d_cloud_z,
                         const int cloud_size, const cudaStream_t stream,
                         int *d_op_cnt, int *d_hit_cnt);

// Host-side launcher for applyUpdateKernel
//
// One CUDA thread per voxel: reads (op_cnt, hit_cnt), folds them into the
// log-odds buffer as l_miss * (op - hit) + l_hit * hit, clamps true logit to
// [l_min, l_max], then zeros both counters for the next raycast pass. Voxels
// with op_cnt == 0 early-exit.
//
// d_occ: stored = logit - l_unknown with l_unknown = logit(0.5) (unknown prior
// stored == 0). Pass the same l_unknown used on host for queries so hits/misses
// update true logit then re-encode to stored.
void launchApplyUpdate(float *d_occ, int *d_op_cnt, int *d_hit_cnt,
                       const int voxel_num, const float l_hit,
                       const float l_miss, const float l_min, const float l_max,
                       const float l_unknown, const cudaStream_t stream,
                       unsigned int *d_modified_count = nullptr,
                       int *d_modified_idx = nullptr,
                       float *d_modified_val = nullptr,
                       const unsigned int modified_capacity = 0);

// Reset `d_occ[h]` for each index in `d_indices[0..n-1]` (one CUDA launch).
void launchClearVoxelsByIndex(float *d_occ, const int *d_indices, const int n,
                              const int voxel_num, const cudaStream_t stream);

// Recenter: plan exiting slabs on GPU, clear them, optionally full-reset.
struct RecenterCfg {
  float3 pos;
  float3 origin_f;
  int3 origin_i;
  int3 map_size_i;
  int3 half_map_size_i;
  float resolution;
  float inv_resolution;
  float recenter_threshold_sq;
  bool origin_at_center;
  int map_vox_num;
};

struct RecenterResult {
  int action; // 0 = none, 1 = clear slabs, 2 = full map reset
  int3 new_origin_i;
  int n_slices[3];
};

// Plans slice IDs on device, clears exiting and entering slabs (or full-resets
// `d_occ`), and writes the outcome to `h_result`.
void launchRecenter(float *d_occ, int *d_op_cnt, int *d_hit_cnt,
                    const RecenterCfg &cfg, int *d_slices_x, int *d_slices_y,
                    int *d_slices_z, RecenterResult *d_result,
                    RecenterResult *h_result, const cudaStream_t stream);

// Recenter: reset exiting axis-aligned slabs on GPU.
// One thread per row; inner loop uses contiguous d_occ indices (++h or h +=
// nz) for bandwidth.
void launchClearRecenterSlabsForAxis(float *d_occ, int *d_op_cnt,
                                     int *d_hit_cnt, const int3 &map_size_i,
                                     const int3 &half_map_size_i,
                                     const int *d_slice_local_values,
                                     const int n_slices, const int axis,
                                     const cudaStream_t stream);

// Per-frame config for fetchOccupancySphereKernel (uploaded to __constant__).
struct OccupancyFetchCfg {
  int3 center_g; // sphere center as global voxel index
  int3 origin_i; // sliding-window origin (global voxel index)
  int3 map_size_i;
  int3 half_map_size_i;
  int gz_min;          // inclusive global z slab range intersecting sphere
  int gz_max;          // inclusive
  int disk_radius_vox; // xy half-extent in voxels (ceil(r/res))
  float3 center;       // sphere center in world coordinates [m]
  float radius_sq;     // squared fetch radius [m^2]
  float resolution;
  float l_free;
  float l_occupied;
  float l_unknown;
  bool origin_at_center;
  int state_filter; // -1 = all; 0 = unknown; 1 = free; 2 = occupied
};

// Compact voxels inside a bounding sphere from `d_occ` into device buffers.
// Launches a 3D grid over z-slabs with xy disks (no full AABB cube).
// `d_out_count` is zeroed by the caller before launch. Writes at most
// `output_capacity` entries; `max_results` (0 = unlimited) caps logical count.
void launchFetchOccupancyAround(const OccupancyFetchCfg &cfg,
                                const float *d_occ, unsigned int *d_out_count,
                                float3 *d_out_pos, unsigned char *d_out_state,
                                const unsigned int output_capacity,
                                const unsigned int max_results,
                                const cudaStream_t stream);

} // namespace cublox
