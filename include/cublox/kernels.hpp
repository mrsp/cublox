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
// raycast launch. Total size is 52 bytes on the GPU.
struct RayCastCfg {
  float3 origin;        // sensor position (world frame) - 12 bytes
  int3 map_size_i;      // voxels per axis on the local map - 12 bytes
  int3 half_map_size_i; // (map_size_i - 1) / 2 - 12 bytes
  float resolution;     // 4 bytes
  float inv_resolution; // 4 bytes
  float max_range;      // rays longer than this are clipped - 4 bytes
  int map_vox_num; // == map_size_i.x * map_size_i.y * map_size_i.z - 4 bytes
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
// One CUDA thread per voxel: reads (op_cnt, hit_cnt), folds them into
// the log-odds buffer (hit dominates when both are non-zero, clamps to
// [l_min, l_max], then zeros both counters so the next raycast pass
// starts from a clean slate. Voxels with op_cnt == 0 early-exit.
//
// d_occ:    voxel_num floats, persistent log-odds buffer (UNKNOWN = 0).
// d_op_cnt, d_hit_cnt: voxel_num ints, written by rayCastUpdateKernel.
//
// If `d_modified_count` is non-null, voxels whose log-odds value changes are
// appended as (modified_idx[i], modified_val[i]) with `modified_capacity >=
// voxel_num` (each voxel updates at most once per frame). `d_modified_count`
// must be cleared to 0 before this launch each frame.
void launchApplyUpdate(float *d_occ, int *d_op_cnt, int *d_hit_cnt,
                       const int voxel_num, const float l_hit,
                       const float l_miss, const float l_min, const float l_max,
                       const cudaStream_t stream,
                       unsigned int *d_modified_count = nullptr,
                       int *d_modified_idx = nullptr,
                       float *d_modified_val = nullptr,
                       const unsigned int modified_capacity = 0);

// Zero `d_occ[h]` for each index in `d_indices[0..n-1]` (one CUDA launch).
void launchClearVoxelsByIndex(float *d_occ, const int *d_indices, const int n,
                              const int voxel_num, const cudaStream_t stream);

// Recenter: clear exiting axis-aligned slabs on GPU. One thread per row;
// inner loop uses contiguous d_occ indices (++h or h += nz) for bandwidth.
void launchClearRecenterSlabsForAxis(float *d_occ, const int3 &map_size_i,
                                     const int3 &half_map_size_i,
                                     const int *d_slice_local_values,
                                     const int n_slices, const int axis,
                                     const cudaStream_t stream);

} // namespace cublox
