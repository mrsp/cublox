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

#include <cublox/utils.cuh>

#include <math_constants.h>

namespace cublox {

__constant__ RayCastCfg c_cfg;

__device__ inline bool
insideSlidingWindow(const int3 &id_g, const int3 &origin_i, const int3 &half) {
  return abs(id_g.x - origin_i.x) <= half.x &&
         abs(id_g.y - origin_i.y) <= half.y &&
         abs(id_g.z - origin_i.z) <= half.z;
}

// ─────────────────────────────────────────────────
// 3D Fast Voxel Traversal  (Amanatides & Woo, 1987)
// ─────────────────────────────────────────────────
__global__ void rayCastUpdateKernel(const float *__restrict__ cloud_x,
                                    const float *__restrict__ cloud_y,
                                    const float *__restrict__ cloud_z,
                                    const int cloud_size,
                                    int *__restrict__ op_cnt,
                                    int *__restrict__ hit_cnt) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= cloud_size) {
    return;
  }

  // ---- 1. load endpoint and origin (coalesced) ----
  float ex = cloud_x[tid], ey = cloud_y[tid], ez = cloud_z[tid];
  const float ox = c_cfg.origin.x, oy = c_cfg.origin.y, oz = c_cfg.origin.z;

  // ---- 2. clip ray length to max_range ----
  float dx = ex - ox, dy = ey - oy, dz = ez - oz;
  float L = sqrtf(dx * dx + dy * dy + dz * dz);
  if (L < 1e-6f || !isfinite(L) || !isfinite(ex) || !isfinite(ey) ||
      !isfinite(ez)) {
    return;
  }

  // Clipped rays mark free space only; the clip point is not a sensor hit.
  const bool clipped = (L > c_cfg.max_range);
  if (clipped) {
    const float s = c_cfg.max_range / L;
    ex = ox + dx * s;
    ey = oy + dy * s;
    ez = oz + dz * s;
    dx *= s;
    dy *= s;
    dz *= s;
    L = c_cfg.max_range;
  }

  const float invL = 1.0f / L;
  const float ux = dx * invL, uy = dy * invL, uz = dz * invL;

  // ---- 3. Amanatides–Woo init in registers ----
  const bool origin_ctr = c_cfg.origin_at_center;
  int ix = posToGlobalIndex(ox, c_cfg.inv_resolution, origin_ctr);
  int iy = posToGlobalIndex(oy, c_cfg.inv_resolution, origin_ctr);
  int iz = posToGlobalIndex(oz, c_cfg.inv_resolution, origin_ctr);
  const int ex_i = posToGlobalIndex(ex, c_cfg.inv_resolution, origin_ctr);
  const int ey_i = posToGlobalIndex(ey, c_cfg.inv_resolution, origin_ctr);
  const int ez_i = posToGlobalIndex(ez, c_cfg.inv_resolution, origin_ctr);
  const int sx = (ux > 0) - (ux < 0);
  const int sy = (uy > 0) - (uy < 0);
  const int sz = (uz > 0) - (uz < 0);
  const float tDx = (sx == 0) ? CUDART_INF_F : fabsf(c_cfg.resolution / ux);
  const float tDy = (sy == 0) ? CUDART_INF_F : fabsf(c_cfg.resolution / uy);
  const float tDz = (sz == 0) ? CUDART_INF_F : fabsf(c_cfg.resolution / uz);
  // Corner grids: voxel i spans [i, i+1) in voxel coordinates (faces at k*r).
  // Center grids: voxel i spans [-0.5, +0.5) relative to cell center i*r,
  // so faces lie at planes (k + sx*0.5)*r stepped by full resolution.
  const float bx = origin_ctr
                       ? (((float)ix + 0.5f * (float)sx) * c_cfg.resolution)
                       : (((float)ix + (float)(sx > 0)) * c_cfg.resolution);
  const float by = origin_ctr
                       ? (((float)iy + 0.5f * (float)sy) * c_cfg.resolution)
                       : (((float)iy + (float)(sy > 0)) * c_cfg.resolution);
  const float bz = origin_ctr
                       ? (((float)iz + 0.5f * (float)sz) * c_cfg.resolution)
                       : (((float)iz + (float)(sz > 0)) * c_cfg.resolution);
  float tMx = (sx == 0) ? CUDART_INF_F : (bx - ox) / ux;
  float tMy = (sy == 0) ? CUDART_INF_F : (by - oy) / uy;
  float tMz = (sz == 0) ? CUDART_INF_F : (bz - oz) / uz;

  // ---- 4. traverse: 1 atomic per voxel, period ----
  // Manhattan distance is an upper bound on DDA steps; avoid using while(true)
  // loops on the GPU
  const int max_steps = abs(ex_i - ix) + abs(ey_i - iy) + abs(ez_i - iz) + 4;
  for (int step = 0; step < max_steps; ++step) {
    if (ix == ex_i && iy == ey_i && iz == ez_i) {
      break;
    }
    const int3 id_g = make_int3(ix, iy, iz);
    if (insideSlidingWindow(id_g, c_cfg.origin_i, c_cfg.half_map_size_i)) {
      const int h =
          globalIndexToHashId(id_g, c_cfg.map_size_i, c_cfg.half_map_size_i);
      if (h >= 0 && h < c_cfg.map_vox_num) {
        atomicAdd(&op_cnt[h], 1);
      }
    }

    // ---- 5. advance to next voxel ----
    if (tMx < tMy) {
      if (tMx < tMz) {
        ix += sx;
        tMx += tDx;
      } else {
        iz += sz;
        tMz += tDz;
      }
    } else {
      if (tMy < tMz) {
        iy += sy;
        tMy += tDy;
      } else {
        iz += sz;
        tMz += tDz;
      }
    }
  }

  // ---- 6. endpoint ----
  // Real measurements: op + hit. Range-clipped rays: op only (miss / free).
  const int3 end_g = make_int3(ex_i, ey_i, ez_i);
  if (insideSlidingWindow(end_g, c_cfg.origin_i, c_cfg.half_map_size_i)) {
    const int h =
        globalIndexToHashId(end_g, c_cfg.map_size_i, c_cfg.half_map_size_i);
    if (h >= 0 && h < c_cfg.map_vox_num) {
      atomicAdd(&op_cnt[h], 1);
      if (!clipped) {
        atomicAdd(&hit_cnt[h], 1);
      }
    }
  }
}

// ─────────────────────────────────────────────────
// applyUpdateKernel
//
// Buffer stores logits relative to l_unknown (logit(0.5) == 0): occ[h] unknown
// when occ[h] == 0. Decode before hit/miss, clamp true logit to [l_min, l_max],
// re-encode.
//
// One thread per voxel. Untouched voxels (op_cnt == 0) early-out.
//
// Log-odds: l_miss counts pure traversals (op - hit); l_hit counts endpoints.
//
// After updating, counters zero for the next raycast pass.
// ─────────────────────────────────────────────────
__global__ void applyUpdateKernel(
    float *__restrict__ occ, int *__restrict__ op_cnt,
    int *__restrict__ hit_cnt, const int voxel_num, const float l_hit,
    const float l_miss, const float l_min, const float l_max,
    const float l_unknown, unsigned int *__restrict__ modified_count,
    int *__restrict__ modified_idx, float *__restrict__ modified_val,
    const unsigned int modified_capacity) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= voxel_num) {
    return;
  }

  const int op = op_cnt[h];
  if (op == 0) {
    return;
  }

  const int hit = hit_cnt[h];
  const int miss_ops = op > hit ? op - hit : 0;
  const float stored0 = occ[h];
  float logit = stored0 + l_unknown;
  logit += l_miss * static_cast<float>(miss_ops);
  logit += l_hit * static_cast<float>(hit);
  if (logit < l_min) {
    logit = l_min;
  }
  if (logit > l_max) {
    logit = l_max;
  }

  const float stored = logit - l_unknown;
  occ[h] = stored;
  op_cnt[h] = 0;
  hit_cnt[h] = 0;

  if (modified_count != nullptr && modified_idx != nullptr &&
      modified_val != nullptr && stored != stored0) {
    const unsigned int slot = atomicAdd(modified_count, 1u);
    if (slot < modified_capacity) {
      modified_idx[slot] = h;
      modified_val[slot] = stored;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────
// Host-side launcher (matches the declaration in kernels.hpp).
//
// We push `cfg` into __constant__ memory on the same stream as the
// kernel launch so the H2D copy and the kernel are ordered correctly
// even when the caller uses a non-default stream.
// ─────────────────────────────────────────────────────────────────────
void launchRayCastUpdate(const RayCastCfg &cfg, const float *d_cloud_x,
                         const float *d_cloud_y, const float *d_cloud_z,
                         const int cloud_size, const cudaStream_t stream,
                         int *d_op_cnt, int *d_hit_cnt) {
  if (cloud_size <= 0) {
    return;
  }

  // Allocate the constant memory on the GPU for the configuration.
  cudaMemcpyToSymbolAsync(c_cfg, &cfg, sizeof(RayCastCfg),
                          /*offset=*/0, cudaMemcpyHostToDevice, stream);

  constexpr int kBlock = 512;
  const int grid = (cloud_size + kBlock - 1) / kBlock;
  rayCastUpdateKernel<<<grid, kBlock, /*shared_mem_size=*/0, stream>>>(
      d_cloud_x, d_cloud_y, d_cloud_z, cloud_size, d_op_cnt, d_hit_cnt);
}

void launchApplyUpdate(float *d_occ, int *d_op_cnt, int *d_hit_cnt,
                       const int voxel_num, const float l_hit,
                       const float l_miss, const float l_min, const float l_max,
                       const float l_unknown, const cudaStream_t stream,
                       unsigned int *d_modified_count, int *d_modified_idx,
                       float *d_modified_val,
                       const unsigned int modified_capacity) {
  if (voxel_num <= 0) {
    return;
  }

  constexpr int kBlock = 512;
  const int grid = (voxel_num + kBlock - 1) / kBlock;

  unsigned int *dc = nullptr;
  int *di = nullptr;
  float *dv = nullptr;
  unsigned int cap = 0;
  if (d_modified_count != nullptr && d_modified_idx != nullptr &&
      d_modified_val != nullptr &&
      static_cast<unsigned int>(voxel_num) <= modified_capacity) {
    dc = d_modified_count;
    di = d_modified_idx;
    dv = d_modified_val;
    cap = modified_capacity;
  }

  applyUpdateKernel<<<grid, kBlock, /*shared_mem_size=*/0, stream>>>(
      d_occ, d_op_cnt, d_hit_cnt, voxel_num, l_hit, l_miss, l_min, l_max,
      l_unknown, dc, di, dv, cap);
}

// ─────────────────────────────────────────────────────────────────────
// clearVoxelsByIndexKernel — map sliding: reset many scattered voxels in
// one launch.
// ─────────────────────────────────────────────────────────────────────
__global__ void clearVoxelsByIndexKernel(float *__restrict__ occ,
                                         const int *__restrict__ indices,
                                         const int n, const int voxel_num) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  const int h = indices[tid];
  if (h >= 0 && h < voxel_num) {
    occ[h] = 0.0f;
  }
}

void launchClearVoxelsByIndex(float *d_occ, const int *d_indices, const int n,
                              const int voxel_num, const cudaStream_t stream) {
  if (n <= 0 || d_occ == nullptr || d_indices == nullptr || voxel_num <= 0) {
    return;
  }

  constexpr int kBlock = 512;
  const int grid_dim = (n + kBlock - 1) / kBlock;
  clearVoxelsByIndexKernel<<<grid_dim, kBlock, /*shared_mem_size=*/0, stream>>>(
      d_occ, d_indices, n, voxel_num);
}

// ─────────────────────────────────────────────────────────────────────
// Recenter — plan exiting + entering slabs on device, then clear on device.
// ─────────────────────────────────────────────────────────────────────
__constant__ RecenterCfg c_recenter_cfg;

__device__ inline int normalizeLocalSlice(const int x, const int a,
                                          const int b) {
  const int range = b - a + 1;
  int y = (x - a) % range;
  if (y < 0) {
    y += range;
  }
  return y + a;
}

__global__ void planRecenterKernel(int *__restrict__ d_slices_x,
                                   int *__restrict__ d_slices_y,
                                   int *__restrict__ d_slices_z,
                                   RecenterResult *__restrict__ result) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }

  result->action = 0;
  result->n_slices[0] = 0;
  result->n_slices[1] = 0;
  result->n_slices[2] = 0;

  const float dx = c_recenter_cfg.pos.x - c_recenter_cfg.origin_f.x;
  const float dy = c_recenter_cfg.pos.y - c_recenter_cfg.origin_f.y;
  const float dz = c_recenter_cfg.pos.z - c_recenter_cfg.origin_f.z;
  if (dx * dx + dy * dy + dz * dz < c_recenter_cfg.recenter_threshold_sq) {
    return;
  }

  int3 new_origin_i;
  posToGlobalIndex(c_recenter_cfg.pos, c_recenter_cfg.inv_resolution,
                   c_recenter_cfg.origin_at_center, new_origin_i);

  const int3 shift = new_origin_i - c_recenter_cfg.origin_i;
  const int max_shift = max(abs(shift.x), max(abs(shift.y), abs(shift.z)));
  const int max_map =
      max(c_recenter_cfg.map_size_i.x,
          max(c_recenter_cfg.map_size_i.y, c_recenter_cfg.map_size_i.z));

  result->new_origin_i = new_origin_i;
  if (max_shift >= max_map) {
    result->action = 2;
    return;
  }

  result->action = 1;

  const int3 hs = c_recenter_cfg.half_map_size_i;
  const int3 ms = c_recenter_cfg.map_size_i;
  const int3 origin_i = c_recenter_cfg.origin_i;

  for (int axis = 0; axis < 3; ++axis) {
    const int shift_a = axis == 0 ? shift.x : (axis == 1 ? shift.y : shift.z);
    if (shift_a == 0) {
      continue;
    }

    const int half_a = axis == 0 ? hs.x : (axis == 1 ? hs.y : hs.z);
    const int map_a = axis == 0 ? ms.x : (axis == 1 ? ms.y : ms.z);
    const int origin_a =
        axis == 0 ? origin_i.x : (axis == 1 ? origin_i.y : origin_i.z);

    const int min_id_g = -half_a + origin_a;
    const int min_id_l = min_id_g % map_a;

    int *out_slices =
        axis == 0 ? d_slices_x : (axis == 1 ? d_slices_y : d_slices_z);
    int &n_out = axis == 0
                     ? result->n_slices[0]
                     : (axis == 1 ? result->n_slices[1] : result->n_slices[2]);

    if (shift_a > 0) {
      // Exiting low-side slabs (fall off as origin moves forward).
      for (int k = 0; k < shift_a; ++k) {
        out_slices[n_out++] =
            normalizeLocalSlice(min_id_l + k, -half_a, half_a);
      }
      // Entering high-side slabs: new world coords, clear stale hash data.
      for (int k = 0; k < shift_a; ++k) {
        out_slices[n_out++] = normalizeLocalSlice(half_a - k, -half_a, half_a);
      }
    } else {
      const int n_enter = -shift_a;
      // Exiting high-side slabs.
      for (int k = -1; k >= shift_a; --k) {
        out_slices[n_out++] =
            normalizeLocalSlice(min_id_l + k, -half_a, half_a);
      }
      // Entering low-side slabs.
      for (int k = 0; k < n_enter; ++k) {
        out_slices[n_out++] = normalizeLocalSlice(-half_a + k, -half_a, half_a);
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────
// Recenter slab clear — one thread per row along fastest (z) or strided
// (y) index so inner loops touch contiguous / regular d_occ[] addresses.
// ─────────────────────────────────────────────────────────────────────
__global__ void clearRecenterSlabAxis0Kernel(float *__restrict__ occ,
                                             int *__restrict__ op_cnt,
                                             int *__restrict__ hit_cnt,
                                             const int3 &ms, const int3 &hs,
                                             const int *__restrict__ d_vals,
                                             const int n_slices) {
  const int ny = ms.y;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_rows = n_slices * ny;
  if (tid >= n_rows) {
    return;
  }

  const int nz = ms.z;
  const int si = tid / ny;
  const int iy_raw = tid % ny;
  const int lx = d_vals[si];
  const int ly = iy_raw - hs.y;
  const int3 il0 = make_int3(lx, ly, -hs.z);
  int h = localIndexToHashId(il0, ms, hs);
  for (int k = 0; k < nz; ++k) {
    occ[h] = 0.0f;
    if (op_cnt != nullptr) {
      op_cnt[h] = 0;
    }
    if (hit_cnt != nullptr) {
      hit_cnt[h] = 0;
    }
    ++h;
  }
}

__global__ void clearRecenterSlabAxis1Kernel(float *__restrict__ occ,
                                             int *__restrict__ op_cnt,
                                             int *__restrict__ hit_cnt,
                                             const int3 &ms, const int3 &hs,
                                             const int *__restrict__ d_vals,
                                             const int n_slices) {
  const int nx = ms.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_rows = n_slices * nx;
  if (tid >= n_rows) {
    return;
  }

  const int nz = ms.z;
  const int si = tid / nx;
  const int ix_raw = tid % nx;
  const int lx = ix_raw - hs.x;
  const int ly = d_vals[si];
  const int3 il0 = make_int3(lx, ly, -hs.z);
  int h = localIndexToHashId(il0, ms, hs);
  for (int k = 0; k < nz; ++k) {
    occ[h] = 0.0f;
    if (op_cnt != nullptr) {
      op_cnt[h] = 0;
    }
    if (hit_cnt != nullptr) {
      hit_cnt[h] = 0;
    }
    ++h;
  }
}

__global__ void clearRecenterSlabAxis2Kernel(float *__restrict__ occ,
                                             int *__restrict__ op_cnt,
                                             int *__restrict__ hit_cnt,
                                             const int3 &ms, const int3 &hs,
                                             const int *__restrict__ d_vals,
                                             const int n_slices) {
  const int nx = ms.x;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_rows = n_slices * nx;
  if (tid >= n_rows) {
    return;
  }

  const int ny = ms.y;
  const int nz = ms.z;
  const int si = tid / nx;
  const int ix_raw = tid % nx;
  const int lx = ix_raw - hs.x;
  const int lz = d_vals[si];
  const int3 il0 = make_int3(lx, -hs.y, lz);
  int h = localIndexToHashId(il0, ms, hs);
  for (int k = 0; k < ny; ++k) {
    occ[h] = 0.0f;
    if (op_cnt != nullptr) {
      op_cnt[h] = 0;
    }
    if (hit_cnt != nullptr) {
      hit_cnt[h] = 0;
    }
    h += nz;
  }
}

// ─────────────────────────────────────────────────────────────────────
// fetchOccupancySphereKernel — xy disk per z-slab inside a bounding sphere.
// ─────────────────────────────────────────────────────────────────────
__constant__ OccupancyFetchCfg c_fetch_cfg;

__global__ void fetchOccupancySphereKernel(
    const float *__restrict__ occ, unsigned int *__restrict__ out_count,
    float3 *__restrict__ out_pos, unsigned char *__restrict__ out_state,
    const unsigned int output_capacity, const unsigned int max_results) {
  const int gz = c_fetch_cfg.gz_min + blockIdx.z;
  if (gz > c_fetch_cfg.gz_max) {
    return;
  }

  const int lx = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x) -
                 c_fetch_cfg.disk_radius_vox;
  const int ly = static_cast<int>(threadIdx.y + blockIdx.y * blockDim.y) -
                 c_fetch_cfg.disk_radius_vox;
  const int gx = c_fetch_cfg.center_g.x + lx;
  const int gy = c_fetch_cfg.center_g.y + ly;

  const int3 id_g = make_int3(gx, gy, gz);
  if (!insideSlidingWindow(id_g, c_fetch_cfg.origin_i,
                           c_fetch_cfg.half_map_size_i)) {
    return;
  }

  float3 pos;
  globalIndexToPos(id_g, c_fetch_cfg.resolution, c_fetch_cfg.origin_at_center,
                   pos);

  const float dx = pos.x - c_fetch_cfg.center.x;
  const float dy = pos.y - c_fetch_cfg.center.y;
  const float dz = pos.z - c_fetch_cfg.center.z;
  if (dx * dx + dy * dy + dz * dz > c_fetch_cfg.radius_sq) {
    return;
  }

  const int h = globalIndexToHashId(id_g, c_fetch_cfg.map_size_i,
                                    c_fetch_cfg.half_map_size_i);

  const float stored = occ[h];
  const float logit = stored + c_fetch_cfg.l_unknown;
  unsigned char state = 0;
  if (logit >= c_fetch_cfg.l_occupied) {
    state = 2;
  } else if (logit < c_fetch_cfg.l_free) {
    state = 1;
  }

  if (c_fetch_cfg.state_filter >= 0 &&
      static_cast<int>(state) != c_fetch_cfg.state_filter) {
    return;
  }

  const unsigned int slot = atomicAdd(out_count, 1u);
  if ((max_results > 0u && slot >= max_results) || slot >= output_capacity) {
    return;
  }
  out_pos[slot] = pos;
  out_state[slot] = state;
}

void launchFetchOccupancyAround(const OccupancyFetchCfg &cfg,
                                const float *d_occ, unsigned int *d_out_count,
                                float3 *d_out_pos, unsigned char *d_out_state,
                                const unsigned int output_capacity,
                                const unsigned int max_results,
                                const cudaStream_t stream) {
  if (cfg.disk_radius_vox < 0 || cfg.gz_min > cfg.gz_max || d_occ == nullptr ||
      d_out_count == nullptr || d_out_pos == nullptr ||
      d_out_state == nullptr || output_capacity == 0) {
    return;
  }

  cudaMemcpyToSymbolAsync(c_fetch_cfg, &cfg, sizeof(OccupancyFetchCfg),
                          /*offset=*/0, cudaMemcpyHostToDevice, stream);

  const int disk_diam = 2 * cfg.disk_radius_vox + 1;
  const int nz = cfg.gz_max - cfg.gz_min + 1;
  if (disk_diam <= 0 || nz <= 0) {
    return;
  }

  constexpr int kBlock = 16;
  const dim3 block(kBlock, kBlock, 1);
  const dim3 grid((disk_diam + kBlock - 1) / kBlock,
                  (disk_diam + kBlock - 1) / kBlock, nz);
  fetchOccupancySphereKernel<<<grid, block, /*shared_mem_size=*/0, stream>>>(
      d_occ, d_out_count, d_out_pos, d_out_state, output_capacity, max_results);
}

void launchClearRecenterSlabsForAxis(float *d_occ, int *d_op_cnt,
                                     int *d_hit_cnt, const int3 &map_size_i,
                                     const int3 &half_map_size_i,
                                     const int *d_slice_local_values,
                                     const int n_slices, const int axis,
                                     const cudaStream_t stream) {
  if (n_slices <= 0 || d_occ == nullptr || d_slice_local_values == nullptr) {
    return;
  }

  constexpr int kBlock = 512;
  int n_threads = 0;
  if (axis == 0) {
    n_threads = n_slices * map_size_i.y;
  } else if (axis == 1) {
    n_threads = n_slices * map_size_i.x;
  } else {
    n_threads = n_slices * map_size_i.x;
  }

  const int grid_dim = (n_threads + kBlock - 1) / kBlock;
  if (axis == 0) {
    clearRecenterSlabAxis0Kernel<<<grid_dim, kBlock, 0, stream>>>(
        d_occ, d_op_cnt, d_hit_cnt, map_size_i, half_map_size_i,
        d_slice_local_values, n_slices);
  } else if (axis == 1) {
    clearRecenterSlabAxis1Kernel<<<grid_dim, kBlock, 0, stream>>>(
        d_occ, d_op_cnt, d_hit_cnt, map_size_i, half_map_size_i,
        d_slice_local_values, n_slices);
  } else {
    clearRecenterSlabAxis2Kernel<<<grid_dim, kBlock, 0, stream>>>(
        d_occ, d_op_cnt, d_hit_cnt, map_size_i, half_map_size_i,
        d_slice_local_values, n_slices);
  }
}

void launchRecenter(float *d_occ, int *d_op_cnt, int *d_hit_cnt,
                    const RecenterCfg &cfg, int *d_slices_x, int *d_slices_y,
                    int *d_slices_z, RecenterResult *d_result,
                    RecenterResult *h_result, const cudaStream_t stream) {
  if (d_occ == nullptr || d_slices_x == nullptr || d_slices_y == nullptr ||
      d_slices_z == nullptr || d_result == nullptr || h_result == nullptr) {
    return;
  }

  h_result->action = 0;

  cudaMemcpyToSymbolAsync(c_recenter_cfg, &cfg, sizeof(RecenterCfg),
                          /*offset=*/0, cudaMemcpyHostToDevice, stream);
  planRecenterKernel<<<1, 1, 0, stream>>>(d_slices_x, d_slices_y, d_slices_z,
                                          d_result);
  cudaMemcpyAsync(h_result, d_result, sizeof(RecenterResult),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  if (h_result->action == 0) {
    return;
  }

  if (h_result->action == 2) {
    const size_t n = static_cast<size_t>(cfg.map_vox_num);
    cudaMemsetAsync(d_occ, 0, n * sizeof(float), stream);
    if (d_op_cnt != nullptr) {
      cudaMemsetAsync(d_op_cnt, 0, n * sizeof(int), stream);
    }
    if (d_hit_cnt != nullptr) {
      cudaMemsetAsync(d_hit_cnt, 0, n * sizeof(int), stream);
    }
    cudaStreamSynchronize(stream);
    return;
  }

  const int3 ms = cfg.map_size_i;
  const int3 hs = cfg.half_map_size_i;
  if (h_result->n_slices[0] > 0) {
    launchClearRecenterSlabsForAxis(d_occ, d_op_cnt, d_hit_cnt, ms, hs,
                                    d_slices_x, h_result->n_slices[0], 0,
                                    stream);
  }
  if (h_result->n_slices[1] > 0) {
    launchClearRecenterSlabsForAxis(d_occ, d_op_cnt, d_hit_cnt, ms, hs,
                                    d_slices_y, h_result->n_slices[1], 1,
                                    stream);
  }
  if (h_result->n_slices[2] > 0) {
    launchClearRecenterSlabsForAxis(d_occ, d_op_cnt, d_hit_cnt, ms, hs,
                                    d_slices_z, h_result->n_slices[2], 2,
                                    stream);
  }
  cudaStreamSynchronize(stream);
}

} // namespace cublox