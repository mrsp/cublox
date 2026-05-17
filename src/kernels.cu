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

  // ─────────────────────────────────────────────────
  // 3D Fast Voxel Traversal  (Amanatides & Woo, 1987)
  // ─────────────────────────────────────────────────

  // ---- 1. load endpoint and origin (coalesced) ----
  float ex = cloud_x[tid], ey = cloud_y[tid], ez = cloud_z[tid];
  const float ox = c_cfg.origin.x, oy = c_cfg.origin.y, oz = c_cfg.origin.z;

  // ---- 2. clip ray length to max_range ----
  float dx = ex - ox, dy = ey - oy, dz = ez - oz;
  float L = sqrtf(dx * dx + dy * dy + dz * dz);
  if (L < 1e-6f) {
    return;
  }
  if (L > c_cfg.max_range) {
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
  int ix = (int)floorf(ox * c_cfg.inv_resolution);
  int iy = (int)floorf(oy * c_cfg.inv_resolution);
  int iz = (int)floorf(oz * c_cfg.inv_resolution);
  const int ex_i = (int)floorf(ex * c_cfg.inv_resolution);
  const int ey_i = (int)floorf(ey * c_cfg.inv_resolution);
  const int ez_i = (int)floorf(ez * c_cfg.inv_resolution);
  const int sx = (ux > 0) - (ux < 0);
  const int sy = (uy > 0) - (uy < 0);
  const int sz = (uz > 0) - (uz < 0);
  const float tDx = (sx == 0) ? CUDART_INF_F : fabsf(c_cfg.resolution / ux);
  const float tDy = (sy == 0) ? CUDART_INF_F : fabsf(c_cfg.resolution / uy);
  const float tDz = (sz == 0) ? CUDART_INF_F : fabsf(c_cfg.resolution / uz);
  const float bx = (ix + (sx > 0)) * c_cfg.resolution;
  const float by = (iy + (sy > 0)) * c_cfg.resolution;
  const float bz = (iz + (sz > 0)) * c_cfg.resolution;
  float tMx = (sx == 0) ? CUDART_INF_F : (bx - ox) / ux;
  float tMy = (sy == 0) ? CUDART_INF_F : (by - oy) / uy;
  float tMz = (sz == 0) ? CUDART_INF_F : (bz - oz) / uz;

  // ---- 4. traverse: 1 atomic per voxel, period ----
  while (true) {
    if (ix == ex_i && iy == ey_i && iz == ez_i) {
      // Reached the endpoint
      break;
    }
    const int h = globalIndexToHashId(make_int3(ix, iy, iz), c_cfg.map_size_i,
                                      c_cfg.half_map_size_i);
    if (h >= 0 && h < c_cfg.map_vox_num) {
      atomicAdd(&op_cnt[h], 1); // Memory I/O
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

  // ---- 6. endpoint: count as both an op (visited) and a hit ----
  // The op_cnt increment is what makes applyUpdateKernel pick this
  // voxel up; its `op == 0` early-out would otherwise skip endpoints
  // that received hit_cnt only and the hit would be silently dropped.
  const int h = globalIndexToHashId(make_int3(ex_i, ey_i, ez_i),
                                    c_cfg.map_size_i, c_cfg.half_map_size_i);
  if (h >= 0 && h < c_cfg.map_vox_num) {
    atomicAdd(&op_cnt[h], 1);
    atomicAdd(&hit_cnt[h], 1);
  }
}

// ─────────────────────────────────────────────────
// applyUpdateKernel
//
// Buffer stores zero-centered logits: occ[h] == logit - l_unknown (unknown ==
// 0). Decode before hit/miss, clamp true logit to [l_min, l_max], re-encode.
//
// One thread per voxel. Untouched voxels (op_cnt == 0) early-out.
//
// Hit dominates: endpoint hits apply l_hit * hit_count; else miss applies
// l_miss * op_count.
//
// After updating, counters zero for the next raycast pass.
// TODO: detect from→to GridType jumps for inflation map (porting plan).
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
  const float stored0 = occ[h];
  float logit = stored0 + l_unknown;

  if (hit > 0) {
    logit += l_hit * static_cast<float>(hit);
    if (logit > l_max) {
      logit = l_max;
    }
  } else {
    // op > 0 and hit == 0 → all `op` operations on this voxel were
    // free-space crossings (misses).
    logit += l_miss * static_cast<float>(op);
    if (logit < l_min) {
      logit = l_min;
    }
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
// Host-side launcher (matches the declaration in raycast_kernels.hpp).
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

  constexpr int kBlock = 256;
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

  constexpr int kBlock = 256;
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
// one launch (avoids one cudaMemset per voxel).
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

  constexpr int kBlock = 256;
  const int grid_dim = (n + kBlock - 1) / kBlock;
  clearVoxelsByIndexKernel<<<grid_dim, kBlock, /*shared_mem_size=*/0, stream>>>(
      d_occ, d_indices, n, voxel_num);
}

// ─────────────────────────────────────────────────────────────────────
// Recenter slab clear — one thread per row along fastest (z) or strided
// (y) index so inner loops touch contiguous / regular d_occ[] addresses.
// ─────────────────────────────────────────────────────────────────────
__global__ void clearRecenterSlabAxis0Kernel(float *__restrict__ occ, int3 ms,
                                             int3 hs,
                                             const int *__restrict__ d_vals,
                                             int n_slices) {
  const int ny = ms.y;
  const int nz = ms.z;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_rows = n_slices * ny;
  if (tid >= n_rows) {
    return;
  }
  const int si = tid / ny;
  const int iy_raw = tid % ny;
  const int lx = d_vals[si];
  const int ly = iy_raw - hs.y;
  const int3 il0 = make_int3(lx, ly, -hs.z);
  int h = localIndexToHashId(il0, ms, hs);
  for (int k = 0; k < nz; ++k) {
    occ[h] = 0.0f;
    ++h;
  }
}

__global__ void clearRecenterSlabAxis1Kernel(float *__restrict__ occ, int3 ms,
                                             int3 hs,
                                             const int *__restrict__ d_vals,
                                             int n_slices) {
  const int nx = ms.x;
  const int nz = ms.z;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_rows = n_slices * nx;
  if (tid >= n_rows) {
    return;
  }
  const int si = tid / nx;
  const int ix_raw = tid % nx;
  const int lx = ix_raw - hs.x;
  const int ly = d_vals[si];
  const int3 il0 = make_int3(lx, ly, -hs.z);
  int h = localIndexToHashId(il0, ms, hs);
  for (int k = 0; k < nz; ++k) {
    occ[h] = 0.0f;
    ++h;
  }
}

__global__ void clearRecenterSlabAxis2Kernel(float *__restrict__ occ, int3 ms,
                                             int3 hs,
                                             const int *__restrict__ d_vals,
                                             int n_slices) {
  const int nx = ms.x;
  const int ny = ms.y;
  const int nz = ms.z;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_rows = n_slices * nx;
  if (tid >= n_rows) {
    return;
  }
  const int si = tid / nx;
  const int ix_raw = tid % nx;
  const int lx = ix_raw - hs.x;
  const int lz = d_vals[si];
  const int3 il0 = make_int3(lx, -hs.y, lz);
  int h = localIndexToHashId(il0, ms, hs);
  for (int k = 0; k < ny; ++k) {
    occ[h] = 0.0f;
    h += nz;
  }
}

void launchClearRecenterSlabsForAxis(float *d_occ, const int3 &map_size_i,
                                     const int3 &half_map_size_i,
                                     const int *d_slice_local_values,
                                     const int n_slices, const int axis,
                                     const cudaStream_t stream) {
  if (n_slices <= 0 || d_occ == nullptr || d_slice_local_values == nullptr) {
    return;
  }
  constexpr int kBlock = 256;
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
        d_occ, map_size_i, half_map_size_i, d_slice_local_values, n_slices);
  } else if (axis == 1) {
    clearRecenterSlabAxis1Kernel<<<grid_dim, kBlock, 0, stream>>>(
        d_occ, map_size_i, half_map_size_i, d_slice_local_values, n_slices);
  } else {
    clearRecenterSlabAxis2Kernel<<<grid_dim, kBlock, 0, stream>>>(
        d_occ, map_size_i, half_map_size_i, d_slice_local_values, n_slices);
  }
}

} // namespace cublox