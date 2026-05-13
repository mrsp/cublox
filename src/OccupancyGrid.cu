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
// One thread per voxel. Replaces the host-side `update_cache_id_g`
// queue from the original ROG-Map: instead of remembering which voxels
// got touched, we sweep the whole local map and let untouched voxels
// (op_cnt == 0) early-out. The early-out is a single coalesced load +
// branch — essentially free relative to the cost of the raycast pass.
//
// Hit dominates: if any ray ended in this voxel this frame, we apply
// the hit update with `hit` counts (clamped to l_max). Otherwise we
// apply the miss update with `op` counts (clamped to l_min). This
// matches ProbMap::probabilisticMapFromCache in the CPU reference.
//
// After updating, both counters are zeroed so the next frame can run
// raycast atomics into a clean buffer.
// TODO: detect from→to GridType jumps and emit jump events for the
// inflation map kernel (see Tier 1 item 5 in the porting plan).
// ─────────────────────────────────────────────────
__global__ void applyUpdateKernel(float *__restrict__ occ,
                                  int *__restrict__ op_cnt,
                                  int *__restrict__ hit_cnt,
                                  const int voxel_num, const float l_hit,
                                  const float l_miss, const float l_min,
                                  const float l_max,
                                  unsigned int *__restrict__ dirty_count,
                                  int *__restrict__ dirty_idx,
                                  float *__restrict__ dirty_val,
                                  const unsigned int dirty_capacity) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= voxel_num) {
    return;
  }

  const int op = op_cnt[h];
  if (op == 0) {
    return;
  }

  const int hit = hit_cnt[h];
  const float v0 = occ[h];
  float v = v0;

  if (hit > 0) {
    v += l_hit * static_cast<float>(hit);
    if (v > l_max) {
      v = l_max;
    }
  } else {
    // op > 0 and hit == 0 → all `op` operations on this voxel were
    // free-space crossings (misses).
    v += l_miss * static_cast<float>(op);
    if (v < l_min) {
      v = l_min;
    }
  }

  occ[h] = v;
  op_cnt[h] = 0;
  hit_cnt[h] = 0;

  if (dirty_count != nullptr && dirty_idx != nullptr &&
      dirty_val != nullptr && v != v0) {
    const unsigned int slot = atomicAdd(dirty_count, 1u);
    if (slot < dirty_capacity) {
      dirty_idx[slot] = h;
      dirty_val[slot] = v;
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
                         int cloud_size, int *d_op_cnt, int *d_hit_cnt,
                         cudaStream_t stream) {
  if (cloud_size <= 0) {
    return;
  }

  cudaMemcpyToSymbolAsync(c_cfg, &cfg, sizeof(RayCastCfg),
                          /*offset=*/0, cudaMemcpyHostToDevice, stream);

  constexpr int kBlock = 256;
  const int grid = (cloud_size + kBlock - 1) / kBlock;
  rayCastUpdateKernel<<<grid, kBlock, 0, stream>>>(
      d_cloud_x, d_cloud_y, d_cloud_z, cloud_size, d_op_cnt, d_hit_cnt);
}

void launchApplyUpdate(float *d_occ, int *d_op_cnt, int *d_hit_cnt,
                       int voxel_num, float l_hit, float l_miss, float l_min,
                       float l_max, cudaStream_t stream,
                       unsigned int *d_dirty_count, int *d_dirty_idx,
                       float *d_dirty_val, unsigned int dirty_capacity) {
  if (voxel_num <= 0) {
    return;
  }

  constexpr int kBlock = 256;
  const int grid = (voxel_num + kBlock - 1) / kBlock;

  unsigned int *dc = nullptr;
  int *di = nullptr;
  float *dv = nullptr;
  unsigned int cap = 0;
  if (d_dirty_count != nullptr && d_dirty_idx != nullptr &&
      d_dirty_val != nullptr &&
      static_cast<unsigned int>(voxel_num) <= dirty_capacity) {
    dc = d_dirty_count;
    di = d_dirty_idx;
    dv = d_dirty_val;
    cap = dirty_capacity;
  }

  applyUpdateKernel<<<grid, kBlock, 0, stream>>>(
      d_occ, d_op_cnt, d_hit_cnt, voxel_num, l_hit, l_miss, l_min, l_max, dc,
      di, dv, cap);
}

} // namespace cublox