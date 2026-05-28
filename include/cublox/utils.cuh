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

#include <cmath>
#include <cublox/kernels.hpp>
#include <cublox/vector_math.cuh>

namespace cublox {

// ---- pos -> global index ----------------------------------------------------
__host__ __device__ inline int posToGlobalIndex(const float pos,
                                                const float resolution_inv,
                                                const bool origin_at_center) {
  if (origin_at_center) {
    const float s = pos < 0.0f ? -1.0f : 1.0f;
    return static_cast<int>(pos * resolution_inv + s * 0.5f);
  }
  return static_cast<int>(floorf(pos * resolution_inv));
}

__host__ __device__ inline void posToGlobalIndex(const float3 &pos,
                                                 const float resolution_inv,
                                                 const bool origin_at_center,
                                                 int3 &id_g) {
  id_g = make_int3(posToGlobalIndex(pos.x, resolution_inv, origin_at_center),
                   posToGlobalIndex(pos.y, resolution_inv, origin_at_center),
                   posToGlobalIndex(pos.z, resolution_inv, origin_at_center));
}

// ---- global index -> pos ----------------------------------------------------
__host__ __device__ inline float globalIndexToPos(const int id_g,
                                                  const float resolution,
                                                  const bool origin_at_center) {
  return origin_at_center ? id_g * resolution : (id_g + 0.5f) * resolution;
}

__host__ __device__ inline void globalIndexToPos(const int3 &id_g,
                                                 const float resolution,
                                                 const bool origin_at_center,
                                                 float3 &pos) {
  pos = make_float3(globalIndexToPos(id_g.x, resolution, origin_at_center),
                    globalIndexToPos(id_g.y, resolution, origin_at_center),
                    globalIndexToPos(id_g.z, resolution, origin_at_center));
}

// ---- global -> local index (modular sliding-window mapping) -----------------
__host__ __device__ inline int
globalIndexToLocalIndex(const int id_g, const int map_size_i,
                        const int half_map_size_i) {
  int id_l = id_g % map_size_i;
  if (id_l > half_map_size_i) {
    id_l -= map_size_i;
  } else if (id_l < -half_map_size_i) {
    id_l += map_size_i;
  }
  return id_l;
}

__host__ __device__ inline void
globalIndexToLocalIndex(const int3 &id_g, const int3 &map_size_i,
                        const int3 &half_map_size_i, int3 &id_l) {
  id_l = make_int3(
      globalIndexToLocalIndex(id_g.x, map_size_i.x, half_map_size_i.x),
      globalIndexToLocalIndex(id_g.y, map_size_i.y, half_map_size_i.y),
      globalIndexToLocalIndex(id_g.z, map_size_i.z, half_map_size_i.z));
}

// ---- local -> global index (depends on current sliding origin) --------------
__host__ __device__ inline int
localIndexToGlobalIndex(const int id_l, const int map_size_i,
                        const int half_map_size_i, const int origin_g) {
  const int min_id_g = -half_map_size_i + origin_g;
  int min_id_l = min_id_g % map_size_i;
  min_id_l -= min_id_l > half_map_size_i ? map_size_i : 0;
  min_id_l += min_id_l < -half_map_size_i ? map_size_i : 0;
  int diff_id_l = id_l - min_id_l;
  diff_id_l = diff_id_l < 0 ? (map_size_i + diff_id_l) : diff_id_l;
  return min_id_g + diff_id_l;
}

__host__ __device__ inline void
localIndexToGlobalIndex(const int3 &id_l, const int3 &map_size_i,
                        const int3 &half_map_size_i,
                        const int3 &local_map_origin_i, int3 &id_g) {
  id_g =
      make_int3(localIndexToGlobalIndex(id_l.x, map_size_i.x, half_map_size_i.x,
                                        local_map_origin_i.x),
                localIndexToGlobalIndex(id_l.y, map_size_i.y, half_map_size_i.y,
                                        local_map_origin_i.y),
                localIndexToGlobalIndex(id_l.z, map_size_i.z, half_map_size_i.z,
                                        local_map_origin_i.z));
}

// ---- local index -> pos -----------------------------------------------------
__host__ __device__ inline void
localIndexToPos(const int3 &id_l, const int3 &map_size_i,
                const int3 &half_map_size_i, const int3 &local_map_origin_i,
                const float resolution, const bool origin_at_center,
                float3 &pos) {
  int3 id_g;
  localIndexToGlobalIndex(id_l, map_size_i, half_map_size_i, local_map_origin_i,
                          id_g);
  globalIndexToPos(id_g, resolution, origin_at_center, pos);
}

// ---- hash <-> local index ---------------------------------------------------
__host__ __device__ inline void hashIdToLocalIndex(const int hash_id,
                                                   const int3 &map_size_i,
                                                   const int3 &half_map_size_i,
                                                   int3 &id_l) {
  const int yz = map_size_i.y * map_size_i.z;
  id_l.x = hash_id / yz;
  id_l.y = (hash_id - id_l.x * yz) / map_size_i.z;
  id_l.z = hash_id - id_l.x * yz - id_l.y * map_size_i.z;
  id_l = id_l - half_map_size_i;
}

__host__ __device__ inline int localIndexToHashId(const int3 &id_l,
                                                  const int3 &map_size_i,
                                                  const int3 &half_map_size_i) {
  const int3 id = id_l + half_map_size_i;
  return id.x * map_size_i.y * map_size_i.z + id.y * map_size_i.z + id.z;
}

// ---- composed: hash -> global index / pos -----------------------------------
__host__ __device__ inline void
hashIdToGlobalIndex(const int hash_id, const int3 &map_size_i,
                    const int3 &half_map_size_i, const int3 &local_map_origin_i,
                    int3 &id_g) {
  int3 id_l;
  hashIdToLocalIndex(hash_id, map_size_i, half_map_size_i, id_l);
  localIndexToGlobalIndex(id_l, map_size_i, half_map_size_i, local_map_origin_i,
                          id_g);
}

__host__ __device__ inline void
hashIdToPos(const int hash_id, const int3 &map_size_i,
            const int3 &half_map_size_i, const int3 &local_map_origin_i,
            const float resolution, const bool origin_at_center, float3 &pos) {
  int3 id_l;
  hashIdToLocalIndex(hash_id, map_size_i, half_map_size_i, id_l);
  localIndexToPos(id_l, map_size_i, half_map_size_i, local_map_origin_i,
                  resolution, origin_at_center, pos);
}

// ---- composed: global / pos -> hash -----------------------------------------
__host__ __device__ inline int
globalIndexToHashId(const int3 &id_g, const int3 &map_size_i,
                    const int3 &half_map_size_i) {
  int3 id_l;
  globalIndexToLocalIndex(id_g, map_size_i, half_map_size_i, id_l);
  return localIndexToHashId(id_l, map_size_i, half_map_size_i);
}

__host__ __device__ inline int posToHashIndex(const float3 &pos,
                                              const int3 &map_size_i,
                                              const int3 &half_map_size_i,
                                              const float resolution_inv,
                                              const bool origin_at_center) {
  int3 id_g;
  posToGlobalIndex(pos, resolution_inv, origin_at_center, id_g);
  return globalIndexToHashId(id_g, map_size_i, half_map_size_i);
}

} // namespace cublox
