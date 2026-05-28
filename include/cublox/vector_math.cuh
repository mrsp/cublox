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

#include <vector_functions.h>
#include <vector_types.h>

namespace cublox {

__host__ __device__ inline int3 operator+(int3 a, int3 b) {
  return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline int3 operator-(int3 a, int3 b) {
  return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline int3 operator*(int s, int3 a) {
  return make_int3(s * a.x, s * a.y, s * a.z);
}
__host__ __device__ inline int3 operator+(int3 a, int s) {
  return make_int3(a.x + s, a.y + s, a.z + s);
}
__host__ __device__ inline float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline float3 operator*(float3 a, float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}

// Non-zero for all finite x (tie-break at 0 gives +1, matching voxel rounding).
__host__ __device__ inline float signf(float x) {
  return x < 0.0f ? -1.0f : 1.0f;
}

__host__ __device__ inline float3 to_float3(int3 a) {
  return make_float3(static_cast<float>(a.x), static_cast<float>(a.y),
                     static_cast<float>(a.z));
}

} // namespace cublox