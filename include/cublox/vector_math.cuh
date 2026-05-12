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

__host__ __device__ inline float signf(float x) {
  return static_cast<float>((x > 0.0f) - (x < 0.0f));
}

__host__ __device__ inline float3 to_float3(int3 a) {
  return make_float3(static_cast<float>(a.x), static_cast<float>(a.y),
                     static_cast<float>(a.z));
}

} // namespace cublox