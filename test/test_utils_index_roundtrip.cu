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
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <random>
#include <string>
#include <vector>

namespace {

struct MapFixture {
  Eigen::Vector3i half_map_size_i;
  Eigen::Vector3i map_size_i;
  Eigen::Vector3i local_map_origin_i;
  float resolution;
  float resolution_inv;
  bool origin_at_center;
};

MapFixture makeFixture(const Eigen::Vector3i &half_map_size_i,
                       const Eigen::Vector3i &local_map_origin_i,
                       const float resolution, const bool origin_at_center) {
  return MapFixture{
      half_map_size_i,    2 * half_map_size_i + Eigen::Vector3i::Ones(),
      local_map_origin_i, resolution,
      1.0f / resolution,  origin_at_center,
  };
}

int3 toInt3(const Eigen::Vector3i &v) { return make_int3(v.x(), v.y(), v.z()); }

float3 toFloat3(const Eigen::Vector3f &v) {
  return make_float3(v.x(), v.y(), v.z());
}

std::vector<MapFixture> sampleFixtures() {
  return {
      makeFixture(Eigen::Vector3i(200, 200, 100), Eigen::Vector3i(17, -42, 8),
                  0.05f, true),
      makeFixture(Eigen::Vector3i(350, 350, 75), Eigen::Vector3i(-120, 55, 0),
                  0.1f, false),
      makeFixture(Eigen::Vector3i(3, 5, 7), Eigen::Vector3i(2, -1, 4), 0.25f,
                  true),
  };
}

} // namespace

class UtilsCuhIndexRoundTrip : public ::testing::TestWithParam<MapFixture> {};

// Host local -> global -> local round trip test
TEST_P(UtilsCuhIndexRoundTrip, HostGlobalLocalGlobal) {
  const auto &fx = GetParam();
  const int3 map_size_i = toInt3(fx.map_size_i);
  const int3 half_map_size_i = toInt3(fx.half_map_size_i);
  const int3 local_map_origin_i = toInt3(fx.local_map_origin_i);

  std::mt19937 rng(42);
  std::uniform_int_distribution<int> local_dist_x(-fx.half_map_size_i.x(),
                                                  fx.half_map_size_i.x());
  std::uniform_int_distribution<int> local_dist_y(-fx.half_map_size_i.y(),
                                                  fx.half_map_size_i.y());
  std::uniform_int_distribution<int> local_dist_z(-fx.half_map_size_i.z(),
                                                  fx.half_map_size_i.z());

  for (int i = 0; i < 500; ++i) {
    const int3 id_l = toInt3(Eigen::Vector3i(
        local_dist_x(rng), local_dist_y(rng), local_dist_z(rng)));
    int3 id_g;
    cublox::localIndexToGlobalIndex(id_l, map_size_i, half_map_size_i,
                                    local_map_origin_i, id_g);

    int3 id_l_back;
    cublox::globalIndexToLocalIndex(id_g, map_size_i, half_map_size_i,
                                    id_l_back);
    EXPECT_EQ(id_l.x, id_l_back.x);
    EXPECT_EQ(id_l.y, id_l_back.y);
    EXPECT_EQ(id_l.z, id_l_back.z);
  }
}

// Host local -> hash -> local round trip test
TEST_P(UtilsCuhIndexRoundTrip, HostLocalHashLocal) {
  const auto &fx = GetParam();
  const int3 map_size_i = toInt3(fx.map_size_i);
  const int3 half_map_size_i = toInt3(fx.half_map_size_i);

  std::mt19937 rng(43);
  std::uniform_int_distribution<int> local_dist_x(-fx.half_map_size_i.x(),
                                                  fx.half_map_size_i.x());
  std::uniform_int_distribution<int> local_dist_y(-fx.half_map_size_i.y(),
                                                  fx.half_map_size_i.y());
  std::uniform_int_distribution<int> local_dist_z(-fx.half_map_size_i.z(),
                                                  fx.half_map_size_i.z());

  for (int i = 0; i < 500; ++i) {
    const int3 id_l = toInt3(Eigen::Vector3i(
        local_dist_x(rng), local_dist_y(rng), local_dist_z(rng)));
    const int hash_id =
        cublox::localIndexToHashId(id_l, map_size_i, half_map_size_i);

    int3 id_l_back;
    cublox::hashIdToLocalIndex(hash_id, map_size_i, half_map_size_i, id_l_back);
    EXPECT_EQ(id_l.x, id_l_back.x);
    EXPECT_EQ(id_l.y, id_l_back.y);
    EXPECT_EQ(id_l.z, id_l_back.z);
  }
}

// Host pos -> global -> pos round trip test
TEST_P(UtilsCuhIndexRoundTrip, HostPosGlobalPos) {
  const auto &fx = GetParam();
  std::mt19937 rng(45);
  std::uniform_real_distribution<float> pos_dist(-20.0f, 20.0f);

  for (int i = 0; i < 500; ++i) {
    const float3 pos =
        toFloat3(Eigen::Vector3f(pos_dist(rng), pos_dist(rng), pos_dist(rng)));
    int3 id_g;
    cublox::posToGlobalIndex(pos, fx.resolution_inv, fx.origin_at_center, id_g);

    float3 pos_back;
    cublox::globalIndexToPos(id_g, fx.resolution, fx.origin_at_center,
                             pos_back);

    int3 id_g_back;
    cublox::posToGlobalIndex(pos_back, fx.resolution_inv, fx.origin_at_center,
                             id_g_back);
    EXPECT_EQ(id_g.x, id_g_back.x);
    EXPECT_EQ(id_g.y, id_g_back.y);
    EXPECT_EQ(id_g.z, id_g_back.z);
  }
}

struct DeviceRoundTripCase {
  int3 id_l;
  int3 map_size_i;
  int3 half_map_size_i;
  int3 local_map_origin_i;
  float resolution;
  float resolution_inv;
  bool origin_at_center;
};

// Device test kernel for round trip tests
__global__ void roundTripKernel(const DeviceRoundTripCase *cases, int n,
                                int *failures) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const DeviceRoundTripCase c = cases[idx];

  int3 id_g;
  cublox::localIndexToGlobalIndex(c.id_l, c.map_size_i, c.half_map_size_i,
                                  c.local_map_origin_i, id_g);
  int3 id_l_back;
  cublox::globalIndexToLocalIndex(id_g, c.map_size_i, c.half_map_size_i,
                                  id_l_back);
  if (id_l_back.x != c.id_l.x || id_l_back.y != c.id_l.y ||
      id_l_back.z != c.id_l.z) {
    atomicAdd(failures, 1);
    return;
  }

  const int hash_id =
      cublox::localIndexToHashId(c.id_l, c.map_size_i, c.half_map_size_i);
  int3 id_l_from_hash;
  cublox::hashIdToLocalIndex(hash_id, c.map_size_i, c.half_map_size_i,
                             id_l_from_hash);
  if (id_l_from_hash.x != c.id_l.x || id_l_from_hash.y != c.id_l.y ||
      id_l_from_hash.z != c.id_l.z) {
    atomicAdd(failures, 1);
    return;
  }

  int3 id_g_from_hash;
  cublox::hashIdToGlobalIndex(hash_id, c.map_size_i, c.half_map_size_i,
                              c.local_map_origin_i, id_g_from_hash);
  if (id_g_from_hash.x != id_g.x || id_g_from_hash.y != id_g.y ||
      id_g_from_hash.z != id_g.z) {
    atomicAdd(failures, 1);
    return;
  }

  float3 pos;
  cublox::localIndexToPos(c.id_l, c.map_size_i, c.half_map_size_i,
                          c.local_map_origin_i, c.resolution,
                          c.origin_at_center, pos);
  int3 id_g_from_pos;
  cublox::posToGlobalIndex(pos, c.resolution_inv, c.origin_at_center,
                           id_g_from_pos);
  if (id_g_from_pos.x != id_g.x || id_g_from_pos.y != id_g.y ||
      id_g_from_pos.z != id_g.z) {
    atomicAdd(failures, 1);
  }
}

TEST_P(UtilsCuhIndexRoundTrip, DeviceRoundTrip) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device available";
  }

  const auto &fx = GetParam();
  std::vector<DeviceRoundTripCase> cases;
  std::mt19937 rng(48);
  std::uniform_int_distribution<int> local_dist_x(-fx.half_map_size_i.x(),
                                                  fx.half_map_size_i.x());
  std::uniform_int_distribution<int> local_dist_y(-fx.half_map_size_i.y(),
                                                  fx.half_map_size_i.y());
  std::uniform_int_distribution<int> local_dist_z(-fx.half_map_size_i.z(),
                                                  fx.half_map_size_i.z());

  for (int i = 0; i < 128; ++i) {
    cases.push_back(DeviceRoundTripCase{
        toInt3(Eigen::Vector3i(local_dist_x(rng), local_dist_y(rng),
                               local_dist_z(rng))),
        toInt3(fx.map_size_i),
        toInt3(fx.half_map_size_i),
        toInt3(fx.local_map_origin_i),
        fx.resolution,
        fx.resolution_inv,
        fx.origin_at_center,
    });
  }

  DeviceRoundTripCase *d_cases = nullptr;
  int *d_failures = nullptr;
  ASSERT_EQ(cudaMalloc(&d_cases, cases.size() * sizeof(DeviceRoundTripCase)),
            cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d_failures, sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_cases, cases.data(),
                       cases.size() * sizeof(DeviceRoundTripCase),
                       cudaMemcpyHostToDevice),
            cudaSuccess);

  int h_failures = 0;
  ASSERT_EQ(
      cudaMemcpy(d_failures, &h_failures, sizeof(int), cudaMemcpyHostToDevice),
      cudaSuccess);

  // Launch the test kernel
  const int block = 512;
  const int grid = static_cast<int>((cases.size() + block - 1) / block);
  roundTripKernel<<<grid, block>>>(d_cases, static_cast<int>(cases.size()),
                                   d_failures);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Copy failures to host
  ASSERT_EQ(
      cudaMemcpy(&h_failures, d_failures, sizeof(int), cudaMemcpyDeviceToHost),
      cudaSuccess);

  cudaFree(d_cases);
  cudaFree(d_failures);

  EXPECT_EQ(h_failures, 0);
}

INSTANTIATE_TEST_SUITE_P(MapConfigs, UtilsCuhIndexRoundTrip,
                         ::testing::ValuesIn(sampleFixtures()),
                         [](const ::testing::TestParamInfo<MapFixture> &info) {
                           const auto &fx = info.param;
                           return (fx.origin_at_center ? "center" : "corner") +
                                  std::to_string(fx.half_map_size_i.x()) + "x" +
                                  std::to_string(fx.half_map_size_i.y()) + "x" +
                                  std::to_string(fx.half_map_size_i.z());
                         });
