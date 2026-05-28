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

#include <cublox/utils.hpp>
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

class UtilsHppIndexRoundTrip : public ::testing::TestWithParam<MapFixture> {};

// Eigen local -> global -> local round trip test
TEST_P(UtilsHppIndexRoundTrip, GlobalLocalGlobal) {
  const auto &fx = GetParam();
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> local_dist_x(-fx.half_map_size_i.x(),
                                                  fx.half_map_size_i.x());
  std::uniform_int_distribution<int> local_dist_y(-fx.half_map_size_i.y(),
                                                  fx.half_map_size_i.y());
  std::uniform_int_distribution<int> local_dist_z(-fx.half_map_size_i.z(),
                                                  fx.half_map_size_i.z());

  for (int i = 0; i < 500; ++i) {
    const Eigen::Vector3i id_l(local_dist_x(rng), local_dist_y(rng),
                               local_dist_z(rng));
    Eigen::Vector3i id_g;
    cublox::localIndexToGlobalIndex(id_l, fx.map_size_i, fx.half_map_size_i,
                                    fx.local_map_origin_i, id_g);

    Eigen::Vector3i id_l_back;
    cublox::globalIndexToLocalIndex(id_g, fx.map_size_i, fx.half_map_size_i,
                                    id_l_back);
    EXPECT_EQ(id_l, id_l_back) << "global=" << id_g.transpose();
  }
}

// Eigen local -> hash -> local round trip test
TEST_P(UtilsHppIndexRoundTrip, LocalHashLocal) {
  const auto &fx = GetParam();
  std::mt19937 rng(43);
  std::uniform_int_distribution<int> local_dist_x(-fx.half_map_size_i.x(),
                                                  fx.half_map_size_i.x());
  std::uniform_int_distribution<int> local_dist_y(-fx.half_map_size_i.y(),
                                                  fx.half_map_size_i.y());
  std::uniform_int_distribution<int> local_dist_z(-fx.half_map_size_i.z(),
                                                  fx.half_map_size_i.z());

  for (int i = 0; i < 500; ++i) {
    const Eigen::Vector3i id_l(local_dist_x(rng), local_dist_y(rng),
                               local_dist_z(rng));
    const int hash_id =
        cublox::localIndexToHashId(id_l, fx.map_size_i, fx.half_map_size_i);

    Eigen::Vector3i id_l_back;
    cublox::hashIdToLocalIndex(hash_id, fx.map_size_i, fx.half_map_size_i,
                               id_l_back);
    EXPECT_EQ(id_l, id_l_back) << "hash_id=" << hash_id;
  }
}

// Eigen local -> global -> hash -> global round trip test
TEST_P(UtilsHppIndexRoundTrip, GlobalHashGlobal) {
  const auto &fx = GetParam();
  std::mt19937 rng(44);
  std::uniform_int_distribution<int> local_dist_x(-fx.half_map_size_i.x(),
                                                  fx.half_map_size_i.x());
  std::uniform_int_distribution<int> local_dist_y(-fx.half_map_size_i.y(),
                                                  fx.half_map_size_i.y());
  std::uniform_int_distribution<int> local_dist_z(-fx.half_map_size_i.z(),
                                                  fx.half_map_size_i.z());

  for (int i = 0; i < 500; ++i) {
    const Eigen::Vector3i id_l(local_dist_x(rng), local_dist_y(rng),
                               local_dist_z(rng));
    Eigen::Vector3i id_g;
    cublox::localIndexToGlobalIndex(id_l, fx.map_size_i, fx.half_map_size_i,
                                    fx.local_map_origin_i, id_g);

    const int hash_id =
        cublox::globalIndexToHashId(id_g, fx.map_size_i, fx.half_map_size_i);

    Eigen::Vector3i id_g_back;
    cublox::hashIdToGlobalIndex(hash_id, fx.map_size_i, fx.half_map_size_i,
                                fx.local_map_origin_i, id_g_back);
    EXPECT_EQ(id_g, id_g_back) << "hash_id=" << hash_id;
  }
}

// Eigen pos -> global -> pos round trip test
TEST_P(UtilsHppIndexRoundTrip, PosGlobalPos) {
  const auto &fx = GetParam();
  std::mt19937 rng(45);
  std::uniform_real_distribution<float> pos_dist(-20.0f, 20.0f);

  for (int i = 0; i < 500; ++i) {
    const Eigen::Vector3f pos(pos_dist(rng), pos_dist(rng), pos_dist(rng));
    Eigen::Vector3i id_g;
    cublox::posToGlobalIndex(pos, fx.resolution_inv, fx.origin_at_center, id_g);

    Eigen::Vector3f pos_back;
    cublox::globalIndexToPos(id_g, fx.resolution, fx.origin_at_center,
                             pos_back);

    Eigen::Vector3i id_g_back;
    cublox::posToGlobalIndex(pos_back, fx.resolution_inv, fx.origin_at_center,
                             id_g_back);
    EXPECT_EQ(id_g, id_g_back) << "pos=" << pos.transpose();
  }
}

// Eigen pos -> hash -> pos round trip test
TEST_P(UtilsHppIndexRoundTrip, PosHashPos) {
  const auto &fx = GetParam();
  std::mt19937 rng(46);
  std::uniform_real_distribution<float> pos_dist(-20.0f, 20.0f);

  for (int i = 0; i < 500; ++i) {
    const Eigen::Vector3f pos(pos_dist(rng), pos_dist(rng), pos_dist(rng));
    const int hash_id =
        cublox::posToHashIndex(pos, fx.map_size_i, fx.half_map_size_i,
                               fx.resolution_inv, fx.origin_at_center);

    Eigen::Vector3f pos_back;
    cublox::hashIdToPos(hash_id, fx.map_size_i, fx.half_map_size_i,
                        fx.local_map_origin_i, fx.resolution,
                        fx.origin_at_center, pos_back);

    const int hash_id_back =
        cublox::posToHashIndex(pos_back, fx.map_size_i, fx.half_map_size_i,
                               fx.resolution_inv, fx.origin_at_center);
    EXPECT_EQ(hash_id, hash_id_back) << "pos=" << pos.transpose();
  }
}

INSTANTIATE_TEST_SUITE_P(MapConfigs, UtilsHppIndexRoundTrip,
                         ::testing::ValuesIn(sampleFixtures()),
                         [](const ::testing::TestParamInfo<MapFixture> &info) {
                           const auto &fx = info.param;
                           return (fx.origin_at_center ? "center" : "corner") +
                                  std::to_string(fx.half_map_size_i.x()) + "x" +
                                  std::to_string(fx.half_map_size_i.y()) + "x" +
                                  std::to_string(fx.half_map_size_i.z());
                         });
