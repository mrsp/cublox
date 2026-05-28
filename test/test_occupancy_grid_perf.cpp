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

#include <cublox/OccupancyGrid.hpp>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <random>
#include <string>

namespace {

cublox::PointCloud makeSphericalShellCloud(int n, float r_min, float r_max,
                                           std::uint32_t seed) {
  cublox::PointCloud cloud(n, 3);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> r_dist(r_min, r_max);
  std::uniform_real_distribution<float> u_dist(-1.0f, 1.0f);
  std::uniform_real_distribution<float> phi_dist(
      0.0f, 2.0f * static_cast<float>(M_PI));
  for (int i = 0; i < n; ++i) {
    const float r = r_dist(rng);
    const float cos_theta = u_dist(rng);
    const float sin_theta =
        std::sqrt(std::max(0.0f, 1.0f - cos_theta * cos_theta));
    const float phi = phi_dist(rng);
    cloud(i, 0) = r * sin_theta * std::cos(phi);
    cloud(i, 1) = r * sin_theta * std::sin(phi);
    cloud(i, 2) = r * cos_theta;
  }
  return cloud;
}

} // namespace

TEST(OccupancyGridPerf, UpdateSphericalShellTiming) {
  cublox::OccupancyGrid grid(
      /*half_map_size_i=*/Eigen::Vector3i(200, 200, 100),
      /*resolution=*/0.05f,
      /*origin_at_center=*/true,
      /*recenter_threshold=*/std::nullopt,
      /*origin=*/Eigen::Vector3f::Zero());

  constexpr int kNumPoints = 500000;
  const auto cloud = makeSphericalShellCloud(kNumPoints, /*r_min=*/1.5f,
                                             /*r_max=*/8.0f, /*seed=*/42);
  const Eigen::Vector3f sensor_origin = Eigen::Vector3f::Zero();

  auto bench_one = [&]() -> double {
    const auto t0 = std::chrono::steady_clock::now();
    grid.update(cloud, sensor_origin);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  };

  std::fprintf(stderr,
               "grid: %d x %d x %d voxels (%d total) at %.2f m resolution\n",
               grid.config_.map_size_i.x(), grid.config_.map_size_i.y(),
               grid.config_.map_size_i.z(), grid.config_.voxel_num,
               static_cast<double>(grid.config_.resolution));
  std::fprintf(stderr, "cloud: %d points on a [%.1f, %.1f] m spherical shell\n",
               kNumPoints, 1.5f, 8.0f);

  const double warm_ms = bench_one();
  std::fprintf(stderr, "warm-up update():     %.3f ms\n", warm_ms);

  constexpr int kN = 100;
  double total = 0.0;
  double min_ms = std::numeric_limits<double>::max();
  double max_ms = 0.0;
  for (int i = 0; i < kN; ++i) {
    const double ms = bench_one();
    total += ms;
    min_ms = std::min(min_ms, ms);
    max_ms = std::max(max_ms, ms);
  }
  const double mean_ms = total / kN;
  std::fprintf(stderr,
               "steady-state over %d iters: mean=%.3f ms, min=%.3f ms, "
               "max=%.3f ms  (~%.1f Hz at mean)\n",
               kN, mean_ms, min_ms, max_ms, 1000.0 / mean_ms);
}
