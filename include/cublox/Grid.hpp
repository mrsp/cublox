#pragma once

#include <Eigen/Dense>
#include <optional>
#include <vector>

namespace cublox {

class Grid {
public:
  Grid(const Eigen::Vector3i &half_map_size_i, const float resolution,
       const std::optional<double> recenter_threshold,
       const bool origin_at_center);
  Grid() = default;
  ~Grid();

  virtual void reset() = 0;
  virtual void resetVoxel(const int &hash_id) = 0;
  bool inside(const Eigen::Vector3f &pos) const;
  bool inside(const Eigen::Vector3i &id_g) const;
  void clearVoxelsOutOfGrid(const std::vector<int> &clear_id, const int &i);
  void updateOriginAndBound(const Eigen::Vector3f &new_origin_d,
                            const Eigen::Vector3i &new_origin_i);
  void recenter(const Eigen::Vector3f &pos);

  struct Config {
    float resolution{0.0};
    float resolution_inv{0.0};
    Eigen::Vector3i map_size_i{Eigen::Vector3i::Zero()};
    Eigen::Vector3i half_map_size_i{Eigen::Vector3i::Zero()};
    int voxel_num{0};
    bool origin_at_center{false};
    std::optional<double> recenter_threshold{std::nullopt};
  } config_;

  Eigen::Vector3f origin_d_, bound_min_d_, bound_max_d_;
  Eigen::Vector3i origin_i_, bound_min_i_, bound_max_i_;
};

} // namespace cublox