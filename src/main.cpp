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
#include <cublox/utils.hpp>

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/color_rgba.hpp>

namespace {

std::string resolveConfigPath(const std::string &maybe_empty) {
  if (!maybe_empty.empty()) {
    return maybe_empty;
  }
  const std::string share =
      ament_index_cpp::get_package_share_directory("cublox");
  const std::filesystem::path p =
      std::filesystem::path(share) / std::string("config/cublox_driver.yaml");
  return p.string();
}

struct CubloxConfig {
  Eigen::Isometry3f T_base_to_lidar{Eigen::Isometry3f::Identity()};
  std::string map_frame{"odom"};
  std::string pointcloud_topic{"/points"};
  std::string odom_topic{"/odom"};
  bool publish_occupancy_cloud{true};
  int occupancy_viz_subsample{1};
  int occupancy_viz_max_points{100000};
  Eigen::Vector3i half_map_size{32, 32, 8};
  float resolution{0.05f};
  bool origin_at_center{false};
  std::optional<double> recenter_threshold;
  float max_raycast_range{20.0f};
  float l_hit{0.847f};       //  logit(0.70)
  float l_miss{-0.405f};     //  logit(0.40)
  float l_min{-1.992f};      //  logit(0.12)
  float l_max{3.476f};       //  logit(0.97)
  float l_free{-0.0004f};    //  logit(0.499)
  float l_occupied{1.7346f}; //  logit(0.85)
};

CubloxConfig loadConfigFromYaml(const std::string &path) {
  CubloxConfig cfg;
  const YAML::Node root = YAML::LoadFile(path);

  if (root["map_frame"]) {
    cfg.map_frame = root["map_frame"].as<std::string>();
  }
  if (root["pointcloud_topic"]) {
    cfg.pointcloud_topic = root["pointcloud_topic"].as<std::string>();
  }
  if (root["odom_topic"]) {
    cfg.odom_topic = root["odom_topic"].as<std::string>();
  }

  if (root["half_map_size"]) {
    const YAML::Node hm = root["half_map_size"];
    if (hm.IsSequence() && hm.size() == 3) {
      cfg.half_map_size =
          Eigen::Vector3i(hm[0].as<int>(), hm[1].as<int>(), hm[2].as<int>());
    } else {
      throw std::runtime_error("half_map_size must be a sequence of 3 ints");
    }
  } else {
    int hx = 32;
    int hy = 32;
    int hz = 12;
    if (root["half_map_size_x"]) {
      hx = root["half_map_size_x"].as<int>();
    }
    if (root["half_map_size_y"]) {
      hy = root["half_map_size_y"].as<int>();
    }
    if (root["half_map_size_z"]) {
      hz = root["half_map_size_z"].as<int>();
    }
    cfg.half_map_size = Eigen::Vector3i(hx, hy, hz);
  }

  if (root["resolution"]) {
    cfg.resolution = root["resolution"].as<float>();
  }
  if (root["origin_at_center"]) {
    cfg.origin_at_center = root["origin_at_center"].as<bool>();
  }
  if (root["recenter_threshold"] && !root["recenter_threshold"].IsNull()) {
    cfg.recenter_threshold = root["recenter_threshold"].as<double>();
  }
  if (root["max_raycast_range"]) {
    cfg.max_raycast_range = root["max_raycast_range"].as<float>();
  }

  if (root["publish_occupancy_cloud"]) {
    cfg.publish_occupancy_cloud = root["publish_occupancy_cloud"].as<bool>();
  }
  if (root["occupancy_viz_subsample"]) {
    cfg.occupancy_viz_subsample = root["occupancy_viz_subsample"].as<int>();
    if (cfg.occupancy_viz_subsample < 1) {
      throw std::runtime_error("occupancy_viz_subsample must be >= 1");
    }
  }
  if (root["occupancy_viz_max_points"]) {
    cfg.occupancy_viz_max_points = root["occupancy_viz_max_points"].as<int>();
    if (cfg.occupancy_viz_max_points < 0) {
      throw std::runtime_error("occupancy_viz_max_points must be >= 0");
    }
  }

  if (!root["base_to_lidar"]) {
    throw std::runtime_error("YAML missing required key 'base_to_lidar' in " +
                             path);
  }
  const YAML::Node tl = root["base_to_lidar"];
  if (!tl["translation"] || !tl["rotation"]) {
    throw std::runtime_error(
        "base_to_lidar must contain 'translation' and 'rotation' in " + path);
  }

  const YAML::Node tr = tl["translation"];
  if (!tr.IsSequence() || tr.size() != 3) {
    throw std::runtime_error("base_to_lidar.translation must be [x, y, z]");
  }
  Eigen::Vector3f trans(tr[0].as<float>(), tr[1].as<float>(),
                        tr[2].as<float>());

  const YAML::Node rot = tl["rotation"];
  float qx, qy, qz, qw;
  if (rot.IsMap()) {
    qx = rot["x"].as<float>();
    qy = rot["y"].as<float>();
    qz = rot["z"].as<float>();
    qw = rot["w"].as<float>();
  } else if (rot.IsSequence() && rot.size() == 4) {
    qx = rot[0].as<float>();
    qy = rot[1].as<float>();
    qz = rot[2].as<float>();
    qw = rot[3].as<float>();
  } else {
    throw std::runtime_error(
        "base_to_lidar.rotation must be a map {x,y,z,w} or [x,y,z,w] sequence");
  }

  Eigen::Quaternionf q(qw, qx, qy, qz);
  q.normalize();

  cfg.T_base_to_lidar.setIdentity();
  cfg.T_base_to_lidar.linear() = q.toRotationMatrix();
  cfg.T_base_to_lidar.translation() = trans;

  if (root["probabilities"]) {
    const YAML::Node probabilities_params = root["probabilities"];
    if (probabilities_params["hit"]) {
      cfg.l_hit = cublox::logit(probabilities_params["hit"].as<float>());
    }
    if (probabilities_params["miss"]) {
      cfg.l_miss = cublox::logit(probabilities_params["miss"].as<float>());
    }
    if (probabilities_params["min"]) {
      cfg.l_min = cublox::logit(probabilities_params["min"].as<float>());
    }
    if (probabilities_params["max"]) {
      cfg.l_max = cublox::logit(probabilities_params["max"].as<float>());
    }
    if (probabilities_params["free"]) {
      cfg.l_free = cublox::logit(probabilities_params["free"].as<float>());
    }
    if (probabilities_params["occupied"]) {
      cfg.l_occupied =
          cublox::logit(probabilities_params["occupied"].as<float>());
    }
  }
  return cfg;
}

} // namespace

void distanceToOccupancyColor(float d, float d_max,
                              std_msgs::msg::ColorRGBA &c) {
  const float d_max_clamped = std::max(d_max, 1.0f);
  const float t = std::clamp(d / d_max_clamped, 0.0f, 1.0f);
  if (t <= 0.5f) {
    const float k = t * 2.0f;
    c.r = 0.0f;
    c.g = 1.0f - k;
    c.b = k;
  } else {
    const float k = (t - 0.5f) * 2.0f;
    c.r = k;
    c.g = 0.0f;
    c.b = 1.0f - k;
  }
  c.a = 1.0f;
}

float packRGBFloat(const std_msgs::msg::ColorRGBA &col) {
  const uint32_t ur =
      static_cast<uint32_t>(std::clamp(col.r, 0.f, 1.f) * 255.f);
  const uint32_t ug =
      static_cast<uint32_t>(std::clamp(col.g, 0.f, 1.f) * 255.f);
  const uint32_t ub =
      static_cast<uint32_t>(std::clamp(col.b, 0.f, 1.f) * 255.f);
  const uint32_t packed = (ur << 16) | (ug << 8) | ub;
  float rgb_f = 0.f;
  std::memcpy(&rgb_f, &packed, sizeof(float));
  return rgb_f;
}

class CubloxDriver : public rclcpp::Node {
public:
  CubloxDriver() : rclcpp::Node("cublox_node") {
    std::string config_file;
    try {
      config_file =
          resolveConfigPath(declare_parameter<std::string>("config_file", ""));
    } catch (const std::exception &e) {
      throw std::runtime_error(
          "Could not resolve default config path (install the package and "
          "source the workspace, or set param 'config_file'): " +
          std::string(e.what()));
    }

    const CubloxConfig cublox_cfg = loadConfigFromYaml(config_file);
    map_frame_ = cublox_cfg.map_frame;
    pointcloud_topic_ = cublox_cfg.pointcloud_topic;
    odom_topic_ = cublox_cfg.odom_topic;
    T_base_to_lidar_ = cublox_cfg.T_base_to_lidar;

    publish_occupancy_cloud_ = cublox_cfg.publish_occupancy_cloud;
    occupancy_viz_subsample_ = cublox_cfg.occupancy_viz_subsample;
    occupancy_viz_max_points_ = cublox_cfg.occupancy_viz_max_points;

    grid_ = std::make_unique<cublox::OccupancyGrid>(
        cublox_cfg.half_map_size, cublox_cfg.resolution,
        cublox_cfg.origin_at_center, cublox_cfg.recenter_threshold,
        Eigen::Vector3f::Zero());

    grid_->setMaxRaycastRange(cublox_cfg.max_raycast_range);
    grid_->setLogOddsParams(cublox_cfg.l_hit, cublox_cfg.l_miss,
                            cublox_cfg.l_min, cublox_cfg.l_max,
                            cublox_cfg.l_free, cublox_cfg.l_occupied);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, rclcpp::QoS(10),
        std::bind(&CubloxDriver::odomCallback, this, std::placeholders::_1));

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        pointcloud_topic_, rclcpp::SensorDataQoS(),
        std::bind(&CubloxDriver::pointCloudCallback, this,
                  std::placeholders::_1));

    rclcpp::QoS viz_qos(1);
    viz_qos.transient_local();
    if (publish_occupancy_cloud_) {
      occupancy_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
          "/cublox/occupancy_cloud", viz_qos);
    }

    path_pub_ = create_publisher<nav_msgs::msg::Path>(
        "/cublox/odom_path", rclcpp::QoS(1).transient_local());

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(
        "/cublox/odom", rclcpp::QoS(1).transient_local());

    RCLCPP_INFO(get_logger(), "Loaded cublox config from %s; map_frame=%s ",
                config_file.c_str(), map_frame_.c_str());

    odom_path_.header.frame_id = map_frame_;

    timer_ = this->create_wall_timer(std::chrono::milliseconds(1000),
                                     std::bind(&CubloxDriver::run, this));
    publish_thread_ = std::thread(&CubloxDriver::publishLoop, this);
  }

  ~CubloxDriver() override {
    shutdown_ = true;
    publish_cv_.notify_all();
    if (publish_thread_.joinable()) {
      publish_thread_.join();
    }
  }

  void run() {
    if (!rclcpp::ok() || shutdown_) {
      return;
    }

    sensor_msgs::msg::PointCloud2::SharedPtr cloud;
    nav_msgs::msg::Odometry::SharedPtr odom;
    {
      std::lock_guard<std::mutex> data_lock(data_mutex_);
      if (!cloud_ || !odom_) {
        return;
      }
      cloud = std::move(cloud_);
      odom = std::move(odom_);
    }

    cublox::PointCloud pts;
    Eigen::Isometry3f T_odom_to_base = Eigen::Isometry3f::Identity();
    T_odom_to_base.translation() =
        Eigen::Vector3f(static_cast<float>(odom->pose.pose.position.x),
                        static_cast<float>(odom->pose.pose.position.y),
                        static_cast<float>(odom->pose.pose.position.z));
    Eigen::Quaternionf q(
        odom->pose.pose.orientation.w, odom->pose.pose.orientation.x,
        odom->pose.pose.orientation.y, odom->pose.pose.orientation.z);
    T_odom_to_base.linear() = q.toRotationMatrix();
    const Eigen::Isometry3f T_odom_to_lidar = T_odom_to_base * T_base_to_lidar_;

    if (!buildCloudInOdomFrame(*cloud, T_odom_to_lidar,
                               grid_->getMaxRaycastRange(), pts)) {
      return;
    }

    const Eigen::Vector3f robot_pos = T_odom_to_base.translation();
    {
      std::lock_guard<std::mutex> grid_lock(grid_mutex_);
      const auto t0 = std::chrono::steady_clock::now();
      grid_->update(pts, robot_pos);
      const auto t1 = std::chrono::steady_clock::now();
      grid_->recenter(robot_pos);
      const auto t2 = std::chrono::steady_clock::now();
      const double update_ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count();
      const double recenter_ms =
          std::chrono::duration<double, std::milli>(t2 - t1).count();
      RCLCPP_INFO(get_logger(),
                  "grid update: %.3f ms, recenter: %.3f ms (total %.3f ms)",
                  update_ms, recenter_ms, update_ms + recenter_ms);
    }

    {
      std::lock_guard<std::mutex> lk(publish_mutex_);
      latest_odom_ = odom;
      geometry_msgs::msg::PoseStamped ps;
      ps.header = odom->header;
      ps.pose = odom->pose.pose;
      odom_path_.poses.push_back(std::move(ps));
      constexpr size_t kPathCap = 512;
      if (odom_path_.poses.size() > kPathCap) {
        const size_t excess = odom_path_.poses.size() - kPathCap;
        odom_path_.poses.erase(odom_path_.poses.begin(),
                               odom_path_.poses.begin() +
                                   static_cast<std::ptrdiff_t>(excess));
      }
      map_has_data_ = true;
    }
    publish_cv_.notify_one();
  }

private:
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> data_lock(data_mutex_);
    odom_ = msg;
  }

  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    std::lock_guard<std::mutex> data_lock(data_mutex_);
    cloud_ = msg;
  }

  bool buildCloudInOdomFrame(const sensor_msgs::msg::PointCloud2 &cloud,
                             const Eigen::Isometry3f &T_odom_to_lidar,
                             const float max_range, cublox::PointCloud &out) {
    if (cloud.fields.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "PointCloud2 has no fields.");
      return false;
    }

    const size_t n =
        static_cast<size_t>(cloud.width) * static_cast<size_t>(cloud.height);
    if (n == 0) {
      return false;
    }

    out.resize(static_cast<Eigen::Index>(n), 3);
    sensor_msgs::PointCloud2ConstIterator<float> ix(cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iy(cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iz(cloud, "z");

    Eigen::Index row = 0;
    for (size_t i = 0; i < n; ++i, ++ix, ++iy, ++iz) {
      const float x = *ix;
      const float y = *iy;
      const float z = *iz;
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        continue;
      }

      if (std::abs(x) > max_range || std::abs(y) > max_range ||
          std::abs(z) > max_range) {
        continue;
      }

      const Eigen::Vector3f p_odom = T_odom_to_lidar * Eigen::Vector3f(x, y, z);
      out.row(row++) = p_odom.transpose();
    }

    if (row == 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "Point cloud had no finite points after filtering.");
      return false;
    }

    out.conservativeResize(row, 3);
    return true;
  }

  void publishLoop() {
    while (rclcpp::ok() && !shutdown_) {
      std::unique_lock<std::mutex> publish_lock(publish_mutex_);
      publish_cv_.wait(publish_lock, [this] {
        return map_has_data_ || shutdown_ || !rclcpp::ok();
      });
      if (!rclcpp::ok() || shutdown_) {
        break;
      }

      if (!map_has_data_) {
        continue;
      }

      map_has_data_ = false;
      auto latest_odom = latest_odom_;
      nav_msgs::msg::Path path_out = odom_path_;
      publish_lock.unlock();

      if (!latest_odom) {
        continue;
      }

      odom_pub_->publish(*latest_odom);
      path_out.header.frame_id = map_frame_;
      path_out.header.stamp = latest_odom->header.stamp;
      path_pub_->publish(path_out);
      const Eigen::Vector3f latest_pos = Eigen::Vector3f(
          static_cast<float>(latest_odom->pose.pose.position.x),
          static_cast<float>(latest_odom->pose.pose.position.y),
          static_cast<float>(latest_odom->pose.pose.position.z));
      if (publish_occupancy_cloud_ && occupancy_cloud_pub_) {
        std::lock_guard<std::mutex> grid_lock(grid_mutex_);
        publishOccupancyCloud(latest_pos);
      }
    }
  }

  void publishOccupancyCloud(const Eigen::Vector3f &latest_pos) {
    if (!occupancy_cloud_pub_) {
      return;
    }

    const int nv = grid_->config_.voxel_num;
    const int step = occupancy_viz_subsample_;
    const float ray_r = grid_->getMaxRaycastRange();
    const size_t reserve_hint =
        occupancy_viz_max_points_ > 0
            ? static_cast<size_t>(std::min(occupancy_viz_max_points_, nv))
            : std::min(static_cast<size_t>(nv), size_t(500000));
    std::vector<float> interleaved;
    interleaved.reserve(std::max(size_t(4096), reserve_hint) * 4);

    for (int hid = 0; hid < nv; ++hid) {
      if (!grid_->isOccupied(hid)) {
        continue;
      }
      if (step > 1) {
        Eigen::Vector3i id_l;
        cublox::hashIdToLocalIndex(hid, grid_->config_.map_size_i,
                                   grid_->config_.half_map_size_i, id_l);
        const Eigen::Vector3i cell = id_l + grid_->config_.half_map_size_i;
        if ((cell.x() % step) || (cell.y() % step) || (cell.z() % step)) {
          continue;
        }
      }

      Eigen::Vector3f pos;
      cublox::hashIdToPos(hid, grid_->config_.map_size_i,
                          grid_->config_.half_map_size_i,
                          grid_->getOriginIndex(), grid_->config_.resolution,
                          grid_->config_.origin_at_center, pos);
      std_msgs::msg::ColorRGBA col;
      distanceToOccupancyColor((pos - latest_pos).norm(), ray_r, col);
      interleaved.push_back(pos.x());
      interleaved.push_back(pos.y());
      interleaved.push_back(pos.z());
      interleaved.push_back(packRGBFloat(col));
      if (occupancy_viz_max_points_ > 0 &&
          static_cast<int>(interleaved.size() / 4) >=
              occupancy_viz_max_points_) {
        break;
      }
    }

    const size_t npts = interleaved.size() / 4;
    sensor_msgs::msg::PointCloud2 cloud_msg;
    cloud_msg.header.frame_id = map_frame_;
    cloud_msg.header.stamp = now();
    cloud_msg.height = 1;
    cloud_msg.width = static_cast<uint32_t>(npts);
    cloud_msg.is_dense = false;
    sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
    modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    modifier.resize(npts);
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_rgb(cloud_msg, "rgb");
    for (size_t i = 0; i < npts; ++i) {
      const size_t b = i * 4;
      *iter_x = interleaved[b];
      *iter_y = interleaved[b + 1];
      *iter_z = interleaved[b + 2];
      *iter_rgb = interleaved[b + 3];
      ++iter_x;
      ++iter_y;
      ++iter_z;
      ++iter_rgb;
    }
    occupancy_cloud_pub_->publish(cloud_msg);
  }

  std::unique_ptr<cublox::OccupancyGrid> grid_;
  Eigen::Isometry3f T_base_to_lidar_{Eigen::Isometry3f::Identity()};

  std::string map_frame_;
  std::string pointcloud_topic_;
  std::string odom_topic_;

  std::mutex data_mutex_;
  std::mutex grid_mutex_;
  std::mutex publish_mutex_;
  sensor_msgs::msg::PointCloud2::SharedPtr cloud_;
  nav_msgs::msg::Odometry::SharedPtr odom_;

  rclcpp::TimerBase::SharedPtr timer_;
  std::condition_variable publish_cv_;
  bool map_has_data_{false};

  std::atomic<bool> shutdown_{false};
  std::thread publish_thread_;

  bool publish_occupancy_cloud_{true};
  int occupancy_viz_subsample_{1};
  int occupancy_viz_max_points_{0};

  // ROS Subscribers
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;

  // ROS Publishers
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      occupancy_cloud_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  nav_msgs::msg::Path odom_path_;
  nav_msgs::msg::Odometry::SharedPtr latest_odom_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  std::shared_ptr<CubloxDriver> node;
  try {
    node = std::make_shared<CubloxDriver>();
  } catch (const std::exception &e) {
    RCLCPP_ERROR(rclcpp::get_logger("cublox_node"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::spin(node);
  rclcpp::shutdown();
  node.reset();
  return 0;
}
