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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
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
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <tf2_ros/transform_broadcaster.h>

// Hard cap on occupancy voxels fetched / published per frame (viz only).
constexpr int kMaxVizPointsPerFrame = 1000000;
// Hard cap on cloud points processed per frame.
constexpr size_t kMaxCloudPoints = 1000000u;

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

// Drop ~(1 - 1/step) of voxels using global grid coordinates (spatially
// uniform).
inline bool occupancyVizDropBySubsample(const Eigen::Vector3f &pos,
                                        const float resolution_inv,
                                        const bool origin_at_center,
                                        const int step) {
  if (step <= 1) {
    return false;
  }
  Eigen::Vector3i id_g;
  cublox::posToGlobalIndex(pos, resolution_inv, origin_at_center, id_g);
  const unsigned mix = static_cast<unsigned>(id_g.x()) ^
                       static_cast<unsigned>(id_g.y()) ^
                       static_cast<unsigned>(id_g.z());
  return (mix % static_cast<unsigned>(step)) != 0u;
}

struct CubloxConfig {
  Eigen::Isometry3f T_base_to_sensor{Eigen::Isometry3f::Identity()};
  std::string map_frame{"odom"};
  std::string tracking_frame{"base_link"};
  std::optional<std::string> pointcloud_topic;
  std::optional<std::string> depth_topic;
  std::optional<Eigen::Matrix<float, 3, 3>> K;
  std::string odom_topic{"/odom"};
  bool publish_tf{false};
  bool publish_input_cloud{false};
  bool publish_occupancy_cloud{true};
  int viz_subsample{1};
  int viz_max_points{100000};
  Eigen::Vector3i half_map_size{32, 32, 8};
  float resolution{0.05f};
  bool origin_at_center{false};
  std::optional<double> recenter_threshold;
  float max_raycast_range{20.0f};
  float max_viz_radius{20.0f};
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
  if (root["tracking_frame"]) {
    cfg.tracking_frame = root["tracking_frame"].as<std::string>();
  }
  if (root["pointcloud_topic"] && !root["pointcloud_topic"].IsNull()) {
    cfg.pointcloud_topic = root["pointcloud_topic"].as<std::string>();
  }
  if (root["depth_topic"] && !root["depth_topic"].IsNull()) {
    cfg.depth_topic = root["depth_topic"].as<std::string>();
  }
  if (root["K"] && !root["K"].IsNull()) {
    const YAML::Node k_node = root["K"];
    if (!k_node.IsSequence() || k_node.size() != 9) {
      throw std::runtime_error("K must be a sequence of 9 floats");
    }
    Eigen::Matrix<float, 3, 3> K;
    for (int i = 0; i < 9; ++i) {
      K(i / 3, i % 3) = k_node[i].as<float>();
    }
    cfg.K = K;
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
  if (root["max_viz_radius"]) {
    cfg.max_viz_radius = root["max_viz_radius"].as<float>();
  } else {
    cfg.max_viz_radius = cfg.max_raycast_range;
  }
  if (root["publish_input_cloud"]) {
    cfg.publish_input_cloud = root["publish_input_cloud"].as<bool>();
  }
  if (root["publish_occupancy_cloud"]) {
    cfg.publish_occupancy_cloud = root["publish_occupancy_cloud"].as<bool>();
  }
  if (root["publish_tf"]) {
    cfg.publish_tf = root["publish_tf"].as<bool>();
  }
  if (root["viz_subsample"]) {
    cfg.viz_subsample = root["viz_subsample"].as<int>();
    if (cfg.viz_subsample < 1) {
      throw std::runtime_error("viz_subsample must be >= 1");
    }
  }
  if (root["viz_max_points"]) {
    cfg.viz_max_points = root["viz_max_points"].as<int>();
    if (cfg.viz_max_points < 0) {
      throw std::runtime_error("viz_max_points must be >= 0");
    }
  }

  if (!root["base_to_sensor"]) {
    throw std::runtime_error("YAML missing required key 'base_to_sensor' in " +
                             path);
  }
  const YAML::Node tl = root["base_to_sensor"];
  if (!tl["translation"] || !tl["rotation"]) {
    throw std::runtime_error(
        "base_to_sensor must contain 'translation' and 'rotation' in " + path);
  }

  const YAML::Node tr = tl["translation"];
  if (!tr.IsSequence() || tr.size() != 3) {
    throw std::runtime_error("base_to_sensor.translation must be [x, y, z]");
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

  cfg.T_base_to_sensor.setIdentity();
  cfg.T_base_to_sensor.linear() = q.toRotationMatrix();
  cfg.T_base_to_sensor.translation() = trans;

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

void distanceToOccupancyColor(const float d, const float d_max,
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
          "[cublox_node] Could not resolve default config path (install the "
          "package and source the workspace, or set param 'config_file'): " +
          std::string(e.what()));
    }

    const CubloxConfig cublox_cfg = loadConfigFromYaml(config_file);
    map_frame_ = cublox_cfg.map_frame;
    tracking_frame_ = cublox_cfg.tracking_frame;
    T_base_to_sensor_ = cublox_cfg.T_base_to_sensor;
    viz_subsample_ = cublox_cfg.viz_subsample;
    viz_max_points_ = cublox_cfg.viz_max_points;
    max_viz_radius_ = cublox_cfg.max_viz_radius;
    if (viz_max_points_ <= 0 || viz_max_points_ > kMaxVizPointsPerFrame) {
      if (viz_max_points_ > kMaxVizPointsPerFrame) {
        RCLCPP_WARN(get_logger(),
                    "viz_max_points=%d capped to %d for real-time viz",
                    viz_max_points_, kMaxVizPointsPerFrame);
      }
      viz_max_points_ = kMaxVizPointsPerFrame;
    }

    grid_ = std::make_unique<cublox::OccupancyGrid>(
        cublox_cfg.half_map_size, cublox_cfg.resolution,
        cublox_cfg.origin_at_center, cublox_cfg.recenter_threshold,
        Eigen::Vector3f::Zero());

    grid_->setMaxRaycastRange(cublox_cfg.max_raycast_range);
    grid_->setLogOddsParams(cublox_cfg.l_hit, cublox_cfg.l_miss,
                            cublox_cfg.l_min, cublox_cfg.l_max,
                            cublox_cfg.l_free, cublox_cfg.l_occupied);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        cublox_cfg.odom_topic, rclcpp::QoS(10),
        std::bind(&CubloxDriver::odomCallback, this, std::placeholders::_1));

    if (cublox_cfg.pointcloud_topic.has_value()) {
      cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
          cublox_cfg.pointcloud_topic.value(), rclcpp::SensorDataQoS(),
          std::bind(&CubloxDriver::pointCloudCallback, this,
                    std::placeholders::_1));
    } else if (cublox_cfg.depth_topic.has_value() && cublox_cfg.K.has_value()) {
      depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
          cublox_cfg.depth_topic.value(), rclcpp::SensorDataQoS(),
          std::bind(&CubloxDriver::depthCallback, this, std::placeholders::_1));
      K_ = cublox_cfg.K.value();
    } else {
      throw std::runtime_error(
          "[cublox_node] pointcloud_topic or depth_topic and K must be set");
    }

    // Match share/cfg.rviz: Reliable + Transient Local, depth 1.
    rclcpp::QoS viz_qos(rclcpp::KeepLast(1));
    viz_qos.reliable();
    viz_qos.transient_local();
    if (cublox_cfg.publish_input_cloud) {
      input_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
          "/cublox/input_cloud", viz_qos);
    }
    if (cublox_cfg.publish_occupancy_cloud) {
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

    if (cublox_cfg.publish_tf) {
      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

    timer_ = this->create_wall_timer(std::chrono::milliseconds(100),
                                     std::bind(&CubloxDriver::run, this));

    rclcpp::on_shutdown([this]() { requestStop(); });
  }

  ~CubloxDriver() override { requestStop(); }

  void requestStop() {
    if (stopped_.exchange(true)) {
      return;
    }
    shutdown_ = true;
    if (timer_) {
      timer_->cancel();
      timer_.reset();
    }
  }

  void stop() {
    requestStop();
    for (int i = 0;
         i < 100 && publish_inflight_.load(std::memory_order_acquire) > 0;
         ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
      cloud = cloud_;
      odom = odom_;
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
    const Eigen::Isometry3f T_odom_to_sensor =
        T_odom_to_base * T_base_to_sensor_;

    if (!buildCloudInOdomFrame(
            *cloud, T_odom_to_sensor,
            grid_->getMaxRaycastRange() * grid_->getMaxRaycastRange(), pts)) {
      return;
    }

    const Eigen::Vector3f robot_pos = T_odom_to_base.translation();
    const Eigen::Vector3f sensor_origin = T_odom_to_sensor.translation();
    const rclcpp::Time cloud_stamp(cloud->header.stamp);
    const bool new_cloud = !last_mapped_cloud_stamp_valid_ ||
                           cloud_stamp != last_mapped_cloud_stamp_;
    std::vector<cublox::OccupancyGrid::OccupancySample> occ_samples;

    {
      std::lock_guard<std::mutex> grid_lock(grid_mutex_);
      if (shutdown_) {
        return;
      }
      const auto t0 = std::chrono::steady_clock::now();
      if (new_cloud) {
        grid_->update(pts, sensor_origin);
        last_mapped_cloud_stamp_ = cloud_stamp;
        last_mapped_cloud_stamp_valid_ = true;
      }
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

      if (!shutdown_ && occupancy_cloud_pub_) {
        const float fetch_r = std::min(
            max_viz_radius_, grid_->maxHorizontalInWindowRadius(robot_pos));
        const auto t_fetch0 = std::chrono::steady_clock::now();
        occ_samples = grid_->fetchOccupancyAround(
            robot_pos, fetch_r, cublox::OccupancyGrid::VoxelState::OCCUPIED,
            viz_max_points_);
        const auto t_fetch1 = std::chrono::steady_clock::now();
        const double fetch_ms =
            std::chrono::duration<double, std::milli>(t_fetch1 - t_fetch0)
                .count();
        RCLCPP_INFO(get_logger(), "fetch occupancy: %.3f ms (%zu voxels)",
                    fetch_ms, occ_samples.size());
      }
    }

    if (shutdown_) {
      return;
    }

    publishOutputs(robot_pos, odom, cloud_stamp, new_cloud, std::move(pts),
                   std::move(occ_samples));
  }

private:
  void publishOutputs(
      const Eigen::Vector3f &robot_pos,
      const nav_msgs::msg::Odometry::SharedPtr &odom,
      const rclcpp::Time &cloud_stamp, const bool new_cloud,
      cublox::PointCloud pts,
      std::vector<cublox::OccupancyGrid::OccupancySample> occ_samples) {
    nav_msgs::msg::Odometry odom_out = *odom;
    odom_out.header.frame_id = map_frame_;
    odom_out.child_frame_id = tracking_frame_;

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

    odom_pub_->publish(odom_out);
    publishTrackingTf(odom_out);

    nav_msgs::msg::Path path_out = odom_path_;
    path_out.header.frame_id = map_frame_;
    path_out.header.stamp = odom_out.header.stamp;
    path_pub_->publish(path_out);

    if (!shutdown_ && input_cloud_pub_ && new_cloud && pts.rows() > 0) {
      publishInputCloud(cloud_stamp, std::move(pts));
    }

    if (!shutdown_ && occupancy_cloud_pub_ && !occ_samples.empty()) {
      publishOccupancyCloudAsync(robot_pos, std::move(occ_samples));
    }
  }
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> data_lock(data_mutex_);
    odom_ = msg;
  }

  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    std::lock_guard<std::mutex> data_lock(data_mutex_);
    cloud_ = msg;
  }

  void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    const float fx = K_(0, 0);
    const float fy = K_(1, 1);
    const float cx = K_(0, 2);
    const float cy = K_(1, 2);

    const uint32_t width = msg->width;
    const uint32_t height = msg->height;
    if (width == 0 || height == 0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "Depth image has no width or height");
      return;
    }

    const bool is_16bit_depth =
        (msg->encoding == "16UC1" || msg->encoding == "mono16");
    const bool is_32fc1 = (msg->encoding == "32FC1");
    if (!is_16bit_depth && !is_32fc1) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "Unsupported depth encoding '%s' (expected 16UC1, "
                           "mono16, or 32FC1)",
                           msg->encoding.c_str());
      return;
    }

    sensor_msgs::msg::PointCloud2 cloud_msg;
    cloud_msg.header = msg->header;
    cloud_msg.height = 1;
    cloud_msg.is_dense = false;
    sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    modifier.resize(n);

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");

    const auto nan = std::numeric_limits<float>::quiet_NaN();
    if (is_16bit_depth) {
      for (uint32_t v = 0; v < height; ++v) {
        const auto *row = reinterpret_cast<const uint16_t *>(
            msg->data.data() + static_cast<size_t>(v) * msg->step);
        for (uint32_t u = 0; u < width; ++u, ++iter_x, ++iter_y, ++iter_z) {
          const uint16_t raw = row[u];
          if (raw == 0) {
            *iter_x = nan;
            *iter_y = nan;
            *iter_z = nan;
            continue;
          }
          const float z = static_cast<float>(raw) * 0.001f;
          *iter_x = (static_cast<float>(u) - cx) * z / fx;
          *iter_y = (static_cast<float>(v) - cy) * z / fy;
          *iter_z = z;
        }
      }
    } else {
      for (uint32_t v = 0; v < height; ++v) {
        const auto *row = reinterpret_cast<const float *>(
            msg->data.data() + static_cast<size_t>(v) * msg->step);
        for (uint32_t u = 0; u < width; ++u, ++iter_x, ++iter_y, ++iter_z) {
          const float z = row[u];
          if (!std::isfinite(z) || z <= 0.f) {
            *iter_x = nan;
            *iter_y = nan;
            *iter_z = nan;
            continue;
          }
          *iter_x = (static_cast<float>(u) - cx) * z / fx;
          *iter_y = (static_cast<float>(v) - cy) * z / fy;
          *iter_z = z;
        }
      }
    }

    cloud_msg.width = static_cast<uint32_t>(n);
    std::lock_guard<std::mutex> data_lock(data_mutex_);
    cloud_ =
        std::make_shared<sensor_msgs::msg::PointCloud2>(std::move(cloud_msg));
  }

  bool buildCloudInOdomFrame(const sensor_msgs::msg::PointCloud2 &cloud,
                             const Eigen::Isometry3f &T_odom_to_lidar,
                             const float max_range_squared,
                             cublox::PointCloud &out) {
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
    if (n > kMaxCloudPoints) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "PointCloud2 has %zu points; capping at %zu.", n,
                           kMaxCloudPoints);
    }
    const size_t n_cap = std::min(n, kMaxCloudPoints);

    out.resize(static_cast<Eigen::Index>(n_cap), 3);
    sensor_msgs::PointCloud2ConstIterator<float> ix(cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iy(cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iz(cloud, "z");

    Eigen::Index row = 0;
    for (size_t i = 0; i < n_cap; ++i, ++ix, ++iy, ++iz) {
      const float lx = *ix;
      const float ly = *iy;
      const float lz = *iz;
      if (!std::isfinite(lx) || !std::isfinite(ly) || !std::isfinite(lz)) {
        continue;
      }

      const Eigen::Vector3f p_lidar(lx, ly, lz);
      if (p_lidar.squaredNorm() > max_range_squared) {
        continue;
      }

      const Eigen::Vector3f p_odom = T_odom_to_lidar * p_lidar;
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

  void publishTrackingTf(const nav_msgs::msg::Odometry &odom) {
    if (shutdown_ || !tf_broadcaster_) {
      return;
    }

    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = odom.header.stamp;
    tf.header.frame_id = map_frame_;
    tf.child_frame_id = tracking_frame_;
    tf.transform.translation.x = odom.pose.pose.position.x;
    tf.transform.translation.y = odom.pose.pose.position.y;
    tf.transform.translation.z = odom.pose.pose.position.z;
    tf.transform.rotation = odom.pose.pose.orientation;
    tf_broadcaster_->sendTransform(tf);
  }

  void publishInputCloud(const rclcpp::Time &stamp, cublox::PointCloud pts) {
    if (shutdown_ || !input_cloud_pub_) {
      return;
    }

    sensor_msgs::msg::PointCloud2 cloud_msg;
    cloud_msg.header.frame_id = map_frame_;
    cloud_msg.header.stamp = stamp;
    cloud_msg.height = 1;
    cloud_msg.is_dense = true;
    sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    const size_t n = static_cast<size_t>(pts.rows());
    modifier.resize(n);

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
    for (Eigen::Index i = 0; i < pts.rows();
         ++i, ++iter_x, ++iter_y, ++iter_z) {
      *iter_x = pts(i, 0);
      *iter_y = pts(i, 1);
      *iter_z = pts(i, 2);
    }
    cloud_msg.width = static_cast<uint32_t>(n);

    input_cloud_pub_->publish(cloud_msg);
  }

  void publishOccupancyCloudAsync(
      const Eigen::Vector3f &latest_pos,
      std::vector<cublox::OccupancyGrid::OccupancySample> samples) {
    if (shutdown_ || !occupancy_cloud_pub_) {
      return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const float ray_r = max_viz_radius_;
    const int step = viz_subsample_;
    const size_t max_pts = static_cast<size_t>(viz_max_points_);

    sensor_msgs::msg::PointCloud2 cloud_msg;
    cloud_msg.header.frame_id = map_frame_;
    cloud_msg.header.stamp = now();
    cloud_msg.height = 1;
    cloud_msg.is_dense = false;
    sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
    modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    modifier.resize(max_pts);
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_rgb(cloud_msg, "rgb");

    size_t written = 0;
    for (size_t i = 0; i < samples.size() && written < max_pts; ++i) {
      if (shutdown_) {
        return;
      }
      const auto &sample = samples[i];
      if (occupancyVizDropBySubsample(sample.position,
                                      grid_->getResolutionInv(),
                                      grid_->getOriginAtCenter(), step)) {
        continue;
      }
      std_msgs::msg::ColorRGBA col;
      distanceToOccupancyColor((sample.position - latest_pos).norm(), ray_r,
                               col);
      *iter_x = sample.position.x();
      *iter_y = sample.position.y();
      *iter_z = sample.position.z();
      *iter_rgb = packRGBFloat(col);
      ++iter_x;
      ++iter_y;
      ++iter_z;
      ++iter_rgb;
      ++written;
    }

    if (written == 0 || shutdown_) {
      return;
    }
    cloud_msg.width = static_cast<uint32_t>(written);
    modifier.resize(written);

    const double build_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0)
                                .count();
    RCLCPP_INFO(get_logger(), "viz cloud build: %.3f ms (%zu / %zu points)",
                build_ms, written, samples.size());

    auto pub = occupancy_cloud_pub_;
    auto self =
        std::static_pointer_cast<CubloxDriver>(this->shared_from_this());
    publish_inflight_.fetch_add(1, std::memory_order_relaxed);
    std::thread([pub, self, msg = std::move(cloud_msg)]() mutable {
      struct InflightGuard {
        CubloxDriver *driver;
        ~InflightGuard() {
          driver->publish_inflight_.fetch_sub(1, std::memory_order_release);
        }
      } guard{self.get()};
      if (!self->shutdown_.load(std::memory_order_acquire)) {
        pub->publish(msg);
      }
    }).detach();
  }

  std::unique_ptr<cublox::OccupancyGrid> grid_;
  Eigen::Isometry3f T_base_to_sensor_{Eigen::Isometry3f::Identity()};

  std::string map_frame_;
  std::string tracking_frame_;
  Eigen::Matrix<float, 3, 3> K_ = Eigen::Matrix<float, 3, 3>::Identity();

  std::mutex data_mutex_;
  std::mutex grid_mutex_;
  sensor_msgs::msg::PointCloud2::SharedPtr cloud_;
  nav_msgs::msg::Odometry::SharedPtr odom_;

  rclcpp::TimerBase::SharedPtr timer_;

  std::atomic<bool> shutdown_{false};
  std::atomic<bool> stopped_{false};
  std::atomic<int> publish_inflight_{0};

  rclcpp::Time last_mapped_cloud_stamp_{0, 0, RCL_ROS_TIME};
  bool last_mapped_cloud_stamp_valid_{false};

  int viz_subsample_{1};
  int viz_max_points_{0};
  float max_viz_radius_{20.0f};

  // ROS Subscribers
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;

  // ROS Publishers
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr input_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      occupancy_cloud_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  nav_msgs::msg::Path odom_path_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
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
  node->stop();
  node.reset();
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return 0;
}
