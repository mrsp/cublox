#include <cublox/OccupancyGrid.hpp>
#include <cublox/utils.hpp>

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

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

geometry_msgs::msg::Point eigenToPoint(const Eigen::Vector3f &p) {
  geometry_msgs::msg::Point out;
  out.x = p.x();
  out.y = p.y();
  out.z = p.z();
  return out;
}

struct CubloxConfig {
  Eigen::Isometry3f T_base_to_lidar{Eigen::Isometry3f::Identity()};
  std::string map_frame{"odom"};
  std::string pointcloud_topic{"/points"};
  std::string odom_topic{"/odom"};
  std::string visualization_topic{"/cublox/occupancy_markers"};
  std::string path_topic{"/cublox/odom_path"};
  Eigen::Vector3i half_map_size{32, 32, 8};
  float resolution{0.05f};
  bool origin_at_center{false};
  std::optional<double> recenter_threshold;
  float max_raycast_range{20.0f};
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
    int hz = 8;
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
    const double r = root["recenter_threshold"].as<double>();
    if (r >= 0.0) {
      cfg.recenter_threshold = r;
    }
  }
  if (root["max_raycast_range"]) {
    cfg.max_raycast_range = root["max_raycast_range"].as<float>();
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
  return cfg;
}

} // namespace

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
    T_base_to_lidar_ = cublox_cfg.T_base_to_lidar;
    map_frame_ = cublox_cfg.map_frame;
    pointcloud_topic_ = cublox_cfg.pointcloud_topic;
    odom_topic_ = cublox_cfg.odom_topic;

    grid_ = std::make_unique<cublox::OccupancyGrid>(
        cublox_cfg.half_map_size, cublox_cfg.resolution,
        cublox_cfg.origin_at_center, cublox_cfg.recenter_threshold,
        Eigen::Vector3f::Zero());

    grid_->setMaxRaycastRange(cublox_cfg.max_raycast_range);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, rclcpp::QoS(10),
        std::bind(&CubloxDriver::odomCallback, this, std::placeholders::_1));

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        pointcloud_topic_, rclcpp::SensorDataQoS(),
        std::bind(&CubloxDriver::pointCloudCallback, this,
                  std::placeholders::_1));

    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "/cublox/occupancy_markers", rclcpp::QoS(1).transient_local());

    path_pub_ = create_publisher<nav_msgs::msg::Path>(
        "/cublox/odom_path", rclcpp::QoS(1).transient_local());

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(
        "/cublox/odom", rclcpp::QoS(1).transient_local());

    RCLCPP_INFO(get_logger(), "Loaded cublox config from %s; map_frame=%s",
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

    if (shutdown_ || !rclcpp::ok()) {
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

    RCLCPP_INFO(get_logger(), "Building cloud in odom frame");
    RCLCPP_INFO(get_logger(), "Cloud size: %ld", pts.rows());

    const Eigen::Vector3f robot_pos(
        static_cast<float>(odom->pose.pose.position.x),
        static_cast<float>(odom->pose.pose.position.y),
        static_cast<float>(odom->pose.pose.position.z));

    {
      std::lock_guard<std::mutex> grid_lock(grid_mutex_);
      grid_->update(pts, robot_pos);
      grid_->recenter(robot_pos);
    }
    RCLCPP_INFO(get_logger(), "Occupancy grid updated");

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
      map_dirty_ = true;
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
    while (!shutdown_ && rclcpp::ok()) {
      std::unique_lock<std::mutex> publish_lock(publish_mutex_);
      publish_cv_.wait(publish_lock, [this] {
        return map_dirty_ || shutdown_ || !rclcpp::ok();
      });
      if (shutdown_ || !rclcpp::ok()) {
        break;
      }
      if (!map_dirty_) {
        continue;
      }
      map_dirty_ = false;
      auto latest_odom = latest_odom_;
      nav_msgs::msg::Path path_out = odom_path_;
      publish_lock.unlock();

      // Publish latest pose consumed
      if (latest_odom) {
        odom_pub_->publish(*latest_odom);
      }

      if (latest_odom) {
        path_out.header.frame_id = map_frame_;
        path_out.header.stamp = latest_odom->header.stamp;
        path_pub_->publish(path_out);
      }

      std::lock_guard<std::mutex> grid_lock(grid_mutex_);
      publishOccupancyMarkersLocked();
    }
  }

  void publishOccupancyMarkersLocked() {
    visualization_msgs::msg::MarkerArray arr;
    visualization_msgs::msg::Marker del;
    del.header.frame_id = map_frame_;
    del.header.stamp = now();
    del.ns = "cublox_occupancy";
    del.id = 0;
    del.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(del);

    visualization_msgs::msg::Marker cubes;
    cubes.header.frame_id = map_frame_;
    cubes.header.stamp = now();
    cubes.ns = "cublox_occupancy";
    cubes.id = 1;
    cubes.type = visualization_msgs::msg::Marker::CUBE_LIST;
    cubes.action = visualization_msgs::msg::Marker::ADD;
    cubes.scale.x = grid_->config_.resolution;
    cubes.scale.y = grid_->config_.resolution;
    cubes.scale.z = grid_->config_.resolution;

    const int nv = grid_->config_.voxel_num;
    std_msgs::msg::ColorRGBA black;
    black.r = 0.0f;
    black.g = 0.0f;
    black.b = 0.0f;
    black.a = 1.0f;
    cubes.points.reserve(static_cast<size_t>(nv));
    cubes.colors.reserve(static_cast<size_t>(nv));
    for (int hid = 0; hid < nv; ++hid) {
      if (!grid_->isOccupied(hid)) {
        continue;
      }

      Eigen::Vector3f pos;
      cublox::hashIdToPos(hid, grid_->config_.map_size_i,
                          grid_->config_.half_map_size_i,
                          grid_->getOriginIndex(), grid_->config_.resolution,
                          grid_->config_.origin_at_center, pos);
      cubes.points.push_back(eigenToPoint(pos));
      cubes.colors.push_back(black);
    }

    arr.markers.push_back(cubes);
    marker_pub_->publish(arr);
  }

  std::unique_ptr<cublox::OccupancyGrid> grid_;
  Eigen::Isometry3f T_base_to_lidar_{Eigen::Isometry3f::Identity()};

  std::string map_frame_;
  std::string pointcloud_topic_;
  std::string odom_topic_;
  std::string viz_topic_;
  std::string path_topic_;

  std::mutex data_mutex_;
  std::mutex grid_mutex_;
  std::mutex publish_mutex_;
  sensor_msgs::msg::PointCloud2::SharedPtr cloud_;
  nav_msgs::msg::Odometry::SharedPtr odom_;

  rclcpp::TimerBase::SharedPtr timer_;
  std::condition_variable publish_cv_;
  bool map_dirty_{false};

  std::atomic<bool> shutdown_{false};
  std::thread publish_thread_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      marker_pub_;
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
