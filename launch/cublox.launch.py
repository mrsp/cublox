"""Launch cublox with RViz fixed frame taken from cublox_driver.yaml."""

import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_cublox_yaml(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _safe_filename_token(value: str) -> str:
    """Frame ids may contain '/' (e.g. ground_truth/p0); keep temp paths flat."""
    return value.replace("/", "_").replace("\\", "_")


def _materialize_rviz_config(
    pkg_share: str, map_frame: str, tracking_frame: str
) -> str:
    template_path = os.path.join(pkg_share, "cfg.rviz")
    with open(template_path, encoding="utf-8") as handle:
        content = handle.read()
    content = content.replace("@MAP_FRAME@", map_frame)
    content = content.replace("@TRACKING_FRAME@", tracking_frame)

    name = (
        f"cublox_{_safe_filename_token(map_frame)}_"
        f"{_safe_filename_token(tracking_frame)}.rviz"
    )
    out_path = os.path.join(tempfile.gettempdir(), name)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return out_path


def _launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory("cublox")

    config_file = LaunchConfiguration("config_file").perform(context)
    if not config_file:
        config_file = os.path.join(pkg_share, "config", "cublox_driver.yaml")

    cfg = _load_cublox_yaml(config_file)
    map_frame = str(cfg.get("map_frame", "odom"))
    tracking_frame = str(cfg.get("tracking_frame", "base_link"))

    nodes = [
        Node(
            package="cublox",
            executable="cublox_node",
            name="cublox_node",
            output="screen",
            parameters=[{"config_file": config_file}],
        )
    ]

    if LaunchConfiguration("rviz").perform(context).lower() in ("1", "true", "yes"):
        rviz_config = _materialize_rviz_config(
            pkg_share, map_frame, tracking_frame
        )
        nodes.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
            )
        )

    return nodes


def generate_launch_description():
    pkg_share = get_package_share_directory("cublox")
    default_config = os.path.join(pkg_share, "config", "cublox_driver.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Path to cublox_driver.yaml",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Launch RViz with frames from config_file",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
