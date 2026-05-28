# README

## This repository is a work in progress.
Contributions, feedback, or suggestions are highly appreciated as we continue to enhance this project!

## cublox
`cublox` is a standalone C/C++ and CUDA library for real-time 3D occupancy mapping. It maintains an ego-centric voxel grid centered on the robot and updates it incrementally from point clouds. When the robot drifts far enough from the grid origin, the map recenters on the GPU so the robot stays near the center of a fixed-size sliding window.

All heavy work runs on the GPU: raycasting, occupancy updates, and recentering. Only visualization is done on the CPU.

This repository also ships a dummy ROS 2 driver - `cublox_node` - that wires cublox to your robot. It subscribes to a point cloud and odometry, publishes an occupancy cloud and TF, and includes an RViz launch file for live visualization. For better performance we need to synchronize the odometry to the point cloud received and also deskew the point cloud.

**Package contents**

- **`cublox` library** — static C++/CUDA library you can link into your own projects (`include/cublox/`, `src/`)
- **`cublox_node`** — ROS 2 node driven by `config/cublox_driver.yaml` (grid size, resolution, topics, recenter threshold, sensor extrinsics)

**Requirements:** CUDA toolkit, Eigen, yaml-cpp; ROS 2 dependencies for the driver only.

## ROS 2

1. Configure `config/cublox_driver.yaml`

2. In your ROS 2 workspace:

```
colcon build --packages-select cublox --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash 
ros2 launch cublox cublox.launch.py
```

## Demo on 3090 GPU
Indicative execution times for a grid of 73,500,000 voxels at a 0.1m resolution:

<b>Voxel Grid Update: 1.344ms / Voxel Grid Recenter: 4.822 ms (total 6.167 ms)</b>
![Demo](share/demo.gif)


# License
[GNU GPLv3](LICENSE)