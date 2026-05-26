cmake_minimum_required(VERSION 3.18)
project(cublox LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Language / build flags
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Default to a sensible spread of modern arches; user can override with
#   -DCMAKE_CUDA_ARCHITECTURES="86" (or "native" on CMake >=3.24 + present GPU).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
endif()

# Cross-TU device linking (so device functions can live in one .cu and be
# called from another). Cheap to enable up front; very annoying to add later.
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>
                    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                    $<$<COMPILE_LANGUAGE:CXX>:-O3>
                    $<$<COMPILE_LANGUAGE:CXX>:-g>)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

# CUDAToolkit gives us the `CUDA::cudart`, `CUDA::cublas`, etc imported
# targets without leaking absolute include paths. enable_language(CUDA)
# above already verified nvcc is on PATH.
find_package(CUDAToolkit REQUIRED)

set(ros_deps
    rclcpp
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    tf2_ros
)

# ---------------------------------------------------------------------------
# Library target
# ---------------------------------------------------------------------------
file(GLOB_RECURSE cublox_cpp_srcs CONFIGURE_DEPENDS
    src/*.cpp)
file(GLOB_RECURSE cublox_cuda_srcs CONFIGURE_DEPENDS
    src/*.cu)
file(GLOB_RECURSE cublox_hdrs CONFIGURE_DEPENDS
    include/*.h
    include/*.hpp
    include/*.cuh)
list(REMOVE_ITEM cublox_cpp_srcs
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

add_library(${PROJECT_NAME} STATIC
    ${cublox_cpp_srcs}
    ${cublox_cuda_srcs}
    ${cublox_hdrs})

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIRS})

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)

ament_target_dependencies(${PROJECT_NAME} PUBLIC ${ros_deps})

target_link_libraries(${PROJECT_NAME} PUBLIC
    yaml-cpp
    CUDA::cudart)

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------
add_executable(${PROJECT_NAME}_node src/main.cpp)
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME})
ament_target_dependencies(${PROJECT_NAME}_node ${ros_deps}
    ament_index_cpp)

# ---------------------------------------------------------------------------
# Install / export
# ---------------------------------------------------------------------------
install(TARGETS ${PROJECT_NAME}
    EXPORT export_${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)

install(TARGETS ${PROJECT_NAME}_node
    RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
    DESTINATION include/)

install(DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config)

install(FILES share/cfg.rviz
    DESTINATION share/${PROJECT_NAME})

install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_include_directories(include)
ament_export_dependencies(${ros_deps} Eigen3 yaml-cpp CUDAToolkit)

ament_package()
