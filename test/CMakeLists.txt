cmake_minimum_required(VERSION 3.18)
project(cublox_perf_standalone LANGUAGES CXX CUDA)

set(CUBLOX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." CACHE PATH "cublox package root")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
endif()

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>
                    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                    $<$<COMPILE_LANGUAGE:CXX>:-O3>
                    $<$<COMPILE_LANGUAGE:CXX>:-g>)

find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)

file(GLOB_RECURSE cublox_cpp_srcs CONFIGURE_DEPENDS
  "${CUBLOX_ROOT}/src/*.cpp")
file(GLOB_RECURSE cublox_cuda_srcs CONFIGURE_DEPENDS
  "${CUBLOX_ROOT}/src/*.cu")
file(GLOB_RECURSE cublox_hdrs CONFIGURE_DEPENDS
  "${CUBLOX_ROOT}/include/*.h"
  "${CUBLOX_ROOT}/include/*.hpp"
  "${CUBLOX_ROOT}/include/*.cuh")

list(REMOVE_ITEM cublox_cpp_srcs
  "${CUBLOX_ROOT}/src/main.cpp")

add_library(cublox STATIC
  ${cublox_cpp_srcs}
  ${cublox_cuda_srcs}
  ${cublox_hdrs})

target_include_directories(cublox PUBLIC
  "$<BUILD_INTERFACE:${CUBLOX_ROOT}/include>"
  ${EIGEN3_INCLUDE_DIRS})

set_target_properties(cublox PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(cublox PUBLIC CUDA::cudart)

add_executable(occupancy_grid_perf_test
  test_occupancy_grid_perf.cpp)
target_link_libraries(occupancy_grid_perf_test PRIVATE
  cublox
  GTest::gtest_main)

enable_testing()
add_test(NAME occupancy_grid_perf COMMAND occupancy_grid_perf_test)
