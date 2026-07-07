cmake_minimum_required(VERSION 3.18)
project(vertex_baking_utils LANGUAGES CXX)

add_library(vertex_baking_utils SHARED vertex_baking_utils.cpp)
target_compile_features(vertex_baking_utils PRIVATE cxx_std_17)

find_path(
    VBAKE_EIGEN_INCLUDE_DIR
    NAMES Eigen/SparseCholesky
    PATHS
        "$ENV{EIGEN3_INCLUDE_DIR}"
        "$ENV{EIGEN_INCLUDE_DIR}"
        "${CMAKE_CURRENT_LIST_DIR}/../third_party/eigen"
)
if(VBAKE_EIGEN_INCLUDE_DIR)
    target_include_directories(vertex_baking_utils PRIVATE "${VBAKE_EIGEN_INCLUDE_DIR}")
    target_compile_definitions(vertex_baking_utils PRIVATE VBAKE_USE_EIGEN=1)
    message(STATUS "vertex_baking_utils: using Eigen at ${VBAKE_EIGEN_INCLUDE_DIR}")
else()
    message(STATUS "vertex_baking_utils: Eigen not found; using built-in dense Cholesky fallback")
endif()

if(MSVC)
    target_compile_options(vertex_baking_utils PRIVATE /W4)
else()
    target_compile_options(vertex_baking_utils PRIVATE -Wall -Wextra -Wpedantic)
endif()
