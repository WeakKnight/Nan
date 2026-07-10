cmake_minimum_required(VERSION 3.18)
project(vertex_baking_utils LANGUAGES CXX)

add_library(
    vertex_baking_utils
    SHARED
    vertex_baking_utils.cpp
    tinybvh_visibility.cpp
)
target_compile_features(vertex_baking_utils PRIVATE cxx_std_17)
target_include_directories(vertex_baking_utils PRIVATE "${CMAKE_CURRENT_LIST_DIR}/third_party/tinybvh")
find_package(Threads REQUIRED)
target_link_libraries(vertex_baking_utils PRIVATE Threads::Threads)

option(VBAKE_TINYBVH_AVX2 "Enable experimental TinyBVH BVH8_CPU traversal on x86-64" OFF)
if(VBAKE_TINYBVH_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64)$")
    if(MSVC)
        set_source_files_properties(
            tinybvh_visibility.cpp
            PROPERTIES
                COMPILE_OPTIONS "/arch:AVX2"
                COMPILE_DEFINITIONS "VBAKE_TINYBVH_AVX2=1"
        )
    else()
        set_source_files_properties(
            tinybvh_visibility.cpp
            PROPERTIES
                COMPILE_OPTIONS "-mavx2;-mfma"
                COMPILE_DEFINITIONS "VBAKE_TINYBVH_AVX2=1"
        )
    endif()
    message(STATUS "vertex_baking_utils: TinyBVH BVH8_CPU enabled")
else()
    message(STATUS "vertex_baking_utils: TinyBVH portable BVH only")
endif()

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
    message(STATUS "vertex_baking_utils: Eigen not found; using built-in native solvers")
endif()

if(MSVC)
    target_compile_options(vertex_baking_utils PRIVATE /W4)
else()
    target_compile_options(vertex_baking_utils PRIVATE -Wall -Wextra -Wpedantic)
endif()
