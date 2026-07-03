cmake_minimum_required(VERSION 3.20)

project(StaticShadowTreeEncoder LANGUAGES CXX)

add_library(static_shadow_tree_encoder SHARED
    static_shadow_tree_encoder.cpp
)

target_compile_features(static_shadow_tree_encoder PRIVATE cxx_std_17)
target_compile_definitions(static_shadow_tree_encoder PRIVATE STATIC_SHADOW_TREE_ENCODER_BUILD)

if(MSVC)
    target_compile_options(static_shadow_tree_encoder PRIVATE /O2 /W4 /permissive-)
else()
    target_compile_options(static_shadow_tree_encoder PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()
