cmake_minimum_required(VERSION 3.20)

project(StaticShadowTreeEncoder LANGUAGES CXX)

add_library(static_shadow_tree_encoder SHARED
    static_shadow_tree_encoder.cpp
)

add_library(surface_probe_sampler SHARED
    surface_probe_sampler.cpp
)

add_executable(surface_probe_sampler_tests
    surface_probe_sampler_tests.cpp
)

target_compile_features(static_shadow_tree_encoder PRIVATE cxx_std_17)
target_compile_definitions(static_shadow_tree_encoder PRIVATE STATIC_SHADOW_TREE_ENCODER_BUILD)

target_compile_features(surface_probe_sampler PRIVATE cxx_std_17)
target_compile_definitions(surface_probe_sampler PRIVATE
    SURFACE_PROBE_SAMPLER_BUILD
    NOMINMAX
)
target_include_directories(surface_probe_sampler PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cyCodeBase"
)
target_compile_features(surface_probe_sampler_tests PRIVATE cxx_std_17)
target_link_libraries(surface_probe_sampler_tests PRIVATE surface_probe_sampler)

enable_testing()
add_test(NAME surface_probe_sampler_tests COMMAND surface_probe_sampler_tests)

if(MSVC)
    target_compile_options(static_shadow_tree_encoder PRIVATE /O2 /W4 /permissive-)
    target_compile_options(surface_probe_sampler PRIVATE /O2 /W4 /permissive-)
    target_compile_options(surface_probe_sampler_tests PRIVATE /O2 /W4 /permissive-)
else()
    target_compile_options(static_shadow_tree_encoder PRIVATE -O3 -Wall -Wextra -Wpedantic)
    target_compile_options(surface_probe_sampler PRIVATE -O3 -Wall -Wextra -Wpedantic)
    target_compile_options(surface_probe_sampler_tests PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()
