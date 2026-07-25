cmake_minimum_required(VERSION 3.16)

project(ML_Library)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")

# Include directories
include_directories(include)
include_directories(src)

# Core library containing the logic
add_library(ml_lib
    src/model.c
    src/layers/dense.c
)

# Main executable
add_executable(ml_model
    src/main.c
)
target_link_libraries(ml_model ml_lib)

# Testing setup
enable_testing()
add_executable(ml_test
    tests/test_main.c
)
target_link_libraries(ml_test ml_lib)
add_test(NAME dummy_test COMMAND ml_test)
