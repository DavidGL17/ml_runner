cmake_minimum_required(VERSION 3.16)

project(ML_Library)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")

include_directories(include)

add_executable(ml_model 
    include/layers/dense.h
    include/layers/conv.h
    include/layer.h
    include/model.h
    src/layers/conv.c
    src/layers/dense.c 
    src/model.c
    src/model_weights.h
    src/main.c
)
