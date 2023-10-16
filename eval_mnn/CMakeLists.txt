cmake_minimum_required(VERSION 3.0)
project(mnist_mnn)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(run run.cpp)

# link
target_link_libraries(run MNN)