cmake_minimum_required(VERSION 3.0)
project(mnist_mnn)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# source files
FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/run.cpp)

# target
add_executable(run ${SRCS})

# link
if (MSVC)
    target_link_libraries(run MNN)
else()
    target_link_libraries(run MNN MNN_Express)
endif()
