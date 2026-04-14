cmake_minimum_required(VERSION 3.21)
project(GraphBench LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)

find_package(hip REQUIRED)

add_executable(graph_bench graph_bench.cpp)
target_link_libraries(graph_bench PRIVATE hip::host)
target_compile_options(graph_bench PRIVATE -O2)
