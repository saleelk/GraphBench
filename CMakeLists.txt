cmake_minimum_required(VERSION 3.21)

# ---------------------------------------------------------------------------
# Backend selection: HIP (default) or CUDA (-DUSE_CUDA=ON)
# ---------------------------------------------------------------------------
option(USE_CUDA "Build against CUDA instead of HIP/ROCm" OFF)

if(USE_CUDA)
  project(GraphBench LANGUAGES CXX CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # Default to native arch; override with -DCMAKE_CUDA_ARCHITECTURES=...
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
else()
  project(GraphBench LANGUAGES CXX HIP)
  set(CMAKE_HIP_STANDARD 17)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  find_package(hip REQUIRED CONFIG PATHS
    $ENV{ROCM_PATH} /opt/rocm
    $ENV{ROCM_PATH}/hip /opt/rocm/hip)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
if(USE_CUDA)
  # Treat the .cpp as CUDA source so nvcc compiles it.
  set_source_files_properties(graph_bench.cpp PROPERTIES LANGUAGE CUDA)
  add_executable(graph_bench graph_bench.cpp)
  target_compile_options(graph_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2 -x cu>)
else()
  add_executable(graph_bench graph_bench.cpp)
  target_link_libraries(graph_bench PRIVATE hip::host)
  target_compile_options(graph_bench PRIVATE -O2)
endif()
