cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_PREFIX_PATH "C:/Users/maxfi/Desktop/libtorch")
project(MyCUDAProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

include_directories(include)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

add_executable(MyCUDAExecutable src/main.cpp cuda/kernel.cu)
target_link_libraries(MyCUDAExecutable "${TORCH_LIBRARIES}" ${CUDA_LIBRARIES})

# Set this 
set_property(TARGET MyCUDAExecutable PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
