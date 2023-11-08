cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_PREFIX_PATH "C:/Users/maxfi/Desktop/libtorch")
project(MyCUDAProject LANGUAGES CXX CUDA)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

include_directories(include)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

add_executable(MyCUDAExecutable src/main.cpp cuda/kernel.cu)
target_link_libraries(MyCUDAExecutable "${TORCH_LIBRARIES}" ${CUDA_LIBRARIES})

# Set this 
set_property(TARGET MyCUDAExecutable PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Define the GPU architecture(s) to compile for. By setting this, you're instructing NVCC (the CUDA compiler) 
# to generate code optimized for the compute capability of your specific GPU model. For me the compute capability is 7.5 or 75
set_property(TARGET MyCUDAExecutable PROPERTY CUDA_ARCHITECTURES 75)