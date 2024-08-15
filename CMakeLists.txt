cmake_minimum_required(VERSION 3.5.0)
project(CudaImageProcessing VERSION 0.1.0 LANGUAGES C CXX CUDA)
include_directories(${PROJECT_SOURCE_DIR}/include)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
include_directories(${CUDA_INCLUDE_DIRS})

set(KernelSources
    kernels/imageTools.cu
)


add_executable(CudaImageProcessing main.cpp ${KernelSources})
set_target_properties(CudaImageProcessing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(CudaImageProcessing ${CUDA_LIBRARIES} cufft)