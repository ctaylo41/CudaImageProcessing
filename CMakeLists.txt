cmake_minimum_required(VERSION 3.5.0)
project(CudaImageProcessing VERSION 0.1.0 LANGUAGES C CXX CUDA)
include_directories(${PROJECT_SOURCE_DIR}/include)
set(KernelSources
    kernels/imageLoad.cu
)


add_executable(CudaImageProcessing main.cpp ${KernelSources})
set_target_properties(CudaImageProcessing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


