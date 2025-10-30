cmake_minimum_required(VERSION 3.18)
project(glmmrGPU LANGUAGES CXX CUDA VERSION 1.0.0)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(ENABLE_TESTS "Build tests" OFF)
option(ENABLE_BENCHMARKS "Build benchmarks" OFF)

# Set standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# CUDA architectures - auto-detect or manual
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
    message(STATUS "Detected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Find packages
find_package(CUDAToolkit REQUIRED)

set(EIGEN3_INCLUDE_DIR "C:/cpp/eigen/eigen-3.4.0" CACHE PATH "Path to Eigen3")

if(NOT EXISTS ${EIGEN3_INCLUDE_DIR}/Eigen/Dense)
    message(FATAL_ERROR "Eigen3 not found at ${EIGEN3_INCLUDE_DIR}. Please set EIGEN3_INCLUDE_DIR")
endif()

message(STATUS "Using Eigen3 from: ${EIGEN3_INCLUDE_DIR}")

# Create library (reusable)
add_library(cuda_cholesky_lib
    src/cuda_cholesky.cu
)

target_include_directories(cuda_cholesky_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(cuda_cholesky_lib PUBLIC
    CUDA::cudart
    CUDA::cusolver
    CUDA::cublas
)

set_target_properties(cuda_cholesky_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Create executable
add_executable(glmmrGPU src/main.cpp)
target_link_libraries(glmmrGPU PRIVATE cuda_cholesky_lib)

get_target_property(SOURCES glmmrGPU SOURCES)
message(STATUS "Target sources:")
foreach(src ${SOURCES})
    message(STATUS "  ${src}")
endforeach()

# Installation
include(GNUInstallDirs)
install(TARGETS glmmrGPU cuda_cholesky_lib
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES include/cuda_cholesky.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)