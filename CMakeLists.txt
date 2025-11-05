cmake_minimum_required(VERSION 3.18)
project(glmmrGPU LANGUAGES CXX CUDA VERSION 1.0.0)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
    message(STATUS "MSVC detected: enabling /bigobj flag")
endif()

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(ENABLE_TESTS "Build tests" OFF)
option(ENABLE_BENCHMARKS "Build benchmarks" OFF)

# Set standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(MSVC)
    set(CXX_RELEASE_FLAGS "/O2 /DNDEBUG")
    set(CUDA_RELEASE_FLAGS "-O3 -use_fast_math")
else()
    set(CXX_RELEASE_FLAGS "-O3 -march=native -DNDEBUG")
    set(CUDA_RELEASE_FLAGS "-O3 -use_fast_math")
endif()

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

# Create directory for external dependencies
set(DEPS_DIR ${CMAKE_BINARY_DIR}/_deps)
file(MAKE_DIRECTORY ${DEPS_DIR})

# ===== Download and extract Eigen =====
set(EIGEN_DIR ${DEPS_DIR}/eigen)
if(NOT EXISTS ${EIGEN_DIR}/Eigen/Dense)
    message(STATUS "Downloading Eigen...")
    file(DOWNLOAD
        https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        ${DEPS_DIR}/eigen.tar.gz
        SHOW_PROGRESS
    )
    
    message(STATUS "Extracting Eigen...")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${DEPS_DIR}/eigen.tar.gz
        WORKING_DIRECTORY ${DEPS_DIR}
    )
    
    file(RENAME ${DEPS_DIR}/eigen-3.4.0 ${EIGEN_DIR})
endif()

# ===== Download and extract Boost ====
set(BOOST_EXTRACT_DIR ${DEPS_DIR}/boost-extract)
set(BOOST_DIR ${DEPS_DIR}/boost)

if(NOT EXISTS ${BOOST_DIR}/boost/version.hpp)
    message(STATUS "Downloading Boost (150MB, may take a few minutes)...")
    
    # Try SourceForge (more reliable for pre-built archives)
    file(DOWNLOAD
        https://sourceforge.net/projects/boost/files/boost/1.83.0/boost_1_83_0.tar.gz
        ${DEPS_DIR}/boost.tar.gz
        SHOW_PROGRESS
        TIMEOUT 600
    )
    
    message(STATUS "Extracting Boost...")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${DEPS_DIR}/boost.tar.gz
        WORKING_DIRECTORY ${DEPS_DIR}
    )
    
    # Check what was extracted
    file(GLOB BOOST_EXTRACTED "${DEPS_DIR}/boost_*")
    if(BOOST_EXTRACTED)
        list(GET BOOST_EXTRACTED 0 BOOST_EXTRACTED_DIR)
        message(STATUS "Found extracted Boost at: ${BOOST_EXTRACTED_DIR}")
        
        # Rename to standard location
        if(EXISTS ${BOOST_EXTRACTED_DIR})
            file(RENAME ${BOOST_EXTRACTED_DIR} ${BOOST_DIR})
        endif()
    endif()
endif()

# Verify Boost structure and find boost/version.hpp
message(STATUS "Looking for Boost headers...")

# Check various possible locations
set(POSSIBLE_BOOST_PATHS
    ${BOOST_DIR}
    ${BOOST_DIR}/boost
    ${DEPS_DIR}/boost_1_83_0
)

set(BOOST_FOUND FALSE)
foreach(path ${POSSIBLE_BOOST_PATHS})
    if(EXISTS ${path}/boost/version.hpp)
        set(BOOST_INCLUDE_DIR ${path})
        set(BOOST_FOUND TRUE)
        message(STATUS "✓ Found boost/version.hpp at: ${BOOST_INCLUDE_DIR}")
        break()
    endif()
endforeach()

if(NOT BOOST_FOUND)
    # Debug: List what we actually have
    message(STATUS "Searching in ${DEPS_DIR}:")
    file(GLOB_RECURSE VERSION_FILES "${DEPS_DIR}/*/version.hpp")
    foreach(file ${VERSION_FILES})
        message(STATUS "  Found version.hpp at: ${file}")
    endforeach()
    
    message(FATAL_ERROR "Could not find boost/version.hpp in any expected location")
endif()

# Verify Eigen
if(NOT EXISTS ${EIGEN_DIR}/Eigen/Dense)
    message(FATAL_ERROR "Eigen/Dense not found at ${EIGEN_DIR}")
else()
    message(STATUS "✓ Found Eigen/Dense at: ${EIGEN_DIR}")
endif()

find_package(OpenMP)

# Create library (reusable)
add_library(cuda_cholesky_lib
    src/cuda_cholesky.cu
)

target_include_directories(cuda_cholesky_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN_DIR}
    ${BOOST_INCLUDE_DIR}
)

target_link_libraries(cuda_cholesky_lib PUBLIC
    CUDA::cudart_static
    CUDA::cusolver
    CUDA::cublas
)

set_target_properties(cuda_cholesky_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Create executable
add_executable(glmmrGPU src/main.cpp)

set_target_properties(glmmrGPU PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(glmmrGPU PRIVATE 
    cuda_cholesky_lib
    CUDA::cudart_static
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(glmmrGPU PUBLIC OpenMP::OpenMP_CXX)
endif()

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
