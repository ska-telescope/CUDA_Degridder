cmake_minimum_required(VERSION 3.10.2)

#! Base degridder project
project(degridder CXX CUDA)

#! Flags for the compiler
if ("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
	set(CMAKE_C_FLAGS "-Wno-unused-result -Wall -Wextra")
	set(CMAKE_C_FLAGS_DEBUG "-g -O0")
	set(CMAKE_C_FLAGS_RELEASE "-O3")
	set(CMAKE_CXX_FLAGS "-Wno-unused-result -Wall -Wextra")
	set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

#! Locate CUDA
find_package(CUDA REQUIRED)

set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS};
	-std=c++11 -arch=sm_60 -Xptxas -v -O3 -restrict -D_FORCE_INLINES -D_DEBUG -lineinfo
)

#! build and link degridder
cuda_add_executable(degridder main.cpp degridder.cu)
target_link_libraries(degridder m cufft) # math and CUDA FFT




#! Unit testing for degridder
project(tests CXX CUDA)

# find_package(GTest REQUIRED)
# include_directories(${GTEST_INCLUDE_DIRS})

# cuda_add_executable(tests unit_testing.cpp degridder.cu)
# target_link_libraries(tests ${GTEST_LIBRARIES} pthread m cufft)

cuda_add_executable(tests unit_testing.cpp degridder.cu)
target_link_libraries(tests m cufft)
