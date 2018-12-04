cmake_minimum_required(VERSION 2.8.4) 

if (WIN32)
  set(OpenCL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../opencl-headers-msvc")
  set(OpenCL_LIBRARY     "${CMAKE_SOURCE_DIR}/../opencl-lib/OpenCL.lib")
endif (WIN32)

set(CMAKE_CXX_STANDARD 11)

project(unsharp_mask)

find_package(OpenCL REQUIRED)

add_executable(unsharp_mask unsharp_mask.cpp unsharp_mask.hpp blur.hpp blur.cl add_weighted.hpp add_weighted.cl ppm.hpp)

target_include_directories(unsharp_mask PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(unsharp_mask ${OpenCL_LIBRARY})

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT unsharp_mask)

string(REPLACE "-Od" "-O1" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string (REGEX REPLACE "/RTC(su|[1su])" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")




