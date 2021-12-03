cmake_minimum_required(VERSION 3.8) 

set(OpenCL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/dependencies/opencl-headers-msvc")
set(OpenCL_LIBRARY     "${CMAKE_SOURCE_DIR}/dependencies/opencl-lib/OpenCL.lib")

set(CMAKE_CXX_STANDARD 11)

project(unsharp_mask LANGUAGES CXX)

file(GLOB_RECURSE headerFiles "headers/*.h" "headers/*.hpp")
file(GLOB_RECURSE sourceFiles "sources/*.cpp" "sources/*.cl")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/headers)

find_package(OpenCL REQUIRED)

add_executable(${PROJECT_NAME} ${headerFiles} ${sourceFiles})

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARY})

set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

string(REPLACE "-Od" "-O1" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string (REGEX REPLACE "/RTC(su|[1su])" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")