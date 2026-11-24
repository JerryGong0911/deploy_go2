cmake_minimum_required(VERSION 3.16)
project(go2_deploy_cpp)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ONNX Runtime (prebuilt package: include/ + lib/libonnxruntime.so).
# Use the x64 package on PC and the aarch64 package on Jetson; override with
#   cmake .. -DORT_DIR=/path/to/onnxruntime
set(ORT_DIR "${CMAKE_CURRENT_LIST_DIR}/library/thirdparty/onnxruntime"
    CACHE PATH "ONNX Runtime root directory")
if(NOT EXISTS "${ORT_DIR}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime not found in ORT_DIR=${ORT_DIR}")
endif()
find_library(ORT_LIB onnxruntime PATHS "${ORT_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
message(STATUS "ONNX Runtime: ${ORT_LIB}")

file(GLOB SOURCES "src/*.cpp")
add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
                            "${CMAKE_CURRENT_LIST_DIR}/inc"
                            "${ORT_DIR}/include"
                            "/usr/local/include/ddscxx"
                            "/usr/local/include/iceoryx/v2.0.2")
find_package(yaml-cpp REQUIRED)
target_link_libraries(${PROJECT_NAME} unitree_sdk2 ddsc ddscxx yaml-cpp pthread ${ORT_LIB})
set_target_properties(${PROJECT_NAME} PROPERTIES
                      BUILD_RPATH "${ORT_DIR}/lib"
                      INSTALL_RPATH "${ORT_DIR}/lib")
