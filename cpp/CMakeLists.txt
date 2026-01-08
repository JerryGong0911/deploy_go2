cmake_minimum_required(VERSION 3.16)
project(go2_deploy_cpp)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(JETSON_LIBTORCH_DIR "$ENV{HOME}/.local/lib/python3.8/site-packages/torch/")
set(LIBTORCH_DIR "${CMAKE_CURRENT_LIST_DIR}/library/thirdparty/libtorch")

if(CMAKE_VERSION VERSION_LESS "3.18")
    set(TORCH_CUDA_ARCH_LIST "8.7")
endif()

if(EXISTS "/etc/nv_tegra_release")
    set(CMAKE_PREFIX_PATH "${JETSON_LIBTORCH_DIR};${CMAKE_PREFIX_PATH}")
else()
    set(CMAKE_PREFIX_PATH "${LIBTORCH_DIR};${CMAKE_PREFIX_PATH}")
endif()
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

file(GLOB SOURCES "src/*.cpp")
add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE 
                            "${CMAKE_CURRENT_LIST_DIR}/inc"
                            "/usr/local/include/ddscxx"
                            "/usr/local/include/iceoryx/v2.0.2")
find_package(yaml-cpp REQUIRED)
target_link_libraries(${PROJECT_NAME} unitree_sdk2 ddsc ddscxx yaml-cpp pthread ${TORCH_LIBRARIES})
