cmake_minimum_required(VERSION 3.16)
project(go2_deploy_cpp)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)


set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/library/thirdparty/libtorch/share/cmake")
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
