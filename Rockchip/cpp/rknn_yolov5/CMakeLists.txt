cmake_minimum_required(VERSION 3.0.0)
project(rknn-yolov5)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
add_definitions(-w)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} /usr/include/rga 3rdparty/librknn_api/include include)

add_executable(rknn-yolov5 csrc/single_thread/detect.cpp)
target_link_libraries(rknn-yolov5 PRIVATE dl rt pthread rknnrt rga ${OpenCV_LIBRARIES})

add_executable(rknn-yolov5-mt csrc/multi_thread/detect-mt.cpp)
target_link_libraries(rknn-yolov5-mt PRIVATE dl rt pthread rknnrt rga ${OpenCV_LIBRARIES})