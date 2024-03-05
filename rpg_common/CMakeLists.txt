project(rpg_common)
cmake_minimum_required(VERSION 2.8.3)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)
set(OpenCV_DIR /usr/share/OpenCV)
find_package(OpenCV REQUIRED)

if(${OpenCV_VERSION} STREQUAL "3.2.0")
  message(STATUS "OpenCV version ${OpenCV_VERSION}")
else()
  message(WARNING "OpenCV version ${OpenCV_VERSION}, OpenCV version must be 3.2.0")
endif()

add_definitions(-std=c++11)

FILE(GLOB_RECURSE HEADERS "include/*")

list(APPEND SOURCES src/fs.cpp)

cs_add_library(${PROJECT_NAME} ${HEADERS} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})

cs_install()
cs_export()
