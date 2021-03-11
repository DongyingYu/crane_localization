cmake_minimum_required(VERSION 3.0.2)


add_compile_options(-std=c++14)
add_compile_options(-g)
add_compile_options(-O0)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Library Status: ")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3:")
message(STATUS "    include path: ${EIGEN3_INCLUDE_DIR}")

# find yaml cpp lib
find_library(YAML_CPP_LIBRARIES yaml-cpp)
if(NOT YAML_CPP_LIBRARIES)
  # If yaml-cpp not found in the system, try finding it as a user CMake-generated project
  find_package(yaml-cpp REQUIRED)
  include_directories(${YAML_CPP_INCLUDE_DIRS})
endif(NOT YAML_CPP_LIBRARIES)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/g2o
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/g2o/build
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/DBoW2
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/DBoW2/DBoW2
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/DBoW2/DUtils
)

add_subdirectory(test)
