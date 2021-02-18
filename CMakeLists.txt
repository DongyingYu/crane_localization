cmake_minimum_required(VERSION 3.0.2)


add_compile_options(-std=c++11)
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

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/g2o
)


add_subdirectory(test)