cmake_minimum_required(VERSION 3.0.2)


add_compile_options(-std=c++11)
add_compile_options(-g)
add_compile_options(-O0)

find_package(OpenCV REQUIRED)
message(STATYS "OpenCV Library Status: ")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
)


add_subdirectory(test)